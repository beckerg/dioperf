/*
 * This tool generates various r/w workloads and measures and graphs
 * the throughput and latency of the device under test.
 *
 * cc -O2 -D_GNU_SOURCE -g dioperf.c -o dioperf -lm -lpthread
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include <unistd.h>
#include <getopt.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <sysexits.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/resource.h>

#include <immintrin.h>
#include <x86intrin.h>

#if __FreeBSD__
#include <sys/sysctl.h>
#endif

#define HAVE_RDTSC          (__has_builtin(__builtin_ia32_rdtsc))
#define HAVE_RDTSCP         (__has_builtin(__builtin_ia32_rdtscp))
#define HAVE_RDRAND64       (__has_builtin(__builtin_ia32_rdrand64_step))
#define HAVE_PAUSE          (__has_builtin(__builtin_ia32_pause))

/* By default we use rdtsc() to measure timing intervals, but if
 * it's not available we'll fall back to using clock_gettime().
 */
#ifndef USE_CLOCK
#define USE_CLOCK           (!HAVE_RDTSC)
#endif

/* Number of buckets in latency histogram ((1u << BKT_SHIFT) cycles per bucket).
 */
#define BKT_MAX             (1024 * 1024)

#if USE_CLOCK
#define BKT_SHIFT           (0)
#else
#define BKT_SHIFT           (7)
#endif

#ifndef __aligned
#define __aligned(_size)    __attribute__((__aligned__(_size)))
#endif

#define NELEM(_arr)         (sizeof((_arr)) / sizeof((_arr)[0]))

struct tdargs {
    bool        random __aligned(64);
    bool        read;
    int         fd;
    size_t      iosz;
    u_long     *opsv;
    u_long     *bktv;
    pthread_t   thr;
    u_int       tid;
    size_t      bktvsz;
    u_long      opstot;
    long        usecs;
};

struct latdat {
    double pct;
    u_long thresh;
    double latency;
    u_long hits;
};

struct latres {
    struct latdat v[16];
    double latavg_latency;
    u_long latavg_hits;
    u_long peakhits;
    u_int c;
    const char *name;
};

const char *progname;
const char *prefix;
int verbosity;
size_t riosz;
size_t wiosz;
bool rsequential;
bool wsequential;
u_int rjobs;
u_int wjobs;
off_t partend;
off_t partsz;
uint64_t itv_freq;
uint64_t itv_alpha;
uint64_t itv_omega;
double usecs_per_cycle;
time_t duration;
char bnfile[128];

struct latres latresv[2];

char percentilestrv[] = "50,90,95,99";
char *percentilestr = percentilestrv;
double *percentilev;
u_int percentilec;

pthread_barrier_t rwbarrier;
struct tdargs *tdargsv;

__thread uint64_t state[2];


static inline uint64_t
rotl(const uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

static inline void
xoroshiro128plus_init(uint64_t *s, uint64_t seed)
{
    uint64_t z;

    z = (seed += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    s[0] = z ^ (z >> 31);

    z = (seed += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    s[1] = z ^ (z >> 31);
}

static inline uint64_t
xoroshiro128plus(uint64_t *s)
{
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); /* a, b */
    s[1] = rotl(s1, 36); /* c */

    return result;
}

static inline __attribute__((const))
u_int
ilog2(unsigned long long n)
{
    return (NBBY * sizeof(n) - 1) - __builtin_clzll(n);
}

/* Time interval measurement abstractions...
 */
static inline uint64_t
itv_cycles(void)
{
#if HAVE_RDTSC && !USE_CLOCK
    return __rdtsc();
#else
    struct timespec now;

    clock_gettime(CLOCK_MONOTONIC, &now);

    return now.tv_sec * 1000000000 + now.tv_nsec;
#endif
}

static inline uint64_t
itv_start(void)
{
#if HAVE_RDTSC && !USE_CLOCK
    __asm__ volatile ("cpuid" ::: "eax","ebx","ecx","edx","memory");
#endif

    return itv_cycles();
}

static inline uint64_t
itv_stop(void)
{
#if HAVE_RDTSCP && !USE_CLOCK
    uint aux;

    return __rdtscp(&aux);
#else
    return itv_cycles();
#endif
}

static inline double
itv_to_usecs(uint64_t itv)
{
    return (itv * usecs_per_cycle);
}

static inline time_t
itv_to_seconds(uint64_t itv)
{
    return (itv * usecs_per_cycle) / 1000000;
}

static void
eprint(int xerrno, const char *fmt, ...)
{
    char msg[256];
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);

    fprintf(stderr, "%s: %s%s%s\n",
            progname, msg,
            xerrno ? ": " : "",
            xerrno ? strerror(xerrno) : "");
}

void
syntax(const char *fmt, ...)
{
    char msg[256];
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);

    fprintf(stderr, "%s: %s, use -h for help\n", progname, msg);
}

void *
super_alloc(size_t sz)
{
    int flags = MAP_ANONYMOUS | MAP_PRIVATE;
    int prot = PROT_READ | PROT_WRITE;
    int super = 0;
    void *mem;

#if __FreeBSD__
    super = MAP_ALIGNED_SUPER;
#elif __linux__
    super = MAP_HUGETLB;
#endif

    sz = (sz + (2u << 20) - 1) & ~((2u << 20) - 1);

  again:
    mem = mmap(NULL, sz, prot, flags | super, -1, 0);
    if (mem == MAP_FAILED) {
        if (super) {
            super = 0;
            goto again;
        }

        eprint(errno, "mmap(%zu, %x, %x) failed", sz, prot, flags);
    }

    return mem;
}

int
super_free(void *addr, size_t len)
{
    return munmap(addr, len);
}

void *
test_main(void *arg)
{
    struct timeval tv_start, tv_stop, tv_diff;
    struct tdargs *a = arg;
    uint64_t itv_next;
    u_long elapsed;
    u_long *iobuf;
    uint64_t mask;
    off_t off;
    int i;

    xoroshiro128plus_init(state, rotl(itv_start(), a->tid % 64));

    iobuf = aligned_alloc(4096, roundup(a->iosz, 4096));
    if (!iobuf)
        abort();

    if (!a->read) {
        for (i = 0; i < a->iosz / sizeof(*iobuf); ++i)
            iobuf[i] = xoroshiro128plus(state);
    }

    memset(a->bktv, 0, a->bktvsz);

    mask = ~((1ul << ilog2(a->iosz)) - 1);
    off = (xoroshiro128plus(state) % partsz) & mask;

    pthread_barrier_wait(&rwbarrier);
    gettimeofday(&tv_start, NULL);

    itv_next = itv_alpha + itv_freq;
    elapsed = 0;

    /* TODO: Use locking to prevent overlapped I/O...
     */
    while (itv_next < itv_omega) {
        uint64_t tstart, tstop;
        suseconds_t dt;
        ssize_t cc;

        tstart = itv_start();

        if (a->read)
            cc = pread(a->fd, iobuf, a->iosz, off);
        else
            cc = pwrite(a->fd, iobuf, a->iosz, off);

        tstop = itv_stop();

        if (tstop >= itv_next) {
            itv_next += itv_freq;
            elapsed++;
        }

        dt = (tstop - tstart) >> BKT_SHIFT;

        if (dt > BKT_MAX - 1)
            dt = BKT_MAX - 1;

        ++a->opsv[elapsed];
        ++a->bktv[dt];

        if (cc != a->iosz) {
            char *msg = cc == -1 ? strerror(errno) : "eof";

            fprintf(stderr, "%s: cc %ld (%d %s), iosz %zu, off %zu\n",
                    progname, cc, errno, cc > 0 ? "short" : msg, a->iosz, off);
            exit(1);
        }

        if (a->random) {
            off = (xoroshiro128plus(state) % partsz) & mask;
        } else {
            off += a->iosz;
            if (off >= partsz)
                off = 0;
        }
    }

    gettimeofday(&tv_stop, NULL);
    timersub(&tv_stop, &tv_start, &tv_diff);
    a->usecs = tv_diff.tv_sec * 1000000 + tv_diff.tv_usec;

    for (i = 0; i < duration; ++i)
        a->opstot += a->opsv[i];

    free(iobuf);

    pthread_exit(NULL);
}

void
report_latency(struct tdargs *a, u_int jobs, struct latres *r)
{
    double latsum, usecs;
    u_long latmin, latmax;
    u_long opstot, hits;
    u_long bytespersec;
    u_long *bktv;
    u_int i, j;
    int k;
    char fnlat[256];
    FILE *fp;

    if (jobs < 1)
        return;

    opstot = 0;
    fp = NULL;

    for (j = 0; j < jobs; ++j)
        opstot += a[j].opstot;

    for (k = 0; k < r->c; ++k)
        r->v[k].thresh = opstot * r->v[k].pct;

    /* Create a file and write out all the latency data.
     */
    if (prefix) {
        snprintf(fnlat, sizeof(fnlat), "%s.%clat", bnfile, r->name[0]);

        fp = fopen(fnlat, "w");
        if (!fp) {
            eprint(errno, "fopen(%s) failed", fnlat);
            return;
        }

        fprintf(fp, "#%9s %8s %8s %10s\n",
                "LATENCY", "HITS", "CENTILE", "PERCENTILE");
    }

    latmin = latmax = 0;
    bktv = a->bktv;
    latsum = 0;
    usecs = 0;
    hits = 0;

    for (i = 0; i < BKT_MAX; ++i) {
        for (j = 1; j < jobs; ++j) {
            bktv[i] += a[j].bktv[i];
            if (a[j].usecs > a->usecs)
                a->usecs = a[j].usecs;
        }

        if (bktv[i] == 0)
            continue;

        usecs = itv_to_usecs(i << BKT_SHIFT);

        if (bktv[i] > r->peakhits)
            r->peakhits = bktv[i];
        hits += bktv[i];

        if (latmin == 0)
            latmin = usecs;
        latsum += bktv[i] * usecs;

        for (k = r->c - 1; k >= 0; --k) {
            if (r->v[k].latency > 0)
                break;

            if (hits >= r->v[k].thresh) {
                r->v[k].latency = usecs;
                r->v[k].hits = bktv[i];
            }
        }

        if (!fp || r->c < 1)
            continue;

        if (r->v[r->c - 1].latency && verbosity < 2)
            continue;

        fprintf(fp, "%10.3lf %8lu %8lu", usecs, bktv[i], (hits * 100) / opstot);

        if (k >= 0)
            fprintf(fp, " %10.2lf\n", r->v[k].pct * 100);
        else
            fprintf(fp, " %10s\n", "-");
    }

    if (fp)
        fclose(fp);

    r->latavg_latency = latsum / hits;
    r->latavg_hits = bktv[ (u_long)(r->latavg_latency / usecs_per_cycle) >> BKT_SHIFT ];

    bytespersec = (opstot * a->iosz * 1000000) / a->usecs;
    latmax = usecs;

    printf("%12lu  %s avg latency (us)\n", (u_long)r->latavg_latency, r->name);
    printf("%12lu  %s min latency (us)\n", latmin, r->name);
    printf("%12lu  %s max latency (us)\n", latmax, r->name);

    printf("%12u  %s threads\n", jobs, r->name);
    printf("%12zu  %s I/O size\n", a->iosz, r->name);
    printf("%12lu  %s operations\n", opstot, r->name);
    printf("%12lu  %s avg ops/sec\n", (opstot * 1000000ul) / a->usecs, r->name);
    printf("%12lu  %s avg MiB/sec\n", bytespersec >> 20, r->name);
    printf("\n");
}

void
report_ops(struct tdargs *a, u_int jobs, struct latres *r, time_t t0)
{
    char fnops[256], fnplot[256];
    const char *wcolor = "orange";
    const char *rcolor = "cyan";
    const char *pcolor = "blue";
    char buf[128 + jobs * 16];
    const char *term = "png";
    int xtics, mxtics;
    FILE *fp;
    time_t i;
    u_int j;
    int k;

    if (!prefix)
        return;

    /* Create a file and write out all the time series data.
     */
    snprintf(fnops, sizeof(fnops), "%s.ops", bnfile);

    fp = fopen(fnops, "w");
    if (!fp) {
        eprint(errno, "fopen(%s) failed", fnops);
        return;
    }

    fprintf(fp, "#%11s %5s %7s %7s %7s",
            "TIME", "SECS", "ROPS", "WOPS", "RWOPS");

    for (j = 0; j < rjobs + wjobs; ++j)
        fprintf(fp, " %6u%s", j, (j < rjobs) ? "r" : "w");
    fprintf(fp, "\n");

    for (i = 0; i < duration; ++i) {
        u_long rops = 0, wops = 0;
        size_t pos = 0;
        int n;

        for (j = 0; j < jobs; ++j) {
            n = snprintf(buf + pos, sizeof(buf) - pos, " %7lu", a[j].opsv[i]);
            if (n < 1 || n >= sizeof(buf) - pos)
                abort();

            if (j < rjobs)
                rops += a[j].opsv[i];
            else
                wops += a[j].opsv[i];

            pos += n;
        }

        fprintf(fp, "%12ld %5ld %7lu %7lu %7lu%s\n",
                t0 + i, i, rops, wops, rops + wops, buf);
    }

    fclose(fp);

    /* Create a file and write out the gnuplot config file.
     */
    snprintf(fnplot, sizeof(fnplot), "%s.plot", bnfile);

    fp = fopen(fnplot, "w");
    if (!fp) {
        eprint(errno, "fopen(%s) failed", fnplot);
        return;
    }

    switch (duration / 1800) {
    case 0:
        if (duration > 180) {
            mxtics = 6;
            xtics = 60;
        } else {
            mxtics = 2;
            xtics = 10;
        }
        break;
    case 1:
        mxtics = 5;
        xtics = 300;
        break;
    default:
        mxtics = 10;
        xtics = 600;
        break;
    }

    fprintf(fp, "# Created on %s", ctime(&t0));

    fprintf(fp, "set output '%s.%s'\n", bnfile, term);
    fprintf(fp, "set term %s size 2048,1152\n", term);
    //fprintf(fp, "set size 1, 0.76\n");
    //fprintf(fp, "set origin 0, 0.24\n");
    fprintf(fp, "set autoscale\n");
    fprintf(fp, "set grid\n");
    fprintf(fp, "set ytics autofreq\n");

    fprintf(fp, "set multiplot layout %d,1 columnsfirst\n",
            rjobs && wjobs ? 3 : 2);

    fprintf(fp, "set title '%s operations per second'\n",
            rjobs && wjobs ? "r/w" : (rjobs ? "read" : "write"));

    fprintf(fp, "set xlabel 'seconds'\n");
    fprintf(fp, "set xtics 0, %d rotate by -30\n", xtics);
    fprintf(fp, "set mxtics %d\n", mxtics);

    fprintf(fp, "set ylabel 'ops'\n");
    fprintf(fp, "set mytics 2\n");

    fprintf(fp, "plot ");

    if (rjobs > 0) {
        fprintf(fp,
                "'%s' every ::1:::0 using ($2):($3) with lines "
                "lc rgb '%s' title 'readers %d, %s, %zu-bytes' %s",
                fnops, rcolor, rjobs,
                rsequential ? "sequential" : "random",
                riosz,
                wjobs ? "," : "\n");
    }

    if (wjobs > 0) {
        fprintf(fp,
                "'%s' every ::1:::0 using ($2):($4) with lines "
                "lc rgb '%s' title 'writers %d, %s, %zu-bytes' %s",
                fnops, wcolor, wjobs,
                wsequential ? "sequential" : "random",
                wiosz,
                rjobs ? "," : "\n");
    }

    if (rjobs && wjobs) {
        fprintf(fp,
                "'%s' every ::1:::0 using ($2):($5) with lines "
                "lc rgb 'green' title 'combined'\n",
                fnops);
    }

    fprintf(fp, "set xlabel 'buckets (usecs +/- %ldns)'\n",
            (long)(itv_to_usecs(1u << BKT_SHIFT) * 1000));
    fprintf(fp, "set xtics autofreq\n");
    fprintf(fp, "set mxtics 10\n");
    fprintf(fp, "set ylabel 'hits'\n");

    if (rjobs > 0) {
        fprintf(fp, "set title 'read latency histogram'\n");
        fprintf(fp, "set yrange [-%lu:]\n", r[0].peakhits / 10);

        for (k = 0; k < r[0].c; ++k) {
            if (!r[0].v[k].latency)
                continue;

            fprintf(fp, "set label %d '' at %.3lf,%lu point pointtype 3 "
                    "lc rgb '%s' front # %.2lfth percentile\n",
                    k + 1, r[0].v[k].latency, r[0].v[k].hits,
                    pcolor, r[0].v[k].pct * 100);
        }

        fprintf(fp, "set label %d '' at %.3lf,%lu point pointtype 1 "
                "lc rgb '%s' front # average latency\n",
                k + 1, r[0].latavg_latency, r[0].latavg_hits, pcolor);

        fprintf(fp, "plot '%s.rlat' every ::1:::0 using ($1):($2) with lines "
                "lc rgb '%s' title 'reader%s'\n",
                bnfile, rcolor, (rjobs > 1) ? "s" : "");
    }

    if (wjobs > 0) {
        fprintf(fp, "set title 'write latency histogram'\n");
        fprintf(fp, "set yrange [-%lu:]\n", r[1].peakhits / 10);

        for (k = 0; k < r[1].c; ++k) {
            if (!r[1].v[k].latency)
                continue;

            fprintf(fp, "set label %d '' at %.3lf,%lu point pointtype 3 "
                    "lc rgb '%s' front # %.2lfth percentile\n",
                    k + 1, r[1].v[k].latency, r[1].v[k].hits,
                    pcolor, r[1].v[k].pct * 100);
        }

        fprintf(fp, "set label %d '' at %.3lf,%lu point pointtype 1 "
                "lc rgb '%s' front # average latency\n",
                k + 1, r[1].latavg_latency, r[1].latavg_hits, pcolor);

        fprintf(fp, "plot '%s.wlat' every ::1:::0 using ($1):($2) with lines "
                "lc rgb '%s' title 'writer%s'\n",
                bnfile, wcolor, (wjobs > 1) ? "s" : "");
    }

    fprintf(fp, "unset multiplot\n");
    fclose(fp);

    snprintf(buf, sizeof(buf), "gnuplot %s", fnplot);
    fp = popen(buf, "r");
    if (fp) {
        pclose(fp);
    }
}

void
usage(void)
{
    printf("usage: %s [options] <device> ...\n", progname);
    printf("usage: %s -h\n", progname);
    printf("-d secs     specify test duration (seconds) (default: %ld)\n", duration);
    printf("-g prefix   specify output file name prefix\n");
    printf("-h          print this help list\n");
    printf("-I          initialize the partition with random data\n");
    printf("-P pctlist  specify a list of percentiles (default: %s)\n", percentilestr);
    printf("-l partsz   specify the max partition size to use\n");
    printf("-R rdargs   same as -r but sequential\n");
    printf("-r rdargs   args for random i/o reader threads\n");
    printf("-v          increase verbosity\n");
    printf("-W wrargs   same as -w but sequential\n");
    printf("-w wrargs   args for random i/o writer threads\n");
    printf("\n");
    printf("<device>  device or file name to test\n");
    printf("rdargs    rdjobs[,rdsize] (default: %u,%zu)\n", rjobs, riosz);
    printf("wrargs    wrjobs[,wrsize] (default: %u,%zu)\n", wjobs, wiosz);
}

int
main(int argc, char **argv)
{
    struct timeval tv_alpha;
    bool precondition;
    int oflags, rc;
    int fdv[argc];
    u_int i, j;

    progname = strrchr(argv[0], '/');
    progname = progname ? progname + 1 : argv[0];

    xoroshiro128plus_init(state, itv_start());

    oflags = O_RDWR | O_DIRECT;
    precondition = false;
    riosz = wiosz = 4096;
    rjobs = wjobs = 0;
    duration = 60;
    partend = 0;
    partsz = 0;

#if USE_CLOCK
    itv_freq = 1000000000; /* using clock_gettime() for interval measurements */

#elif HAVE_RDTSC

#if __FreeBSD__
    if (!itv_freq) {
        size_t valsz = sizeof(itv_freq);

        rc = sysctlbyname("machdep.tsc_freq", &itv_freq, &valsz, NULL, 0);
        if (rc) {
            eprint(errno, "unable to query sysctlbyname(machdep.tsc_freq)");
            exit(EX_OSERR);
        }
    }

#elif __linux__
    if (!itv_freq) {
        char linebuf[1024];
        double bogomips;
        FILE *fp;
        int n;

        fp = fopen("/proc/cpuinfo", "r");
        if (fp) {
            while (fgets(linebuf, sizeof(linebuf), fp)) {
                n = sscanf(linebuf, "bogomips%*[^0-9]%lf", &bogomips);
                if (n == 1) {
                    itv_freq = (bogomips * 1000000) / 2;
                    break;
                }
            }

            fclose(fp);
        }
    }
#endif
#endif

    if (!itv_freq) {
        eprint(ENOTSUP, "unable to determine TSC frequency, try -f option");
        exit(EX_OSERR);
    }

    usecs_per_cycle = 1000000.0 / itv_freq;

    while (1) {
        char *errmsg, *end;
        int c;

        c = getopt(argc, argv, ":d:g:hiP:p:R:r:vW:w:x");
        if (-1 == c)
            break;

        errmsg = end = NULL;
        errno = 0;

        switch (c) {
        case 'd':
            duration = strtol(optarg, &end, 0);
            if (duration < 1)
                duration = 1;
            errmsg = "invalid duration";
            break;

        case 'g':
            prefix = optarg;
            break;

        case 'h':
            usage();
            exit(0);

        case 'i':
            precondition = true;
            break;

        case 'P':
            percentilestr = optarg;
            break;

        case 'p':
            partsz = strtoul(optarg, &end, 0);
            errmsg = "invalid partition size";
            break;

        case 'R':
            rsequential = true;
            /* FALLTHROUGH */

        case 'r':
            errmsg = "invalid read jobs";
            rjobs = strtoul(optarg, &end, 0);
            if (*end == ',' || *end == ':') {
                errmsg = "invalid read I/O size";
                riosz = strtoul(end + 1, &end, 0);
            }
            break;

        case 'v':
            ++verbosity;
            break;

        case 'W':
            wsequential = true;
            /* FALLTHROUGH */

        case 'w':
            errmsg = "invalid write jobs";
            wjobs = strtoul(optarg, &end, 0);
            if (*end == ',' || *end == ':') {
                errmsg = "invalid write I/O size";
                wiosz = strtoul(end + 1, &end, 0);
            }
            break;

        case 'x':
            oflags &= ~O_DIRECT;
            break;

        case ':':
            syntax("option -%c requires an argument", optopt);
            exit(EX_USAGE);

        case '?':
            syntax("invalid option -%c", optopt);
            exit(EX_USAGE);

        default:
            eprint(0, "option -%c ignored", c);
            break;
        }

        if (errmsg && errno) {
            syntax("%s", errmsg);
            exit(EX_USAGE);
        } else if (end && *end) {
            syntax("%s '%s'", errmsg, optarg);
            exit(EX_USAGE);
        }
    }

    argc -= optind;
    argv += optind;

    if (argc < 1) {
        syntax("device name required");
        exit(EX_USAGE);
    }

    if (rjobs < 1 && wjobs < 1)
        rjobs = 1;

    if (1) {
        char *tok, *end;
        u_int c = 0;

        percentilev = calloc(strlen(percentilestr), sizeof(*percentilev));
        if (!percentilev)
            abort();

        while (( tok = strsep(&percentilestr, ",;- ") ))
            percentilev[c++] = strtod(tok, &end);

        /* TODO: Sort...
         */
        percentilec = c;
    }

    for (i = 0; i < argc; ++i) {
        fdv[i] = open(argv[i], oflags);
        if (-1 == fdv[i]) {
            eprint(errno, "unable to open %s", argv[i]);
            exit(EX_NOINPUT);
        }
    }

    /* TODO: Deal with multiple devices...
     */
    partend = lseek(fdv[0], 0, SEEK_END);
    if (-1 == partend) {
        eprint(errno, "unable to seek to end of %s", argv[0]);
        exit(EX_OSERR);
    }

    if (partsz == 0)
        partsz = partend * .50;

    if (partsz > partend)
        partsz = partend;

    partsz &= ~((1ul << ilog2(riosz | wiosz)) - 1);

    if (partsz < riosz * rjobs || partsz < wiosz * wjobs) {
        eprint(EINVAL, "partition size too small: %ld\n", partsz);
        exit(EX_USAGE);
    }

    if (precondition) {
        const size_t iobufsz = 1ul << 20;
        uint64_t tstart, tnow, tnext;
        off_t precondsz, i;
        char *iobuf;
        ssize_t cc;

        precondsz = partsz;
        precondsz &= ~(iobufsz - 1);
        if (precondsz < iobufsz) {
            eprint(EINVAL, "partition (%ld) must be at least %zu\n", precondsz, iobufsz);
            exit(EX_USAGE);
        }

        if (verbosity > 0) {
            printf("precond: %ld of %ld MiB (%.2lf%%)\n",
                   precondsz >> 20, partend >> 20,
                   precondsz * 100.0 / partend);
            fflush(stdout);
        }

        iobuf = super_alloc(iobufsz * 2);
        if (!iobuf) {
            eprint(errno, "unable to alloc a %zu-byte buffer for preconditioning\n");
            exit(EX_OSERR);
        }

        for (i = 0; i < iobufsz * 2; i += sizeof(u_long))
            *(u_long *)(iobuf + i) = xoroshiro128plus(state);

        tstart = itv_start();
        tnext = tstart + itv_freq;

        for (i = 0; i < precondsz / iobufsz; ++i) {

            cc = pwrite(fdv[0], iobuf + ((i * 4096) % iobufsz), iobufsz, i * iobufsz);

            if (cc != iobufsz) {
                eprint(errno, "\nprecond 1: pwrite cc=%ld (i=%d)\n", cc, i);
                exit(EX_OSERR);
            }

            tnow = itv_stop();
            if (tnow >= tnext) {
                time_t duration = itv_to_seconds(tnow - tstart);

                printf("\rprecond 1: %lds, %ld MiB, %ld MiB/s, %.2lf%%",
                       duration, i,
                       i / duration,
                       (i * 100.0) / (precondsz / iobufsz));
                fflush(stdout);

                tnext = tnow + itv_freq;
            }
        }

        if (verbosity > 0)
            printf("\n");

        super_free(iobuf, iobufsz * 2);
    }

    tdargsv = aligned_alloc(4096, roundup((rjobs + wjobs) * sizeof(*tdargsv), 4096));
    if (!tdargsv) {
        eprint(errno, "alloc tdargsv failed");
        exit(EX_OSERR);
    }

    setpriority(PRIO_PROCESS, 0, -10);

    rc = pthread_barrier_init(&rwbarrier, NULL, rjobs + wjobs + 1);
    if (rc) {
        eprint(rc, "pthread_barrier");
        exit(EX_OSERR);
    }

    for (j = 0; j < rjobs + wjobs; ++j) {
        struct tdargs *a = tdargsv + j;

        a->tid = j;
        a->fd = fdv[j % argc];

        if (j < rjobs) {
            a->iosz = riosz;
            a->random = !rsequential;
            a->read = true;
        } else {
            a->iosz = wiosz;
            a->random = !wsequential;
            a->read = false;
        }

        a->bktvsz = roundup(sizeof(*a->bktv) * (BKT_MAX + duration + 1), 4096);

        a->bktv = super_alloc(a->bktvsz);
        if (!a->bktv)
            abort();

        a->opsv = a->bktv + BKT_MAX;

        rc = pthread_create(&a->thr, NULL, test_main, a);
        if (rc) {
            eprint(rc, "pthread_create");
            exit(EX_OSERR);
        }
    }

    gettimeofday(&tv_alpha, NULL);
    itv_alpha = itv_start();
    itv_omega = itv_alpha + itv_freq * duration;

    if (verbosity > 0) {
        printf("testing: %ld of %ld MiB (%.2lf%%), %u readers, %u writers\n",
               partsz >> 20, partend >> 20,
               partsz * 100.0 / partend,
               rjobs, wjobs);
        fflush(stdout);
    }

    pthread_barrier_wait(&rwbarrier);

    for (j = 0; j < rjobs + wjobs; ++j) {
        struct tdargs *a = tdargsv + j;
        void *val;

        rc = pthread_join(a->thr, &val);
        if (rc) {
            eprint(rc, "pthread_join tid %d failed", j);
            exit(EX_OSERR);
        }
    }

    itv_omega = itv_stop();

    printf("%12ld  total test time (seconds)\n", itv_to_seconds(itv_omega - itv_alpha));
    printf("%12ld  partition size (MiB)\n", partend >> 20);
    printf("%12ld  partition used (MiB)\n", partsz >> 20);
    printf("%12lu  itv_freq\n", itv_freq);
    printf("\n");

    if (prefix) {
        snprintf(bnfile, sizeof(bnfile), "%s-%s%u-%s%u-%ld",
                 prefix,
                 rsequential ? "R" : "r", rjobs,
                 wsequential ? "W" : "w", wjobs,
                 tv_alpha.tv_sec);
    }

    memset(latresv, 0, sizeof(latresv));
    latresv[0].name = "reader";
    latresv[0].c = percentilec;

    latresv[1].name = "writer";
    latresv[1].c = percentilec;

    for (i = 0; i < percentilec; ++i) {
        latresv[0].v[i].pct = percentilev[i] / 100;
        latresv[1].v[i].pct = percentilev[i] / 100;
    }

    report_latency(tdargsv, rjobs, latresv);
    report_latency(tdargsv + rjobs, wjobs, latresv + 1);
    report_ops(tdargsv, rjobs + wjobs, latresv, tv_alpha.tv_sec);

    for (i = 0; i < rjobs + wjobs; ++i) {
        struct tdargs *a = tdargsv + i;

        super_free(a->bktv, a->bktvsz);
    }

    pthread_barrier_destroy(&rwbarrier);
    free(percentilev);

    for (i = 0; i < argc; ++i)
        close(fdv[i]);

    return 0;
}
