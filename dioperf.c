/*
 * This tool generates various r/w workloads and measures and graphs
 * the throughput and latency of the device under test.
 *
 * cc -O2 -D_GNU_SOURCE -g dioperf.c -o dioperf -lm -lpthread
 */

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
#include <semaphore.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/resource.h>

#include <immintrin.h>
#include <x86intrin.h>

#if __FreeBSD__
#include <sys/sysctl.h>
#elif __linux__
#include <bsd/string.h>
#endif

#define HAVE_RDTSC          (__has_builtin(__builtin_ia32_rdtsc))
#define HAVE_RDTSCP         (__has_builtin(__builtin_ia32_rdtscp))
#define HAVE_RDRAND64       (__has_builtin(__builtin_ia32_rdrand64_step))
#define HAVE_PAUSE          (__has_builtin(__builtin_ia32_pause))

#define HAVE_STRERROR_S     (0) // TODO: How to detect???

/* By default we use rdtsc() to measure timing intervals, but if
 * it's not available we'll fall back to using clock_gettime().
 */
#ifndef USE_CLOCK
#define USE_CLOCK           (!HAVE_RDTSC)
#endif

#define NELEM(_arr)         (sizeof(_arr) / sizeof((_arr)[0]))

#ifndef unlikely
#define unlikely(_expr)     __builtin_expect(!!(_expr), 0)
#endif

#ifndef __aligned
#define __aligned(_size)    __attribute__((__aligned__(_size)))
#endif

#ifndef OFF_MAX
#define OFF_MAX             (LONG_MAX)
#endif

/* Number of buckets in latency histogram ((1u << BKT_SHIFT) cycles per bucket).
 */
#define BKT_MAX             (4 * 1024 * 1024)

#if USE_CLOCK
#define BKT_SHIFT           (0)
#else
#define BKT_SHIFT           (7)
#endif

typedef ssize_t rwfunc_t(int, void *, size_t, off_t);

struct tdargs {
    rwfunc_t   *rwfunc __aligned(64);
    bool        random;
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
    u_long thresh;
    double latency;
    u_long hits;
    double pct;
};

struct latres {
    const char *name;
    double latavg_latency;
    u_long latavg_hits;
    double latmax_latency;
    u_long latmax_hits;
    u_long peakhits;
    u_int latdatc;
    struct latdat latdatv[16];
};

const char *wcolor[] = { "#cc0000", "#0000cc" };
const char *rcolor[] = { "#00cccc", "#0000cc" };
const char *term = "png";
const char *progname;
const char *ofile;
int verbosity;
bool rsequential;
bool wsequential;
bool use_mmap;
bool dryrun;
size_t riosz;
size_t wiosz;
u_int rjobs;
u_int wjobs;
off_t partend;
off_t partsz;
char *mmapv[16];
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

__thread uint64_t xrand_tls[2];


static inline uint64_t
rotl(const uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

/* xoroshiro128plus PRNG
 */
static inline void
xrand_init(uint64_t seed)
{
    uint64_t z;

    z = (seed += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    xrand_tls[0] = z ^ (z >> 31);

    z = (seed += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    xrand_tls[1] = z ^ (z >> 31);
}

static inline uint64_t
xrand(void)
{
    const uint64_t s0 = xrand_tls[0];
    uint64_t s1 = xrand_tls[1];
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    xrand_tls[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); /* a, b */
    xrand_tls[1] = rotl(s1, 36); /* c */

    return result;
}

static inline __attribute__((__const__))
u_int
ilog2(unsigned long long n)
{
    return (NBBY * sizeof(n) - 1) - __builtin_clzll(n);
}

/* Interval measurement abstractions...
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

#if !HAVE_STRERROR_S
/* Try to avoid the glibc strerror_r mess...
 */
int
strerror_s(char *buf, size_t bufsz, int errnum)
{
    static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    size_t len;

    /* TODO: Do constraint stuff...
     */
#ifdef RSIZE_MAX
    if (!buf || bufsz > RSIZE_MAX)
        return EINVAL;
#endif

    pthread_mutex_lock(&lock);
    len = strlcpy(buf, strerror(errnum) ?: "dunno", bufsz);
    pthread_mutex_unlock(&lock);

    if (len >= bufsz && bufsz > 3)
        strlcpy(buf + bufsz - 4, "...", 4);

    return 0;
}
#endif

static void
eprint(int errnum, const char *fmt, ...)
{
    char errbuf[256] = ": strerror_s failed";
    char msgbuf[256] = ": vsnprintf failed";
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(msgbuf + 2, sizeof(msgbuf) - 2, fmt, ap);
    va_end(ap);

    strerror_s(errbuf + 2, sizeof(errbuf) - 2, errnum);

    fprintf(stderr, "%s%s%s\n", progname, msgbuf, errnum ? errbuf : "");
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

/* On FreeBSD this "just works".  On Linux you need to set
 * vm.nr_hugepages in /etc/sysctl.conf to something adquetely
 * large in order for this to allocate superpages.  Or maybe
 * use transparent huge pages (THP), or hugetblfs.  Yuk...
 */
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

ssize_t
mmap_pread(int fd, void *buf, size_t nbytes, off_t offset)
{
    memcpy(buf, mmapv[fd] + offset, nbytes);

    return nbytes;
}

ssize_t
mmap_pwrite(int fd, void *buf, size_t nbytes, off_t offset)
{
    memcpy(mmapv[fd] + offset, buf, nbytes);

    return nbytes;
}

void *
rwtest(void *arg)
{
    struct tdargs *a = arg;
    uint64_t partsz_mask;
    uint64_t iosz_mask;
    uint64_t itv_next;
    time_t elapsed;
    u_long *iobuf;
    ssize_t iosz;
    ssize_t cc;
    off_t off;
    u_int i;

    xrand_init(rotl(itv_start(), a->tid % 64));

    iobuf = aligned_alloc(4096, roundup(a->iosz, 4096));
    if (!iobuf)
        abort();

    for (i = 0; i < a->iosz / sizeof(*iobuf); ++i)
        iobuf[i] = xrand();

    memset(a->bktv, 0, a->bktvsz);

    iosz_mask = ~((1ul << ilog2(a->iosz)) - 1);
    off = (partsz / (a->tid + 1)) & iosz_mask;
    iosz = dryrun ? 0 : a->iosz;

    /* partsz_mask is used to eliminate the modulus operation
     * otherwise needed to compute the next random offset.
     */
    partsz_mask = ((2ul << ilog2(partsz)) - 1);
    partsz_mask &= iosz_mask;

    itv_next = itv_alpha + itv_freq;
    elapsed = 0;
    cc = iosz;

    pthread_barrier_wait(&rwbarrier);

    /* TODO: Use locking to prevent overlapped I/O...
     */
    while (itv_next < itv_omega) {
        uint64_t tstart, tstop;
        suseconds_t dt;

        tstart = itv_start();

        cc = a->rwfunc(a->fd, iobuf, iosz, off);
        if (cc != iosz)
            break;

        tstop = itv_stop();

        if (unlikely( tstop >= itv_next )) {
            if (++elapsed >= duration)
                break;
            itv_next += itv_freq;
        }

        dt = (tstop - tstart) >> BKT_SHIFT;

        if (dt > BKT_MAX - 1)
            dt = BKT_MAX - 1;

        ++a->opsv[elapsed];
        ++a->bktv[dt];

        if (a->random) {
            off = xrand() & partsz_mask;
        } else {
            off += a->iosz;
        }

        /* TODO: In random mode, if partsz is not a power of two then
         * this will yield a non-uniform distribution (worst case, 25%
         * of the offsets will occur 25% more frequently, right???)
         */
        if (off >= partsz)
            off -= partsz;
    }

    a->usecs = itv_to_usecs(itv_stop() - itv_alpha);

    if (cc != iosz) {
        eprint((cc == -1) ? errno : 0,
               "tid %d: cc %ld != iosz %zu, off %ld\n",
               a->tid, cc, iosz, off);
    }

    for (i = 0; i < duration; ++i)
        a->opstot += a->opsv[i];

    free(iobuf);

    pthread_exit(NULL);
}

void
report_latency(struct tdargs *a, u_int jobs, struct latres *r)
{
    char fnlat[256];
    double latsum, latmin, usecs;
    u_long opstot, hits;
    u_long bytespersec;
    u_long *bktv;
    u_int i, j, k;
    FILE *fp;

    if (jobs < 1)
        return;

    opstot = 0;
    fp = NULL;

    for (j = 0; j < jobs; ++j)
        opstot += a[j].opstot;

    for (k = 0; k < r->latdatc; ++k)
        r->latdatv[k].thresh = opstot * r->latdatv[k].pct;

    /* Create a file and write out all the latency data.
     */
    if (ofile) {
        snprintf(fnlat, sizeof(fnlat), "%s.%clat", bnfile, r->name[0]);

        fp = fopen(fnlat, "w");
        if (!fp) {
            eprint(errno, "fopen(%s) failed", fnlat);
            return;
        }

        fprintf(fp, "#%9s %9s %8s %10s\n",
                "LATENCY", "HITS", "CENTILE", "PERCENTILE");
    }

    latsum = latmin = usecs = 0;
    bktv = a->bktv;
    hits = 0;
    k = 0;

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

        if (k < r->latdatc) {
            if (hits >= r->latdatv[k].thresh) {
                r->latdatv[k].latency = usecs;
                r->latdatv[k].hits = bktv[i];
                ++k;
            }
        }

        if (!fp)
            continue;

        fprintf(fp, "%10.3lf %9lu %8lu", usecs, bktv[i], (hits * 100) / opstot);

        if (k > 0)
            fprintf(fp, " %10.2lf\n", r->latdatv[k - 1].pct * 100);
        else
            fprintf(fp, " %10s\n", "-");
    }

    if (fp)
        fclose(fp);

    r->latavg_latency = latsum / hits;
    r->latavg_hits = bktv[ (u_long)(r->latavg_latency / usecs_per_cycle) >> BKT_SHIFT ];

    r->latmax_latency = usecs;
    r->latmax_hits = bktv[ (u_long)(r->latmax_latency / usecs_per_cycle) >> BKT_SHIFT ];

    bytespersec = (opstot * a->iosz * 1000000) / a->usecs;

    printf("%12.1lf  %s avg latency (us)\n", r->latavg_latency, r->name);
    printf("%12.1lf  %s min latency (us)\n", latmin, r->name);
    printf("%12.1lf  %s max latency (us)\n", r->latmax_latency, r->name);

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
    char buf[128 + jobs * 16];
    int xtics, mxtics;
    int fontsize = 10;
    u_long peakops;
    u_int j, k;
    FILE *fp;
    time_t i;

    if (!ofile)
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

    peakops = 0;

    for (i = 0; i < duration; ++i) {
        u_long rops = 0, wops = 0;
        size_t pos = 0;
        int n;

        for (j = 0; j < jobs; ++j) {
            n = snprintf(buf + pos, sizeof(buf) - pos, " %7lu", a[j].opsv[i]);
            if (n < 1 || (size_t)n >= sizeof(buf) - pos)
                abort();

            if (j < rjobs)
                rops += a[j].opsv[i];
            else
                wops += a[j].opsv[i];

            pos += n;
        }

        if (rops + wops > peakops)
            peakops = rops + wops;

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

    fprintf(fp, "set multiplot layout %d,1 columnsfirst\n",
            rjobs && wjobs ? 3 : 2);

    fprintf(fp, "set title '%s operations' offset 0, -1\n",
            rjobs && wjobs ? "r/w" : (rjobs ? "read" : "write"));

    fprintf(fp, "set xlabel 'seconds'\n");
    fprintf(fp, "set xtics autofreq nomirror font ',%d'\n", fontsize);
    fprintf(fp, "set xtics 0, %d rotate by -30\n", xtics);
    fprintf(fp, "set mxtics %d\n", mxtics);

    fprintf(fp, "set ylabel '%s operations per second'\n",
            rjobs && wjobs ? "r/w" : (rjobs ? "read" : "write"));
    fprintf(fp, "set ytics autofreq font ',%d'\n", fontsize);
    fprintf(fp, "set mytics 2\n");
    fprintf(fp, "set yrange [-%lu:%lu]\n",
            peakops / 12, peakops + (peakops / 12));

    fprintf(fp, "plot ");

    if (rjobs > 0) {
        fprintf(fp,
                "'%s' every ::1:::0 using ($2):($3) with lines"
                " lc rgb '%s' title 'readers %d, %s, %zu-bytes' %s",
                fnops, rcolor[0], rjobs,
                rsequential ? "sequential" : "random",
                riosz,
                wjobs ? "," : "\n");
    }

    if (wjobs > 0) {
        fprintf(fp,
                "'%s' every ::1:::0 using ($2):($4) with lines"
                " lc rgb '%s' title 'writers %d, %s, %zu-bytes' %s",
                fnops, wcolor[0], wjobs,
                wsequential ? "sequential" : "random",
                wiosz,
                rjobs ? "," : "\n");
    }

    if (rjobs && wjobs) {
        fprintf(fp,
                "'%s' every ::1:::0 using ($2):($5) with lines "
                "lc rgb '#009900' title 'combined'\n",
                fnops);
    }

    fprintf(fp, "set xlabel 'buckets (usecs +/- %ldns)'\n",
            (long)(itv_to_usecs(1u << BKT_SHIFT) * 1000));
    fprintf(fp, "set mxtics 10\n");
    fprintf(fp, "set ylabel 'hits'\n");

    if (rjobs > 0) {
        double latmax = r[0].latmax_latency;

        fprintf(fp, "set title 'read latency' offset 0, -1\n");
        fprintf(fp, "set xtics auto nomirror font ',%d'\n", fontsize);
        fprintf(fp, "set autoscale xfix\n");

        if (r[0].latdatc >= 0) {
            const char *comma = "";
            size_t len = 0;

            fprintf(fp, "set x2tics auto nomirror rotate by 30 font ',%d' (", fontsize);

            for (k = 0; k < r[0].latdatc; ++k) {
                if (r[0].latdatv[k].latency) {
                    fprintf(fp, "%s '%s' %.3lf",
                            comma, percentilestr + len, r[0].latdatv[k].latency);

                    len += strlen(percentilestr + len) + 1;
                    latmax = r[0].latdatv[k].latency;
                    comma = ",";
                }
            }

            fprintf(fp, "%s 'avg' %.3lf )\n", comma, r[0].latavg_latency);
            fprintf(fp, "set autoscale x2fix\n");
        }

        for (k = 0; k < r[0].latdatc; ++k) {
            if (!r[0].latdatv[k].latency)
                continue;

            fprintf(fp, "set label %d '' at %.3lf,%lu point pointtype 1"
                    " lc rgb '%s' front # %.2lf percentile\n",
                    k + 1, r[0].latdatv[k].latency, r[0].latdatv[k].hits,
                    rcolor[1], r[0].latdatv[k].pct * 100);
        }

        fprintf(fp, "set label %d '' at %.3lf,%lu point pointtype 3"
                " lc rgb '%s' front # average latency\n",
                k + 1, r[0].latavg_latency, r[0].latavg_hits, rcolor[1]);

        fprintf(fp, "set xrange [:%.3lf]\n", latmax + (latmax * 3) / 100);
        fprintf(fp, "set yrange [-%lu:%lu]\n",
                r[0].peakhits / 12, r[0].peakhits + (r[0].peakhits / 12));

        fprintf(fp, "plot '%s.rlat' every ::1:::0 using ($1):($2) with lines"
                " lc rgb '%s' title 'reader%s'\n",
                bnfile, rcolor[0], (rjobs > 1) ? "s" : "");
    }

    if (wjobs > 0) {
        double latmax = r[1].latmax_latency;

        fprintf(fp, "set title 'write latency' offset 0, -1\n");
        fprintf(fp, "set xtics autofreq nomirror font ',%d'\n", fontsize);
        fprintf(fp, "set autoscale xfix\n");

        if (r[0].latdatc >= 0) {
            const char *comma = "";
            size_t len = 0;

            fprintf(fp, "set x2tics auto nomirror rotate by 30 font ',%d' (", fontsize);

            for (k = 0; k < r[1].latdatc; ++k) {
                if (r[1].latdatv[k].latency) {
                    fprintf(fp, "%s '%s' %.3lf",
                            comma, percentilestr + len, r[1].latdatv[k].latency);

                    len += strlen(percentilestr + len) + 1;
                    latmax = r[1].latdatv[k].latency;
                    comma = ",";
                }
            }

            fprintf(fp, "%s 'avg' %.3lf )\n", comma, r[1].latavg_latency);
            fprintf(fp, "set autoscale x2fix\n");
        }


        for (k = 0; k < r[1].latdatc; ++k) {
            if (!r[1].latdatv[k].latency)
                continue;

            fprintf(fp, "set label %d '' at %.3lf,%lu point pointtype 1"
                    " lc rgb '%s' front # %.2lf percentile\n",
                    k + 1, r[1].latdatv[k].latency, r[1].latdatv[k].hits,
                    wcolor[1], r[1].latdatv[k].pct * 100);
        }

        fprintf(fp, "set label %d '' at %.3lf,%lu point pointtype 3"
                " lc rgb '%s' front # average latency\n",
                k + 1, r[1].latavg_latency, r[1].latavg_hits, wcolor[1]);

        fprintf(fp, "set xrange [:%.3lf]\n", latmax + (latmax * 3) / 100);
        fprintf(fp, "set yrange [-%lu:%lu]\n",
                r[1].peakhits / 12, r[1].peakhits + (r[1].peakhits / 12));

        fprintf(fp, "plot '%s.wlat' every ::1:::0 using ($1):($2) with lines"
                " lc rgb '%s' title 'writer%s'\n",
                bnfile, wcolor[0], (wjobs > 1) ? "s" : "");
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
    printf("usage: %s -V\n", progname);
    printf("-d secs     specify test duration (seconds) (default: %ld)\n", duration);
    printf("-h          print this help list\n");
    printf("-l partsz   specify the max partition size to use\n");
    printf("-m          use mmap rather than pread/pwrite\n");
    printf("-n          dry run (i.e., issue zero-length reads/writes)\n");
    printf("-o prefix   specify output file name prefix\n");
    printf("-P pctlist  specify a list of percentiles (default: %s)\n", percentilestr);
    printf("-R rdargs   sequential I/O reader thread args\n");
    printf("-r rdargs   random I/O reader thread args\n");
    printf("-T term     specify gnuplot term type (default: %s)\n", term);
    printf("-V          show version\n");
    printf("-v          increase verbosity\n");
    printf("-W rdargs   sequential I/O writer thread args\n");
    printf("-w rdargs   random I/O writer thread args\n");
    printf("\n");
    printf("<device>  device or file name to test\n");
    printf("rdargs    rdjobs[,rdsize] (default: %u,%zu)\n", rjobs, riosz);
    printf("wrargs    wrjobs[,wrsize] (default: %u,%zu)\n", wjobs, wiosz);
}

int
main(int argc, char **argv)
{
    struct timeval tv_alpha;
    bool version, help;
    int fdv[argc], rc;
    int directio;
    u_int i, j;

    progname = strrchr(argv[0], '/');
    progname = progname ? progname + 1 : argv[0];

    xrand_init(itv_start());

    directio = O_DIRECT;
    riosz = wiosz = 4096;
    rjobs = wjobs = 0;
    partend = OFF_MAX;
    use_mmap = false;
    version = false;
    dryrun = false;
    help = false;
    duration = 60;
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
        eprint(ENOTSUP, "unable to determine TSC frequency, try -DUSE_CLOCK");
        exit(EX_OSERR);
    }

    usecs_per_cycle = 1000000.0 / itv_freq;

    while (1) {
        const char *delim = ",:: ";
        char *errmsg, *end;
        int c;

        c = getopt(argc, argv, ":d:hl:mno:P:R:r:T:VvW:w:x");
        if (c == -1)
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

        case 'h':
            help = true;
            break;

        case 'l':
            partsz = strtoul(optarg, &end, 0);
            errmsg = "invalid partition size";
            break;

        case 'm':
            use_mmap = true;
            break;

        case 'n':
            dryrun = true;
            break;

        case 'o':
            ofile = optarg;
            break;

        case 'P':
            percentilestr = optarg;
            break;

        case 'R':
            rsequential = true;
            /* FALLTHROUGH */

        case 'r':
            errmsg = "invalid read jobs";
            rjobs = strtoul(optarg, &end, 0);
            if (strpbrk(end, delim)) {
                errmsg = "invalid read I/O size";
                riosz = strtoul(end + 1, &end, 0);
                if (strpbrk(end, delim)) {
                    rcolor[0] = end + 1;
                    end = NULL;
                }
            }
            break;

        case 'T':
            term = optarg;
            break;

        case 'v':
            ++verbosity;
            break;

        case 'V':
            version = true;
            break;

        case 'W':
            wsequential = true;
            /* FALLTHROUGH */

        case 'w':
            errmsg = "invalid write jobs";
            wjobs = strtoul(optarg, &end, 0);
            if (strpbrk(end, delim)) {
                errmsg = "invalid write I/O size";
                wiosz = strtoul(end + 1, &end, 0);
                if (strpbrk(end, delim)) {
                    wcolor[0] = end + 1;
                    end = NULL;
                }
            }
            break;

        case 'x':
            directio = 0;
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

        if (errno && errmsg) {
            syntax("%s", errmsg);
            exit(EX_USAGE);
        } else if (end && *end) {
            syntax("%s '%s'", errmsg, optarg);
            exit(EX_USAGE);
        }
    }

    if (rjobs < 1 && wjobs < 1)
        rjobs = 1;


    if (help) {
        usage();
        exit(0);
    }
    else if (version) {
        printf("%s\n", DIOPERF_VERSION);
        exit(0);
    }

    if (1) {
        char *str = percentilestr;
        char *tok, *end;
        double prev, val;

        percentilev = calloc(strlen(percentilestr), sizeof(*percentilev));
        if (!percentilev)
            abort();

        errno = 0;
        prev = 0;

        while (( tok = strsep(&str, ",;: ") )) {
            val = strtod(tok, &end);
            if (errno || (end && *end) || val < prev) {
                eprint(errno ?: EINVAL, "invalid percentile '%s' ignored", tok);
                errno = 0;
                continue;
            }

            percentilev[percentilec++] = val;
            prev = val;
        }

        if (percentilec > NELEM(latresv->latdatv)) {
            percentilec = NELEM(latresv->latdatv);
        }
    }

    argc -= optind;
    argv += optind;

    if (argc < 1) {
        syntax("device name required");
        exit(EX_USAGE);
    }
    else if (use_mmap && (size_t)argc >= NELEM(mmapv)) {
        eprint(EINVAL, "using at most %zu devices", NELEM(mmapv));
        argc = NELEM(mmapv);
    }

    for (i = 0; i < (u_int)argc; ++i) {
        off_t end;

        fdv[i] = open(argv[i], O_RDWR | directio);
        if (-1 == fdv[i]) {
            eprint(errno, "unable to open %s", argv[i]);
            exit(EX_NOINPUT);
        }

        end = lseek(fdv[0], 0, SEEK_END);
        if (-1 == end) {
            eprint(errno, "unable to seek to end of %s", argv[0]);
            exit(EX_OSERR);
        }

        /* Use the smallest partition from the group...
         */
        if (end < partend)
            partend = end;
    }

    if (partsz == 0)
        partsz = 1ul << ilog2(partend - 1);

    if (partsz > partend)
        partsz = partend;

    partsz &= ~((2ul << ilog2((riosz | wiosz) - 1)) - 1);

    if (partsz / 2 < (off_t)(riosz * rjobs) || partsz / 2 < (off_t)(wiosz * wjobs)) {
        eprint(EINVAL, "partition size too small: %ld\n", partsz);
        exit(EX_USAGE);
    }

    if (use_mmap) {
        int prot = PROT_READ | PROT_WRITE;
        int flags = MAP_PRIVATE;

#if __Free_BSD
        flags |= MAP_NOCORE | MAP_NOSYNC;
#endif

        for (i = 0; i < (u_int)argc; ++i) {
            mmapv[i] = mmap(NULL, partsz, prot, flags, fdv[i], 0);
            if (mmapv[i] == MAP_FAILED) {
                eprint(errno, "mmap %s failed", argv[i]);
                exit(EX_OSERR);
            }

            madvise(mmapv[i], partsz, MADV_RANDOM);

            close(fdv[i]);
            fdv[i] = i;
        }
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
            a->rwfunc = use_mmap ? mmap_pread : pread;
            a->iosz = riosz;
            a->random = !rsequential;
        } else {
            a->rwfunc = use_mmap ? mmap_pwrite : (rwfunc_t *)pwrite;
            a->iosz = wiosz;
            a->random = !wsequential;
        }

        a->bktvsz = roundup(sizeof(*a->bktv) * (BKT_MAX + duration + 1), 4096);

        a->bktv = super_alloc(a->bktvsz);
        if (!a->bktv)
            abort();

        a->opsv = a->bktv + BKT_MAX;

        rc = pthread_create(&a->thr, NULL, rwtest, a);
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

    printf("%12.3lf  total test time (seconds)\n",
           itv_to_usecs(itv_omega - itv_alpha) / 1000000);
    printf("%12ld  partition size (MiB)\n", partend >> 20);
    printf("%12ld  partition used (MiB)\n", partsz >> 20);
    printf("%12lu  itv_freq\n", itv_freq);
    printf("\n");

    if (ofile) {
        snprintf(bnfile, sizeof(bnfile), "%s-%s%u-%s%u-%ld",
                 ofile,
                 rsequential ? "R" : "r", rjobs,
                 wsequential ? "W" : "w", wjobs,
                 tv_alpha.tv_sec);
    }

    memset(latresv, 0, sizeof(latresv));
    latresv[0].name = "reader";
    latresv[0].latdatc = percentilec;

    latresv[1].name = "writer";
    latresv[1].latdatc = percentilec;

    for (i = 0; i < percentilec; ++i) {
        latresv[0].latdatv[i].pct = percentilev[i] / 100;
        latresv[1].latdatv[i].pct = percentilev[i] / 100;
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

    return 0;
}
