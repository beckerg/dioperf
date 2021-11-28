# Copyright (c) 2021 Greg Becker.  All rights reserved.


# The only variables you might need to change in this makefile are:
# PROG, SRC, HDR, LDLIBS, VPATH, and CDEFS.
#
PROG	:= dioperf

SRC	:= dioperf.c

HDR	:= ${patsubst %.c,%.h,${SRC}}
HDR	+=

LDLIBS	:= -lpthread
VPATH	:=

NCT_VERSION	:= $(shell git describe --abbrev=10 --dirty --always --tags)
PLATFORM	:= $(shell uname -s | tr 'a-z' 'A-Z')

INCLUDE 	:= -I. -I../lib -I../../src/include
CDEFS 		:= -DNCT_VERSION=\"${NCT_VERSION}\"
CDEFS 		:= -D_GNU_SOURCE

CFLAGS		+= -Wall -g -O2 ${INCLUDE}
DEBUG		:= -O0 -DDEBUG -UNDEBUG -fno-omit-frame-pointer
CPPFLAGS	:= ${CDEFS}
OBJ		:= ${SRC:.c=.o}

# Always delete partially built targets.
#
.DELETE_ON_ERROR:

.PHONY:	all asan clean clobber debug native


all: ${PROG}

asan: CFLAGS += ${DEBUG}
asan: CFLAGS += -fsanitize=address -fsanitize=undefined
asan: LDLIBS += -fsanitize=address -fsanitize=undefined
asan: ${PROG}

clean:
	rm -f ${PROG} ${OBJ} *.core
	rm -f $(patsubst %.c,.%.d,${SRC})

cleandir clobber distclean: clean

debug: CFLAGS += ${DEBUG}
debug: ${PROG}

native: CFLAGS += -march=native
native: ${PROG}

# Use gmake's link rule to produce the target.
#
${PROG}: ${OBJ}
	$(LINK.o) $^ $(LOADLIBES) $(LDLIBS) -o $@


# We make ${OBJ} depend on the GNUmakefile so that all objects are rebuilt
# if the makefile changes.
#
${OBJ}: GNUmakefile

# Automatically generate/maintain dependency files.
#
.%.d: %.c
	@set -e; rm -f $@; \
	$(CC) -M $(CPPFLAGS) ${INCLUDE} $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

-include $(patsubst %.c,.%.d,${SRC})
