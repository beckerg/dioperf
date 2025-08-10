# Copyright (c) 2021,2025 Greg Becker.  All rights reserved.

PROG := dioperf

SRC := ${PROG}.c
HDR := ${patsubst %.c,%.h,${SRC}}
OBJ := ${SRC:.c=.o}

PLATFORM := $(shell uname -s | tr 'A-Z' 'a-z')
VERSION  := $(shell git describe --abbrev=10 --dirty --always --tags)
PROGUC   := $(shell echo -n ${PROG} | tr 'a-z' 'A-Z')

INCLUDE  := -I. -I../lib -I../../src/include
CPPFLAGS := -D${PROGUC}_VERSION=\"${VERSION}\" -DNDEBUG
#CPPFLAGS += -DUSE_CLOCK

LDLIBS   := -lpthread

ifeq (${PLATFORM},linux)
CPPFLAGS += -D_GNU_SOURCE
LDLIBS   += -lbsd
endif

CFLAGS   += -Wall -Wextra -O2 ${INCLUDE}

# Always delete partially built targets.
#
.DELETE_ON_ERROR:

.PHONY:	all asan clean clobber debug native


all: ${PROG}

asan: CFLAGS += -O0 -g3 -fno-omit-frame-pointer
asan: CFLAGS += -fsanitize=address -fsanitize=undefined
asan: LDLIBS += -fsanitize=address -fsanitize=undefined
asan: CPPFLAGS += -UNDEBUG
asan: ${PROG}

clean:
	rm -f ${PROG} ${OBJ} *.core
	rm -f $(patsubst %.c,.%.d,${SRC})

cleandir clobber distclean: clean

debug: CFLAGS += -O0 -g3 -fno-omit-frame-pointer
debug: CPPFLAGS += -UNDEBUG
debug: ${PROG}

native: CFLAGS += -march=native -flto
native: ${PROG}

# Use gmake's link rule to produce the target.
#
${PROG}: ${OBJ}

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
