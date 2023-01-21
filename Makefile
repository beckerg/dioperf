
all asan clean clobber debug native:
	echo running \"gmake ${MAKEFLAGS} $@\"
	gmake ${MAKEFLAGS} $@
