SW =GNU3
#SW=FUGAKU

# default values
LDFLAGS := -fopenmp -g # -pg
FUGAKU_POWER_MEASUREMENT=false
VERBOSE = false
VERTEX_REORDERING = 2
REAL_BENCHMARK = false
ifeq ($(VERBOSE), false)
VERBOSE_OPT = -DVERVOSE_MODE=0
else
VERBOSE_OPT = -DVERVOSE_MODE=1
endif
VERTEX_REORDERING_OPT = -DVERTEX_REORDERING=$(VERTEX_REORDERING)
ifeq ($(REAL_BENCHMARK), false)
REAL_BENCHMARK_OPT =
else
REAL_BENCHMARK_OPT = -DREAL_BENCHMARK
endif
ifeq ($(FUGAKU_POWER_MEASUREMENT), true)
FUGAKU_POWER_CFLAGS=-D_FUGAKU_POWER_MEASUREMENT
FUGAKU_POWER_CLINK=-lpwr
endif

GCC_BASE := -fopenmp -g -Wall -Drestrict=__restrict__ -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS -D__STDC_FORMAT_MACROS -ffast-math $(VERBOSE_OPT) $(VERTEX_REORDERING_OPT) $(REAL_BENCHMARK_OPT)
FCC_BASE := -Kopenmp -Xg -Drestrict=__restrict__ -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS -D__STDC_FORMAT_MACROS -g #-Nquickdbg=heapchk # -Koptmsg=2

MPICC := mpicc
MPICPP := mpicxx
BINS := runnable
OBJS := main.o ../generator/splittable_mrg.o
LIBS := -lm

ifeq ($(SW), GNU3)
CFLAGS := $(GCC_BASE) -O3 -DNDEBUG -DPROFILE_REGIONS
LIBS := $(LIBS) -lnuma
endif
ifeq ($(SW), INTEL3)
MPICC := mpiicc
MPICPP := mpiicpc
CFLAGS := -ipo -no-prec-div -qopenmp -g3 -Drestrict=__restrict__ -O3 -DNDEBUG $(VERBOSE_OPT) $(VERTEX_REORDERING_OPT) $(REAL_BENCHMARK_OPT)
LDFLAGS := -qopenmp -g
LIBS := $(LIBS) -lnuma -lmpi
endif
ifeq ($(SW), FUGAKU)
MPICC := mpifccpx
MPICPP := mpiFCCpx
FCC_BASE := -Kopenmp -Drestrict=__restrict__  $(VERBOSE_OPT) $(VERTEX_REORDERING_OPT) $(REAL_BENCHMARK_OPT)
CFLAGS := $(FCC_BASE) -Kfast -DNDEBUG -DPROFILE_REGIONS $(FUGAKU_POWER_CFLAGS)
LDFLAGS := $(CFLAGS) -Nfjomplib $(FUGAKU_POWER_CLINK)
endif

CPPFLAGS := $(CFLAGS)

all: cpu
cpu: $(OBJS)
	$(MPICPP) $(LDFLAGS) -o runnable $(OBJS) $(LIBS)
gnu_func.o: gnu_func.cc
	cp ../../$*.o_ $*.o
#	g++ -c -g -O3 -Wall $< -o $*.o

.SUFFIXES: .o .c .cpp .cc
.c.o:
	$(MPICC) $(INC) -c $(CFLAGS) $< -o $*.o
.cpp.o:
	$(MPICPP) $(INC) -c $(CPPFLAGS) $< -o $*.o
.cc.o:
	$(MPICPP) $(INC) -c $(CPPFLAGS) $< -o $*.o

.PHONY: clean
clean:
	-rm -f $(BINS) $(OBJS)
