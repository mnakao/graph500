#SW=FUGAKU
SW=GNU3
VERTEX_REORDERING_OPT = -DVERTEX_REORDERING=2
REAL_BENCHMARK_OPT =
#REAL_BENCHMARK_OPT = -DREAL_BENCHMARK

BINS := runnable
OBJS := main.o low_level_func.o ../generator/splittable_mrg.o
LIBS := -lm

ifeq ($(SW), GNU3)
MPICC := mpicc
MPICPP := mpicxx
GCC_BASE := -fopenmp -g -Wall -Drestrict=__restrict__ -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS -D__STDC_FORMAT_MACROS -ffast-math $(VERTEX_REORDERING_OPT) $(REAL_BENCHMARK_OPT)
CFLAGS := $(GCC_BASE) -O3 -DNDEBUG -DPROFILE_REGIONS
LDFLAGS := -fopenmp -g
LIBS := $(LIBS) -lnuma
endif

ifeq ($(SW), FUGAKU)
MPICC := mpifccpx
MPICPP := mpiFCCpx
FCC_BASE := -Kopenmp -Drestrict=__restrict__  $(VERTEX_REORDERING_OPT) $(REAL_BENCHMARK_OPT)
FCC_BASE := -Kopenmp -Xg -Drestrict=__restrict__ -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS -D__STDC_FORMAT_MACROS -g #-Nquickdbg=heapchk # -Koptmsg=2
CFLAGS := $(FCC_BASE) -Kfast -DNDEBUG -DPROFILE_REGIONS
LDFLAGS := $(CFLAGS) -Nfjomplib
endif

ifeq ($(SW), INTEL3)
MPICC := mpiicc
MPICPP := mpiicpc
CFLAGS := -ipo -no-prec-div -qopenmp -g3 -Drestrict=__restrict__ -O3 -DNDEBUG $(VERTEX_REORDERING_OPT) $(REAL_BENCHMARK_OPT)
LDFLAGS := -qopenmp -g
LIBS := $(LIBS) -lnuma -lmpi
endif

CPPFLAGS := $(CFLAGS)

all: cpu
cpu: $(OBJS)
	$(MPICPP) $(LDFLAGS) -o runnable $(OBJS) $(LIBS)

gnu_func.o: gnu_func.cc
	cp ../../$*.o_ $*.o

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
