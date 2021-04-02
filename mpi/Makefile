#SW=FUGAKU
SW =GNU3
VAMP = false
SCOREP = false

# default values
LDFLAGS := -fopenmp -g # -pg
NVCCFLAGS := -Drestrict=__restrict__ -Xcompiler -fopenmp,-ffast-math,-Wall -arch=compute_20 -code=sm_20

VERTEX_REORDERING = 2
REAL_BENCHMARK = false
VERTEX_REORDERING_OPT = -DVERTEX_REORDERING=$(VERTEX_REORDERING)
ifeq ($(REAL_BENCHMARK), false)
REAL_BENCHMARK_OPT =
else
REAL_BENCHMARK_OPT = -DREAL_BENCHMARK
endif

GCC_BASE := -fopenmp -g -Wall -Drestrict=__restrict__ -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS -D__STDC_FORMAT_MACROS -ffast-math $(VERTEX_REORDERING_OPT) $(REAL_BENCHMARK_OPT) # -pg
FCC_BASE := -Kopenmp -Xg -Drestrict=__restrict__ -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS -D__STDC_FORMAT_MACROS -g #-Nquickdbg=heapchk # -Koptmsg=2
MPICC := mpicc
MPICPP := mpicxx

BINS := runnable
OBJS := main.o low_level_func.o ../generator/splittable_mrg.o
LIBS := -lm

ifeq ($(SW), GNU3)
CFLAGS := $(GCC_BASE) -O3 -DNDEBUG -DPROFILE_REGIONS
NVCCFLAGS := $(NVFLAG_BASE) -g -O3 -DNDEBUG -keep
LIBS := $(LIBS) -lnuma
endif

ifeq ($(SW), FUGAKU)
MPICC := mpifccpx
MPICPP := mpiFCCpx
FCC_BASE := -Kopenmp -Drestrict=__restrict__  $(VERTEX_REORDERING_OPT) $(REAL_BENCHMARK_OPT)
CFLAGS := $(FCC_BASE) -Kfast -DNDEBUG -DPROFILE_REGIONS
LDFLAGS := $(CFLAGS) -Nfjomplib
#LDFLAGS := $(CFLAGS)
endif

ifeq ($(SW), INTEL3)
MPICC := mpiicc
MPICPP := mpiicpc
CFLAGS := -ipo -no-prec-div -qopenmp -g3 -Drestrict=__restrict__ -O3 -DNDEBUG $(VERTEX_REORDERING_OPT) $(REAL_BENCHMARK_OPT)
#CFLAGS := -ipo -openmp -g3 -Drestrict=__restrict__ -O2 -DNDEBUG
NVCCFLAGS := $(NVFLAG_BASE) -g -O3 -DNDEBUG -keep
LDFLAGS := -qopenmp -g
LIBS := $(LIBS) -lnuma -lmpi
endif

ifeq ($(VAMP), true)
MPICC := vtcc -vt:cc $(MPICC) -vt:inst manual -vt:hyb -DVTRACE
MPICPP := vtcc -vt:cc $(MPICPP) -vt:inst manual -vt:hyb -DVTRACE
endif
ifeq ($(SCOREP), true)
MPICC := scorep --nocompiler --user --static $(MPICC) -DSCOREP
MPICPP := scorep --nocompiler --user --static $(MPICPP) -DSCOREP
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
