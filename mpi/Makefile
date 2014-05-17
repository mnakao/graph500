SW = FUJI3
VAMP = false

# default values
MPICC = mpicc
MPICPP = mpicxx

CUDALIB = -L$(CUDA_PATH)/lib64 -lcudart -lcuda
NVCC = nvcc

INC = # -I$(CUDA_PATH)/include
BINS = runnable
OBJS = main.o low_level_func.o ../generator/splittable_mrg.o
CUOBJ = cuda_kernel.o
LIBS = -lm

NVFLAG_BASE = -Drestrict=__restrict__ -Xcompiler -fopenmp,-ffast-math,-Wall -arch=compute_20 -code=sm_20
ifeq ($(SW), GNU3)
CFLAGS = -fopenmp -g -Wall -Drestrict=__restrict__ -O3 -DNDEBUG -ffast-math -msse4.2 # -g -pg
NVCCFLAGS = $(NVFLAG_BASE) -g -O3 -DNDEBUG -keep
LDFLAGS = -fopenmp -g # -g -pg
LIBS := $(LIBS) -lnuma
endif
ifeq ($(SW), GNU0)
CFLAGS = -fopenmp -g3 -Wall -Drestrict=__restrict__ -O0 -ffast-math
#CFLAGS = -g3 -Wall -Drestrict=__restrict__ -O0 -ffast-math
#NVCCFLAGS = $(NVFLAG_BASE) -g -G -O0
NVCCFLAGS = $(NVFLAG_BASE) -g -O0
LDFLAGS = -fopenmp -g # -g -pg
LIBS := $(LIBS) -lnuma
endif
ifeq ($(SW), UVGNU3)
MPICC = mpicc-gcc4.7
MPICPP = mpicxx-gcc4.7
CFLAGS = -fopenmp -g -Wall -Drestrict=__restrict__ -O3 -DNDEBUG -ffast-math # -g -pg
NVCCFLAGS = $(NVFLAG_BASE) -g -O3 -DNDEBUG -keep
LDFLAGS = -fopenmp -g # -g -pg
LIBS = -lm -lnuma -lmpi
endif
ifeq ($(SW), UVGNU0)
MPICC = mpicc-gcc4.7
MPICPP = mpicxx-gcc4.7
CFLAGS = -fopenmp -g3 -Wall -Drestrict=__restrict__ -O0 -ffast-math
#CFLAGS = -g3 -Wall -Drestrict=__restrict__ -O0 -ffast-math
#NVCCFLAGS = $(NVFLAG_BASE) -g -G -O0
NVCCFLAGS = $(NVFLAG_BASE) -g -O0
LDFLAGS = -fopenmp -g # -g -pg
LIBS = -lm -lnuma -lmpi
endif
ifeq ($(SW), INTEL3)
MPICC = icc
MPICPP = icpc
CFLAGS = -ipo -no-prec-div -openmp -g3 -Drestrict=__restrict__ -O3 -DNDEBUG
#CFLAGS = -ipo -openmp -g3 -Drestrict=__restrict__ -O2 -DNDEBUG
NVCCFLAGS = $(NVFLAG_BASE) -g -O3 -DNDEBUG -keep
LDFLAGS = -openmp -g
LIBS = -lm -lnuma -lmpi
endif
ifeq ($(SW), INTEL0)
MPICC = icc
MPICPP = icpc
CFLAGS = -ipo -no-prec-div -openmp -g3 -Drestrict=__restrict__ -O0
NVCCFLAGS = $(NVFLAG_BASE) -g -G -O0
LDFLAGS = -openmp -g
LIBS = -lm -lnuma -lmpi
endif
ifeq ($(SW), FXGNU3)
MPICC = gcc
MPICPP = g++
CFLAGS = -fopenmp -g -Wall -Drestrict=__restrict__ -O3 -DNDEBUG -ffast-math # -g -pg
NVCCFLAGS = $(NVFLAG_BASE) -g -O3 -DNDEBUG -keep
LDFLAGS = -fopenmp -g # -g -pg
INC := $(INC) -I/opt/FJSVtclang/GM-1.2.0-16/include/mpi/fujitsu
LIBS = -L/opt/FJSVtclang/GM-1.2.0-16/lib64 -lm -lmpi -lmpi_cxx #-ltrt -ltrtmetcpp -ltrtmet_c
endif
ifeq ($(SW), FXGNU0)
MPICC = gcc
MPICPP = g++
CFLAGS = -fopenmp -g3 -Wall -Drestrict=__restrict__ -O0 -ffast-math
#CFLAGS = -g3 -Wall -Drestrict=__restrict__ -O0 -ffast-math
#NVCCFLAGS = $(NVFLAG_BASE) -g -G -O0
NVCCFLAGS = $(NVFLAG_BASE) -g -O0
LDFLAGS = -fopenmp -g # -g -pg
INC := $(INC) -I/opt/FJSVtclang/GM-1.2.0-16/include/mpi/fujitsu
LIBS = -L/opt/FJSVtclang/GM-1.2.0-16/lib64 -lm -lmpi -lmpi_cxx #-ltrt -ltrtmetcpp -ltrtmet_c
endif
ifeq ($(SW), FUJI3)
ifeq ($(PJM_JOBNAME), STDIN)
MPICC = mpifcc
MPICPP = mpiFCC
else
MPICC = mpifccpx
MPICPP = mpiFCCpx
endif
CFLAGS = -Kopenmp -Xg -Kfast -Klargepage -Drestrict=__restrict__ -O3 -DNDEBUG -DNNUMA # -Koptmsg=2
LDFLAGS = $(CFLAGS)
LIBS = -lm
endif
ifeq ($(SW), FUJI0)
ifeq ($(PJM_JOBNAME), STDIN)
MPICC = mpifcc
MPICPP = mpiFCC
else
MPICC = mpifccpx
MPICPP = mpiFCCpx
endif
CFLAGS = -Kopenmp -Koptmsg=2 -Xg -g -Drestrict=__restrict__ -O0 -DNNUMA
#CFLAGS = -Koptmsg=2 -Xg -g -Drestrict=__restrict__ -O0 -DNNUMA
LDFLAGS = -Kopenmp -Koptmsg=2 -Xg -g
LIBS = -lm
endif

ifeq ($(VAMP), true)
CC = vtcc -vt:cc $(MPICC) -vt:inst manual -vt:hyb -DVTRACE
CPP = vtcc -vt:cc $(MPICPP) -vt:inst manual -vt:hyb -DVTRACE
else
CC = $(MPICC)
CPP = $(MPICPP)
endif

CPPFLAGS = $(CFLAGS)

all: cpu

cpu: $(OBJS)
	$(CPP) $(LDFLAGS) -o runnable $(OBJS) $(LIBS)
	
gpu: $(OBJS) $(CUOBJ)
	$(CPP) $(LDFLAGS) -o runnable $(OBJS) $(CUOBJ) $(LIBS) $(CUDALIB)
	
gnu_func.o: gnu_func.cc
	cp ../../$*.o_ $*.o
#	g++ -c -g -O3 -Wall $< -o $*.o

.SUFFIXES: .o .c .cpp .cc .cu

.c.o:
	$(CC) $(INC) -c $(CFLAGS) $< -o $*.o
	
.cpp.o:
	$(CPP) $(INC) -c $(CPPFLAGS) $< -o $*.o
	
.cc.o:
	$(CPP) $(INC) -c $(CPPFLAGS) $< -o $*.o
	
.cu.o:
	$(NVCC) -c $(NVCCFLAGS) $< -o $*.o
	
.PHONY: clean
clean:
	-rm -f $(BINS) $(OBJS) $(CUOBJ)

