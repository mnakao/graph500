MPICC := mpifccpx
MPICPP := mpiFCCpx
FCC_BASE := -Kopenmp -Drestrict=__restrict__ -DVERTEX_REORDERING=2 -DVERVOSE_MODE=0
CFLAGS := $(FCC_BASE) -Kfast -DNDEBUG
LDFLAGS := $(CFLAGS) -Nfjomplib
CPPFLAGS := $(CFLAGS)

BINS := runnable
OBJS := main.o low_level_func.o ../generator/splittable_mrg.o

all: cpu
cpu: $(OBJS)
	$(MPICPP) $(LDFLAGS) -o runnable $(OBJS) $(LIBS)

gnu_func.o: gnu_func.cc
	cp ../../$*.o_ $*.o

.SUFFIXES: .o .c .cpp .cc .cu

.c.o:
	$(MPICC) $(INC) -c $(CFLAGS) $< -o $*.o

.cpp.o:
	$(MPICPP) $(INC) -c $(CPPFLAGS) $< -o $*.o

.cc.o:
	$(MPICPP) $(INC) -c $(CPPFLAGS) $< -o $*.o

.PHONY: clean
clean:
	-rm -f $(BINS) $(OBJS) $(CUOBJ)
