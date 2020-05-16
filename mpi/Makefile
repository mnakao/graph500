MPICC := mpifccpx
MPICPP := mpiFCCpx
FCC_BASE := -Kopenmp -Drestrict=__restrict__
CFLAGS := $(FCC_BASE) -Kfast -g
LDFLAGS := $(CFLAGS) -Nfjomplib
CPPFLAGS := $(CFLAGS)

BINS := runnable
OBJS := main.o ../generator/splittable_mrg.o

all: cpu
cpu: $(OBJS)
	$(MPICPP) $(LDFLAGS) -o runnable $(OBJS) $(LIBS)

.SUFFIXES: .o .c .cc

.c.o:
	$(MPICC) $(INC) -c $(CFLAGS) $< -o $*.o

.cc.o:
	$(MPICPP) $(INC) -c $(CPPFLAGS) $< -o $*.o

.PHONY: clean
clean:
	-rm -f $(BINS) $(OBJS) $(CUOBJ)
