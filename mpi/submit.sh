#!/bin/bash
#PJM --rsc-list "node=4"
#PJM --rsc-list "rscunit=rscunit_ft01"
#PJM --rsc-list "rscgrp=dvgb1152"
#PJM --rsc-list "elapse=00:10:00"
#PJM --mpi "proc=4"
#####
PROCS=4
SCALE=23
PROCS_PER_NODE=1
####
NODES=$(( $PROCS / $PROCS_PER_NODE ))
THREADS=$(( 48 / $PROCS_PER_NODE ))
#####

module load lang
export OMP_NUM_THREADS=${THREADS}
export MPI_NUM_NODE=${PROCS_PER_NODE}
#export XOS_MMM_L_HPAGE_TYPE=none
F=n${NODES}p${PROCS}t${THREADS}s${SCALE}
EXE=./runnable
for i in $(seq 4 10); do
  mpiexec --stdout ${F}i${i}.out --stderr ${F}i${i}.err ${EXE} ${SCALE}
done

echo $SECONDS
