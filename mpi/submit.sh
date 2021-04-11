#!/bin/bash
#PJM -L "node=16x32"               # C x R
#PJM -L "elapse=2:00:00"
#PJM --mpi "max-proc-per-node=1" # 1ノードあたりに生成するMPIプロセス数の上限値
#PJM -L "rscgrp=large"
##PJM -s

PROBLEM_SIZE=33
#PROBLEM_SIZE=30
R=32
NUM_NODES=512
NUM_PROCESSES=${NUM_NODES}
################
export OMP_NUM_THREADS=48
export TWOD_R=${R}
K=1.0
L=1.0
OUTPUT_FILE=n${NUM_NODES}s${PROBLEM_SIZE}L${L}K${K}
mpiexec -std ${OUTPUT_FILE} -mca mpi_print_stats 1 ./runnable ${PROBLEM_SIZE} ${L} ${K}
  
for K in 0.01 0.02 0.05 0.1 0.2 0.5; do
  OUTPUT_FILE=n${NUM_NODES}s${PROBLEM_SIZE}L${L}K${K}
  mpiexec -std ${OUTPUT_FILE} -mca mpi_print_stats 1 ./runnable ${PROBLEM_SIZE} ${L} ${K}
done

K=1.0
for L in 0.01 0.02 0.05 0.1 0.2 0.5; do
  OUTPUT_FILE=n${NUM_NODES}s${PROBLEM_SIZE}L${L}K${K}
  mpiexec -std ${OUTPUT_FILE} -mca mpi_print_stats 1 ./runnable ${PROBLEM_SIZE} ${L} ${K}
done
echo $SECONDS >> ${OUTPUT_FILE}

