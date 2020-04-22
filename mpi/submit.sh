#!/bin/bash
#PJM -L "node=16x32"               # C x R
#PJM -L "rscunit=rscunit_ft01"   # リソースユニットの指定
#PJM -L "rscgrp=dvall"           # リソースグループの指定
#PJM -L "elapse=0:15:00"
#PJM --mpi "max-proc-per-node=1" # 1ノードあたりに生成するMPIプロセス数の上限値
##PJM -s

PROBLEM_SIZE=31
R=32
NUM_NODES=512
NUM_PROCESSES=${NUM_NODES}
################
OUTPUT_FILE=n${NUM_NODES}p${NUM_PROCESSES}s${PROBLEM_SIZE}
export OMP_NUM_THREADS=48
export TWOD_R=${R}
#mpiexec -std ${OUTPUT_FILE} ./runnable ${PROBLEM_SIZE}
mpiexec -std ${OUTPUT_FILE} -mca mpi_print_stats 1 ./runnable ${PROBLEM_SIZE}
echo $SECONDS >> ${OUTPUT_FILE}

