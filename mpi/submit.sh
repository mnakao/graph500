#!/bin/bash
#PJM -L "node=64"               # C x R
#PJM -L "rscunit=rscunit_ft01"   # リソースユニットの指定
#PJM -L "rscgrp=dvsmall"           # リソースグループの指定
#PJM -L "elapse=0:10:00"
#PJM --name "n"

PROBLEM_SIZE=30
export TWOD_R=8
mpiexec -mca btl_tofu_eager_limit 512000 -mca mpi_print_stats 3 runnable ${PROBLEM_SIZE}
echo $SECONDS
