#!/bin/bash
#PJM -L "node=64"               # C x R
#PJM -L "rscunit=rscunit_ft01"   # リソースユニットの指定
#PJM -L "rscgrp=dvsmall"           # リソースグループの指定
#PJM -L "elapse=0:10:00"
#PJM --name "n"

export TWOD_R=8
mpiexec ./runnable 30
echo $SECONDS
