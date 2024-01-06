#!/bin/bash
LOG=$1.$OMPI_COMM_WORLD_RANK
cuda-memcheck --log-file $LOG.log --save $LOG.memcheck $*

# use w/
# mpiexec -np 2 cuda-memcheck-script.bash ./myapp <args>
# mpiexec --allow-run-as-root -np 2 ./scripts/cuda-memcheck-script.bash ./sync_test.bin