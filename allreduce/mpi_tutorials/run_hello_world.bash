#! /usr/bin/env bash
set -euxo pipefail
mpicc -o hello_world.bin hello_world.c
mpirun -np 8 ./hello_world.bin