#! /usr/bin/env bash
BASE_PATH=csrc/mpi_cuda_helloworld
KERNEL_PATH=$BASE_PATH/split_kernel
KERNEL_OBJ_FILE=$KERNEL_PATH.o
DRIVER_OBJ_FILE=$BASE_PATH/split_driver.o
nvcc -c $KERNEL_PATH.cu -o $KERNEL_OBJ_FILE -gencode=arch=compute_86,code=sm_86
mpic++ -c $BASE_PATH/split_driver.cpp -o $DRIVER_OBJ_FILE -I/usr/local/cuda/include
mpic++ $KERNEL_OBJ_FILE $DRIVER_OBJ_FILE -lcudart -L/usr/local/cuda/lib64/ -o split.bin