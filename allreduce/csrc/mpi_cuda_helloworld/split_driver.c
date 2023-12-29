#include <stdio.h>
#include <mpi.h>

// Declare the function from the CUDA file
extern void gpu_work(int rank, int world_size);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    gpu_work(rank, world_size);

    MPI_Finalize();
    return 0;
}
