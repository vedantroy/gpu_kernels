## Compilation
### add_one
- modal (8 cpu) = ~70s
- laptop = ~56s
- vast (ryzen 9) = ~45s
- vast (ryzen 9, no optimization) = ~40s
    - ~ 38s to run add_one.o
    - ~ 20s to compile pybind.o

I suspect most of the compilation time for the very simple add_one extension comes from the torch headers. Let me benchmark compiling a simple kernel that doesn't use torch at all.

### mpi_cuda_helloworld
Installation:
- `sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev`
- `sudo apt install libnccl-dev libnccl2`
- Use `mpicc -show` to find where your mpi headers are

Runtime is near instant. *Torch extension process must have been adding a lot of overhead*
Run with `mpirun --allow-run-as-root -np 2 ./fastallreduce_test.bin`

## File Structure
- fast_allreduce.cuh => implements allreduce w/o touching Pytorch
- fast_allreduce.cu => torch bindings
- fast_all_reduce_test.cu => driver to test (does not include fast_allreduce.cu)

## TODOs
[01-01-24]
- [ ] Get CUDA running on Modal (*abandoned -- too much effort*)
- [ ] Figure out if register_graph_buffer is only used in tests
- [ ] Understand how rank data works (how much is allocated / why is it allocated?)

- [ ] Understand basic MPI programming model
- [ ] Play a bit w/ launching MPI + the basic MPI programs
- Run MPI allgather on CPU, figure out how to run it on 2/4 GPUs w/ modal

- figure out how to compile the CUDA test & run it on modal
    - how to compile w/ mpi + torch
    - how to run w/ mpi
- write a small standalone example that uses torch + mpi. Maybe it does a simple mpi_allgather on 2 tensors

- [ ] Do an all gather on 2 GPUs on Modal
    - [ ] How to run MPI on multiple GPUs ...
    - What to install
    - Skypilot? Tensor Dock? Modal?