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
- [ ] Get CUDA running on Modal
    - Abandoned, too much effort
- [X] Figure out if register_graph_buffer is only used in tests
    - No, it's used in the Python bindings

- Each *buffer* has a `RankData`
- > The rank data is an array to store the registered IPC addresses for all ranks for each input address passed in. 
- `buffers_` is (pointer => `RankData`), the pointer is to an input on the current rank?
    - The input in the CUDA test is `self_data`, which is the second section of the main test buffer
    - > The first section is a temporary buffer for storing intermediate allreduce results, if a particular algorithm requires it. The second section is for the input to the allreduce
- Thus, `std::unordered_map<void *, RankData *> buffers_;` is map of (pointer on current rank) => (pointers on all ranks)
- Also, `RankData *` is a pointer to `RankData` stored on GPU memory
- `*d_rank_data_base` and `*d_rank_data_end` are pointers that mark the start and end, respectively, of a segment of GPU memory allocated for storing `RankData` instances. As new `RankData` instances are copied to the device, `*d_rank_data_base` is incremented, effectively moving the 'start' pointer forward. This means that `*d_rank_data_base` always points to the next available location within the allocated memory segment where new RankData can be copied.
- `ipc_handles_` stores all the ipc handles so they can be closed once the `FastAllreduce` class is destroyed
- `Metadata` stores a `Signal` + a counter (both for the current rank). The `Signal` contains a start/end field, both of which are a `union` of 64-bit int and 8-bytes. The `Signal` is a synchronization primitive.
- `RankSignals` consists of an array of 8 device-pointers, each pointing to a `Signal` on a different rank
- `RankSignals` itself + the device-pointers are stored in CPU memory

- [X] Understand the test data
   - nccl reduces `self_data_copy` => `self_data`
   - custom impl reduces `self_data` => `result`
   - `self_data` is the 2nd section of the buffer
- [ ] Understand synchronization primitive

- [ ] Understand the rank data array
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