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

Ok. If I'm doing this for real. Trading off against *all the other things I could be doing*, let's get serious.

- [X] Roughly understand the barrier code
- Grid shape: `<<<numBlocks, numThreads, shared mem, stream>>>`
- 1D block grid + 1D thread grid per block (evenly splitting `size` over the # of blocks)
- `packed_t` explanation:
```cpp
/*
Maximize memory efficiency w/ ld.128,st.128 instructions
The GPU cannot load more than 128 bytes at a time
CUDA threads can read a maximum of 16 bytes at once
So, each thread loads as many elements of type T as can be accommodated within 16 bytes
https://stackoverflow.com/questions/72147025/what-are-cuda-global-memory-32-64-and-128-byte-transactions
*/
template <typename T>
struct packed_t {
  // the (P)acked type for load/store
  using P = array_t<T, 16 / sizeof(T)>;
  // the (A)ccumulator type for reduction
  using A = array_t<float, 16 / sizeof(T)>;
};
```

Start sync:
- The 1st block in each rank's grid uses its 1st warp to write to the signal in all other ranks and its 2nd warp to reset the end sync flag
- All threads busy-wait until the 1st block per rank has written to all other ranks
- *I guess this is 1st block synchronization?*

End sync:
- The final block to *reach* the sync point uses its 2nd warp to reset the start flag and its 1st warp to write to the signal across all other ranks
- Its important we use the final block to *reach* the sync point for writing to the signal across all other ranks. 
- To illustrate the above point, consider a scenario where a rank has three blocks: block0, block1, and block2. Let's assume that block0 is used to write to the signal and the blocks reach the sync point in the order of block0, block1, block2. In this case, block0 and block1 would proceed to the next task before block2 has necessarily completed its previous task. Similarly, block0 would move on to the next task before block1 has necessarily completed its previous task.
- **QUESTION**: If the kernel exit can serve as a sync, then why even sync at all?

- [X] Understand where the metadata is stored for the Pytorch bindings
   - [X] Read the torch binding functions
      - Expectation (seems confirmed?):
      - 1 metadata: place to store signals + counter for current rank
      - a register_buffer function / something to create new rank data (requires passing in IPC handles to torch)

```python
def manual_registration(world_size, rank, distributed_init_port):
    init_test_distributed_environment(1, world_size, rank,
                                      distributed_init_port)
    sz = 1024
    fast_ar.init_fast_ar()
    fa = fast_ar.get_handle()
    inp = torch.ones(sz,
                     dtype=torch.float32,
                     device=torch.cuda.current_device())
    fa.register_buffer(inp)
    out = fa.all_reduce(inp)
    assert torch.allclose(out, inp * world_size)
```
- [X] Understand where `init_fast_ar` and `get_handle` are called
```python
class FastAllreduce:

    # max_size: max supported allreduce size
    def __init__(self, rank, world_size, max_size=8192 * 1024) -> None:
        # buffers memory are owned by this Python class and passed to C++
        self.meta = torch.zeros(fast_ar.meta_size() + max_size,
                                dtype=torch.uint8,
                                device="cuda")
        self.rank_data = torch.empty(16 * 1024 * 1024,
                                     dtype=torch.uint8,
                                     device="cuda")
        self.max_size = max_size
        self.world_size = world_size
        handles, offsets = self._get_ipc_meta(self.meta)
        self.full_nvlink = _is_full_nvlink(rank, world_size)
        self._ptr = fast_ar.init_fast_ar(self.meta, self.rank_data, handles,
                                         offsets, rank, self.full_nvlink)
        self.fast_cond = self.full_nvlink or world_size <= 2

    def _get_ipc_meta(self, inp: torch.Tensor):
        data = inp.storage()._share_cuda_()
        shard_data = (
            data[1],  # ipc handle to base ptr
            data[3],  # offset of base ptr
        )
        return self._gather_ipc_meta(shard_data)

    def _gather_ipc_meta(self, shard_data):
        all_data = [None] * self.world_size
        dist.all_gather_object(all_data, shard_data)

        handles = []
        offsets = []
        for i in range(len(all_data)):
            handles.append(all_data[i][0])
            offsets.append(all_data[i][1])
        return handles, offsets

    def register_buffer(self, inp: torch.Tensor):
        handles, offsets = self._get_ipc_meta(inp)
        fast_ar.register_buffer(self._ptr, inp, handles, offsets)
```
- [X] Understand the pytorch bindings for single buffer registration
   - Pytorch offers an API, `_share_cuda_`, to get the IPC handle + offset of a given tensor. (I'm guessing CUDA memory is allocated in large blocks, and tensors are given subsections)

Return values of `_share_cuda_`:
```cpp
  PyTuple_SET_ITEM(tuple.get(), 0, device.release());
  // cudaIpcMemHandle_t(of basePtr)
  PyTuple_SET_ITEM(tuple.get(), 1, _handle.release());
  // Size(in bytes) of the real storage, note this is not the size of basePtr
  // memory block.
  PyTuple_SET_ITEM(tuple.get(), 2, size_bytes.release());
  // Offset(in bytes) of the real storage in the basePtr memory block.
  // NB: this offset MUST be in bytes instead of numel, since we use
  // (storage_handle, offset)
  //     as key in shared_cache(multiprocessing/reduction.py).
  //     Offset in numel cannot uniquely represent a storage.
  PyTuple_SET_ITEM(tuple.get(), 3, _offset_bytes.release());
  PyTuple_SET_ITEM(tuple.get(), 4, _ref_counter.release());
  PyTuple_SET_ITEM(tuple.get(), 5, _ref_counter_offset.release());
  PyTuple_SET_ITEM(tuple.get(), 6, _event_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 7, _event_sync_required.release());
```

- [X] Email author asking why `end_sync` is needed? *Anything else I should email him on?*
- [ ] 

- [ ] Sanity-check by adding print statements + nanosleep
- [ ] Implement the synchronization primitive
- [ ] Write a CUDA test verifying that it works