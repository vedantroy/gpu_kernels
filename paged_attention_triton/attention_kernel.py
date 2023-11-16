from einops import rearrange
import triton
import triton.language as tl

# Expect block table to map
# logical bid (block id) -> (physical bid, # filled)
# In tests, it maps: logical bid -> physical bid


@triton.jit
def paged_attention(
    debug_block_idxs_ptr,
    debug_key_cache_load_ptr,
    debug_key_cache_load_ptr2,
    debug_block_idx_ptr2,
    debug_key_cache_load_ptr3,
    debug_key_cache_load_ptr4,
    debug_key_cache_load_ptr5,
    debug_scores_ptr,
    debug_softmax_ptr,
    debug_output_ptr,

    # need these b/c we can't use view/reshape
    scratchpad_key_ptr,  # [num_seqs, max_context_len, num_heads, head_size]
    scratchpad_value_ptr,  # [num_seqs, max_context_len, num_heads, head_size]
    output_ptr,  # [num_seqs, num_query_heads, head_size]
    query_ptr,  # [num_seqs, num_query_heads, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size, block_size]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, block_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    context_lens_ptr,  # [num_seqs]
    scale,  # float32
    num_seqs,  # int
    num_heads,  # int
    cache_block_stride,  # int
    MAX_CONTEXT_LEN: tl.constexpr,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    MAX_NUM_BLOCKS_PER_SEQ: tl.constexpr,  # int, must be power of 2
):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    query_offset = seq_idx * num_seqs + head_idx * HEAD_SIZE
    query_head = tl.load(query_ptr + query_offset + tl.arange(0, HEAD_SIZE))
    block_table_offset = seq_idx * MAX_NUM_BLOCKS_PER_SEQ

    # physical_block_idxs = tl.load(
    #     block_tables_ptr + block_table_offset + tl.arange(0, MAX_NUM_BLOCKS_PER_SEQ)
    # )

    # if seq_idx == 0 and head_idx == 0:
    #     tl.store(
    #         debug_block_idxs_ptr + tl.arange(0, MAX_NUM_BLOCKS_PER_SEQ),
    #         physical_block_idxs,
    #     )

    # start_of_block_offsets = (physical_block_idxs * cache_block_stride) + (
    #     head_idx * HEAD_SIZE * BLOCK_SIZE
    # )

    # # Test #2 with:
    # if seq_idx == 0 and head_idx == 1:
    #     sample_values = tl.load(key_cache_ptr + start_of_block_offsets)
    #     tl.store(
    #         debug_key_cache_load_ptr + tl.arange(0, MAX_NUM_BLOCKS_PER_SEQ),
    #         sample_values,
    #     )

    # Works, but can't transform [max_num_blocks_per_seq, head_size, block_size] => [seq_len, head_size]
    # https://github.com/openai/triton/discussions/2666
    # https://github.com/openai/triton/issues/2522

    # key_block_offsets = (
    #     start_of_block_offsets[:, None, None]
    #     + (BLOCK_SIZE * tl.arange(0, HEAD_SIZE)[None, :, None])
    #     + (1 * tl.arange(0, BLOCK_SIZE)[None, None, :])
    # )

    # # shape = [max_num_blocks_per_seq, head_size, block_size]
    # key_block = tl.load(key_cache_ptr + key_block_offsets)
    # if seq_idx == 0 and head_idx == 1:
    #     store_offsets = (
    #         (BLOCK_SIZE * HEAD_SIZE * tl.arange(0, MAX_NUM_BLOCKS_PER_SEQ)[:, None, None])
    #         + (BLOCK_SIZE * tl.arange(0, HEAD_SIZE)[None, :, None])
    #         + (1 * tl.arange(0, BLOCK_SIZE)[None, None, :])
    #     )
    #     tl.store(
    #         debug_key_cache_load_ptr2 + store_offsets,
    #         key_block
    #     )

    context_len = tl.load(context_lens_ptr + seq_idx)

    # Can't allocate memory that's not known at compile time
    # (We could make it known @ compile time by making context_len a tl.constexpr)
    # seq_keys = tl.zeros((context_len, HEAD_SIZE), dtype=tl.float32)
    # seq_values = tl.zeros((context_len, HEAD_SIZE), dtype=tl.float32)

    for tok_idx in range(0, context_len):
        logical_block_idx = tok_idx // BLOCK_SIZE
        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + logical_block_idx
        )

        if (tok_idx == 0 and seq_idx == 0) and (head_idx == 1):
            tl.store(debug_block_idx_ptr2, physical_block_idx)

        start_of_block_offset = (
            physical_block_idx * cache_block_stride + head_idx * HEAD_SIZE * BLOCK_SIZE
        )
        tok_idx_within_block = tok_idx % BLOCK_SIZE
        tok_offsets = (
            start_of_block_offset
            + BLOCK_SIZE * tl.arange(0, HEAD_SIZE)
            + tok_idx_within_block
        )

        tok_key = tl.load(key_cache_ptr + tok_offsets)
        tok_value = tl.load(value_cache_ptr + tok_offsets)

        if (tok_idx == 0 and seq_idx == 0) and (head_idx == 0):
            tl.store(debug_key_cache_load_ptr3 + tl.arange(0, HEAD_SIZE), tok_key)

        if (tok_idx == 1 and seq_idx == 0) and (head_idx == 0):
            tl.store(debug_key_cache_load_ptr4 + tl.arange(0, HEAD_SIZE), tok_key)

        if (tok_idx == 7 and seq_idx == num_seqs - 1) and (head_idx == 0):
            tl.store(debug_key_cache_load_ptr5 + tl.arange(0, HEAD_SIZE), tok_key)

        scratchpad_offset = (
            seq_idx * (MAX_CONTEXT_LEN * num_heads * HEAD_SIZE)
            + tok_idx * (num_heads * HEAD_SIZE)
            + head_idx * HEAD_SIZE
        )
        tl.store(
            scratchpad_key_ptr + scratchpad_offset + tl.arange(0, HEAD_SIZE), tok_key
        )
        tl.store(
            scratchpad_value_ptr + scratchpad_offset + tl.arange(0, HEAD_SIZE),
            tok_value,
        )

    # TODO: Not sure if this is necessary
    tl.debug_barrier()

    # scratchpad_key_ptr,  # [num_seqs, max_context_len, num_heads, head_size]
    start_seq_offset = (MAX_CONTEXT_LEN * num_heads * HEAD_SIZE) * seq_idx
    start_tok_offset = start_seq_offset + tl.arange(0, MAX_CONTEXT_LEN) * (num_heads * HEAD_SIZE) + head_idx * HEAD_SIZE

    # [seq_len, head_size]
    # zero out keys that aren't part of the sequence
    mask = tl.arange(0, MAX_CONTEXT_LEN)[:, None] < context_len
    kv_offs = start_tok_offset[:, None] + tl.arange(0, HEAD_SIZE)[None, :]
    keys = tl.load(scratchpad_key_ptr + kv_offs, mask=mask, other=0.0)
    values = tl.load(scratchpad_value_ptr + kv_offs, mask=mask, other=0.0)

    # keys shape  [seq_len x head_size], query shape = [head_size]

    # Can't do below b/c minimum size on all dimensions is 16
    # scores = tl.dot(query_head[None, :], keys.T)

    # Workaround for matrix, vector dot product
    # shape = [seq_len]
    # tmp_scores = tl.zeros([MAX_CONTEXT_LEN], dtype=tl.float32)
    scores = tl.sum(keys * query_head[None, :], axis=1)

    # This mask is necessary b/c even though we mask out the keys on load
    # that just results in 0s in the attention dot product, 
    # which then get softmaxed and result in non-zero values 
    # in the softmax output (which is wrong)
    # -inf guarantees that the softmax output will be 0 for masked values
    mask = tl.full([MAX_CONTEXT_LEN], -float('inf'), dtype=tl.float32)
    cond = tl.arange(0, MAX_CONTEXT_LEN) < context_len
    scores_masked = tl.where(cond, scores, mask)

    if seq_idx == 0 and head_idx == 0:
        # tl.store(debug_scores_ptr + tl.arange(0, MAX_CONTEXT_LEN), scores)
        tl.store(debug_scores_ptr + tl.arange(0, MAX_CONTEXT_LEN), scores_masked)

    # do a numerically stable softmax on the scores
    scores_minus_max = scores_masked - tl.max(scores_masked, axis=0)
    numerator = tl.exp(scores_minus_max)
    denominator = tl.sum(numerator, axis=0)
    logits = numerator / denominator

    if seq_idx == 0 and head_idx == 0:
        tl.store(debug_softmax_ptr + tl.arange(0, MAX_CONTEXT_LEN), logits)

    weighted_values = tl.sum(values * logits[:, None], axis=0)

    if seq_idx == 0 and head_idx == 0:
        tl.store(debug_output_ptr + tl.arange(0, HEAD_SIZE), weighted_values)

    output_offset = seq_idx * (num_heads * HEAD_SIZE) + head_idx * HEAD_SIZE
    tl.store(output_ptr + output_offset + tl.arange(0, HEAD_SIZE), weighted_values)


def test_triton_paged_attention():
    import random
    import torch

    # num_blocks_in_cache = 32

    # block_size = 32
    # seed = 0
    # head_size = 64
    # num_heads = (8, 8)
    # num_seqs = 8
    # max_seq_len = 512

    num_blocks_in_cache = 8

    block_size = 2
    seed = 0
    head_size = 4
    num_heads = (2, 2)
    num_seqs = 2
    max_seq_len = 8

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(
        num_seqs, num_query_heads, head_size, dtype=torch.float32, device="cuda"
    )
    query.uniform_(-scale, scale)
    output = torch.empty_like(query, device="cuda")

    cache_shape = (num_blocks_in_cache, num_query_heads, head_size, block_size)

    key_cache = torch.empty(cache_shape, dtype=torch.float32, device="cuda")
    key_cache.uniform_(-scale, scale)
    assert key_cache.stride(0) == num_query_heads * head_size * block_size

    value_cache = torch.empty(cache_shape, dtype=torch.float32, device="cuda")
    value_cache.uniform_(-scale, scale)

    context_lens = torch.tensor(
        [random.randint(1, max_seq_len) for _ in range(num_seqs)], device="cuda"
    )
    context_lens[-1] = max_seq_len
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables = [
        [
            random.randint(0, num_blocks_in_cache - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        for _ in range(num_seqs)
    ]
    block_tables = torch.tensor(block_tables, dtype=torch.int, device="cuda")

    # create tensor of all 0s of size 16
    debug_block_idxs = torch.zeros(
        max_num_blocks_per_seq, dtype=torch.int, device="cuda"
    )
    debug_key_cache_load = torch.zeros(
        max_num_blocks_per_seq, dtype=key_cache.dtype, device="cuda"
    )
    debug_key_cache_load2 = torch.zeros(
        max_num_blocks_per_seq,
        head_size,
        block_size,
        dtype=torch.float32,
        device="cuda",
    )
    debug_block_idx_ptr2 = torch.zeros(1, dtype=torch.int, device="cuda")
    debug_key_cache_load3 = torch.zeros(head_size, dtype=torch.float32, device="cuda")
    debug_key_cache_load4 = torch.zeros(head_size, dtype=torch.float32, device="cuda")
    debug_key_cache_load5 = torch.zeros(head_size, dtype=torch.float32, device="cuda")
    debug_scores = torch.zeros(max_seq_len, dtype=torch.float32, device="cuda")
    debug_softmax = torch.zeros(max_seq_len, dtype=torch.float32, device="cuda")
    debug_output_ptr = torch.zeros(head_size, dtype=torch.float32, device="cuda")

    scratchpad_key = torch.zeros(
        (num_seqs, max_seq_len, num_query_heads, head_size),
        dtype=torch.float32,
        device="cuda",
    )
    scratchpad_value = torch.zeros_like(scratchpad_key)

    paged_attention[(num_seqs, num_query_heads)](
        debug_block_idxs_ptr=debug_block_idxs,
        debug_key_cache_load_ptr=debug_key_cache_load,
        debug_key_cache_load_ptr2=debug_key_cache_load2,
        debug_block_idx_ptr2=debug_block_idx_ptr2,
        debug_key_cache_load_ptr3=debug_key_cache_load3,
        debug_key_cache_load_ptr4=debug_key_cache_load4,
        debug_key_cache_load_ptr5=debug_key_cache_load5,
        debug_scores_ptr=debug_scores,
        debug_softmax_ptr=debug_softmax,
        debug_output_ptr=debug_output_ptr,

        scratchpad_key_ptr=scratchpad_key,
        scratchpad_value_ptr=scratchpad_value,
        output_ptr=output,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=block_tables,
        context_lens_ptr=context_lens,
        scale=scale,
        num_seqs=num_seqs,
        num_heads=num_query_heads,
        cache_block_stride=key_cache.stride(0),
        MAX_CONTEXT_LEN=max_seq_len,
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        MAX_NUM_BLOCKS_PER_SEQ=max_num_blocks_per_seq,
    )

    torch.cuda.synchronize()
    # torch.testing.assert_close(debug_block_idxs, block_tables[0])
    # torch.testing.assert_close(
    #     debug_key_cache_load, key_cache[block_tables[0], 1, 0, 0]
    # )

    # seq_0_head_1_keys = key_cache[block_tables[0], 1]
    # torch.testing.assert_close(seq_0_head_1_keys, debug_key_cache_load2)

    assert debug_block_idx_ptr2[0] == block_tables[0, 0]

    seq0_tok0_head0_key = key_cache[block_tables[0, 0], 0, :, 0]
    torch.testing.assert_close(debug_key_cache_load3, seq0_tok0_head0_key)

    seq0_tok1_head0_key = key_cache[block_tables[0, 0], 0, :, 1]
    torch.testing.assert_close(debug_key_cache_load4, seq0_tok1_head0_key)

    last_seq_tok7_head0_key = key_cache[
        block_tables[num_seqs - 1, 7 // block_size], 0, :, 7 % block_size
    ]
    torch.testing.assert_close(debug_key_cache_load5, last_seq_tok7_head0_key)

    seq0_len = context_lens[0]
    seq0_head0_keys = key_cache[block_tables[0], 0]
    divide_round_up = lambda x, y: (x + y - 1) // y
    seq0_num_blocks = divide_round_up(seq0_len, block_size)
    assert seq0_head0_keys.shape == (seq0_num_blocks, head_size, block_size)
    seq0_head0_keys = rearrange(
        seq0_head0_keys,
        "num_blocks head_size block_size -> (num_blocks block_size) head_size",
    )
    assert seq0_head0_keys.shape == (seq0_num_blocks * block_size, head_size)
    seq0_head0_keys_clipped = seq0_head0_keys[:seq0_len]
    assert seq0_head0_keys_clipped.shape == (seq0_len, head_size)
    torch.testing.assert_close(seq0_head0_keys_clipped, scratchpad_key[0, :seq0_len, 0, :])

    # do dot product of query & keys
    scores = seq0_head0_keys @ query[0, 0]
    assert scores.shape == debug_scores.shape
    # emulate triton's masking
    scores[-1] = -float('inf')
    torch.testing.assert_close(scores[:-1], debug_scores[:-1])

    expected_softmax = torch.softmax(scores, dim=0)
    torch.testing.assert_close(debug_softmax, expected_softmax)

    seq0_head0_values = value_cache[block_tables[0], 0]
    seq0_head0_values = rearrange(
        seq0_head0_values,
        "num_blocks head_size block_size -> (num_blocks block_size) head_size",
    )
    assert seq0_head0_values.shape == (seq0_num_blocks * block_size, head_size)
    seq0_head0_values_clipped = seq0_head0_values[:seq0_len]
    assert seq0_head0_values_clipped.shape == (seq0_len, head_size)
    torch.testing.assert_close(
        seq0_head0_values_clipped, scratchpad_value[0, :seq0_len, 0, :]
    )

    expected_output = seq0_head0_values.T @ expected_softmax
    torch.testing.assert_close(expected_output, debug_output_ptr)

    # load output from correct place in output_ptr (ensure location is right)
    loaded_output = output[0, 0]
    torch.testing.assert_close(loaded_output, expected_output)
    print("KERNEL RAN SUCCESSFULLY ...")


if __name__ == "__main__":
    test_triton_paged_attention()
