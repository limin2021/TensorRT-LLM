import pytest
from dataclasses import dataclass

import torch

# Import tensorrt_llm to load custom CUDA operators (indexer_topk_decode, indexer_topk_prefill)
import tensorrt_llm  # noqa: F401


import sys
sys.path.append("/home/lmin/scratch/dkg-repo/dkg/cutlass_ir/compiler/python/examples/blackwell/top_k/varlen")
sys.path.append("/home/lmin/scratch/dkg-repo/dkg/cutlass_ir/compiler/python/examples")
# from filter_top_k import cute_dsl_topk_wrapper
# from filter_top_k_padded import cute_dsl_topk_wrapper
from filter_top_k_decode_varlen import cute_dsl_topk_wrapper

# torch.set_printoptions(threshold=float('inf'))

def create_random_logits(
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    dtype: torch.dtype,
    seed: int,
    dist: str = "uniform",
) -> torch.Tensor:
    """Create random logits tensor for testing.

    Args:
        row_starts: Tensor of shape (num_rows,) indicating the start position of each row
        row_ends: Tensor of shape (num_rows,) indicating the end position (exclusive) of each row
        dtype: Data type for the logits tensor
        seed: Random seed for reproducibility

    Returns:
        Tensor of shape (num_rows, max_row_length) with random values and -inf padding
    """
    torch.manual_seed(seed)
    num_rows = row_starts.shape[0]
    max_len = int(row_ends.max().item())

    # Generate random logits in range [0, 1)
    print("dist: ", dist)
    if dist == "uniform":
        logits = torch.rand(num_rows, max_len, dtype=dtype, device="cuda")
    elif dist == "normal":
        logits = torch.randn(num_rows, max_len, dtype=dtype, device="cuda")
    else:
        raise ValueError(f"Invalid distribution: {dist}")

    # Vectorized masking: set positions outside [row_start, row_end) to -inf
    col_indices = torch.arange(max_len, device="cuda").unsqueeze(0)  # (1, max_len)
    mask_lo = col_indices < row_starts.unsqueeze(1)  # positions before row_start
    mask_hi = col_indices >= row_ends.unsqueeze(1)  # positions at or after row_end
    mask = mask_lo | mask_hi  # positions outside valid range
    logits[mask] = float("-inf")

    return logits


def compare_top_k_results(
    logits: torch.Tensor,
    cuda_indices: torch.Tensor,
    torch_indices: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    top_k: int,
    tolerance: float = 1e-5,
) -> bool:
    """
    Compare results from CUDA top_k_per_row with torch.topk.
    Handles different shapes and -1 placeholders in cuda_indices.

    Args:
        logits: Input logits tensor [num_rows, vocab_size]
        cuda_indices: CUDA implementation output [num_rows, cuda_k], may contain -1
        torch_indices: PyTorch reference output [num_rows, torch_k], may contain -1
        row_starts: Start positions for each row [num_rows]
        row_ends: End positions for each row [num_rows]
        top_k: Target top-k value
        tolerance: Tolerance for floating point comparison

    Returns:
        True if results match within tolerance, False otherwise
    """
    num_rows = cuda_indices.shape[0]

    # Handle potentially different k values
    cuda_indices.shape[1]
    torch_indices.shape[1]

    # Calculate valid lengths for each row (vectorized)
    row_lengths = row_ends - row_starts

    # For each row, compare only the valid indices (non -1)
    for row_idx in range(num_rows):
        row_len = row_lengths[row_idx].item()
        expected_valid = min(row_len, top_k)

        # Get valid indices from both implementations (filter out -1)
        cuda_row = cuda_indices[row_idx]
        torch_row = torch_indices[row_idx]

        # Filter out -1 (invalid) indices
        cuda_valid_mask = cuda_row != -1
        torch_valid_mask = torch_row != -1

        cuda_valid = cuda_row[cuda_valid_mask]
        torch_valid = torch_row[torch_valid_mask]

        # Check if the number of valid indices matches
        if cuda_valid.shape[0] != torch_valid.shape[0]:
            print(
                f"Row {row_idx}: Different number of valid indices - "
                f"CUDA: {cuda_valid.shape[0]}, PyTorch: {torch_valid.shape[0]}"
            )
            return False

        if cuda_valid.shape[0] != expected_valid:
            print(
                f"Row {row_idx}: Expected {expected_valid} valid indices, got {cuda_valid.shape[0]}"
            )
            return False

        # If no valid indices, continue
        if cuda_valid.shape[0] == 0:
            continue

        # Gather the corresponding logit values
        row_start = row_starts[row_idx].item()
        logits_row = logits[row_idx]

        # Adjust indices to absolute positions (add row_start offset)
        cuda_abs_indices = cuda_valid + row_start
        torch_abs_indices = torch_valid + row_start

        # Get logit values for the selected indices
        cuda_values = logits_row[cuda_abs_indices]
        torch_values = logits_row[torch_abs_indices]

        # Sort both value arrays in descending order
        cuda_values_sorted, _ = torch.sort(cuda_values, descending=True)
        torch_values_sorted, _ = torch.sort(torch_values, descending=True)

        # Compare sorted values
        if not torch.allclose(
            cuda_values_sorted, torch_values_sorted, rtol=tolerance, atol=tolerance
        ):
            # Additional debug: check if sets are identical
            cuda_set = set(cuda_valid.cpu().tolist())
            torch_set = set(torch_valid.cpu().tolist())
            if cuda_set != torch_set:
                print("  row_idx: ", row_idx)
                print("  cuda_values_sorted: ", cuda_values_sorted)
                print("  torch_values_sorted: ", torch_values_sorted)
                print("  Different indices selected:")
                print(f"    Only in CUDA: {cuda_set - torch_set}")
                print(f"    Only in Torch: {torch_set - cuda_set}")

            return False

    return True


def generate_seq_lens(batch_size, min_long_seq, num_tokens):
    seq_lens = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    is_long = torch.rand(batch_size, device="cuda") < 0.9
    num_long = is_long.sum().item()
    if num_long > 0:
        seq_lens[is_long] = torch.randint(
            min_long_seq, num_tokens, (num_long,), dtype=torch.int32, device="cuda"
        )

    num_short = (~is_long).sum().item()
    if num_short > 0:
        seq_lens[~is_long] = torch.randint(
            1, min_long_seq, (num_short,), dtype=torch.int32, device="cuda"
        )
    return seq_lens


@pytest.mark.parametrize("batch_size", [1, 64, 512, 2048])
@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("index_topk", [2048, 128])
@pytest.mark.parametrize("num_tokens", [4096, 8192])
def test_indexer_topk_decode(batch_size, next_n, index_topk, num_tokens, dist="uniform"):
    # torch.manual_seed(24)
    # torch.cuda.manual_seed(24)
    torch.manual_seed(1111)
    torch.cuda.manual_seed(1111)

    print("batch_size: ", batch_size)
    print("next_n: ", next_n)
    print("index_topk: ", index_topk)
    print("num_tokens: ", num_tokens)

    # Set input data
    num_gen_tokens = batch_size * next_n  # Use the same variable name as dsa.py
    row_starts = torch.zeros(num_gen_tokens, dtype=torch.int32, device="cuda")
    row_indices = torch.arange(num_gen_tokens, device="cuda") // next_n
    next_n_offset = torch.arange(num_gen_tokens, device="cuda") % next_n
    print("row_starts: ", row_starts)
    print("row_indices: ", row_indices)
    print("next_n_offset: ", next_n_offset)

    seq_lens = generate_seq_lens(batch_size, index_topk, num_tokens)
    print("seq_lens: ", seq_lens)
    row_ends = seq_lens[row_indices] - next_n + next_n_offset + 1
    print("row_ends: ", row_ends)

    logits = create_random_logits(row_starts, row_ends, torch.float32, 42, dist=dist)
    print("logits.shape: ", logits.shape)
    print("logits: ", logits)

    # Create output tensors
    indices = torch.empty((num_gen_tokens, index_topk), dtype=torch.int32, device="cuda")

    # # Run CUDA implementation
    torch.ops.trtllm.indexer_topk_decode(logits, seq_lens, indices, next_n, index_topk)
    torch.cuda.synchronize()

    # Run reference implementation
    max_row_len = row_ends.max().item()
    torch_indices = logits.topk(min(index_topk, max_row_len), dim=-1)[1]
    mask_lo = torch_indices >= 0
    mask_hi = (torch_indices - (row_ends - row_starts)[:, None]) < 0
    mask = mask_lo & mask_hi
    torch_indices = torch_indices.masked_fill(~mask, -1)

    # # Compare results
    # assert compare_top_k_results(
    #     logits, indices, torch_indices, row_starts, row_ends, index_topk
    # ), "CUDA top_k_per_row results don't match torch.topk"
    # print("PASSED")

    cute_dsl_out_indices, cute_dsl_values = cute_dsl_topk_wrapper(logits, seq_lens, index_topk, next_n, return_val=False)
    torch.cuda.synchronize()
    # print("cute_dsl_values: ", cute_dsl_values)
    # print("cute_dsl_out_indices: ", cute_dsl_out_indices)

    assert compare_top_k_results(
        logits, cute_dsl_out_indices, torch_indices, row_starts, row_ends, index_topk
    ), "CUDA top_k_per_row results don't match cute_dsl_topk_wrapper"
    print("PASSED")

    #test perf
    for i in range(5):
        torch.ops.trtllm.indexer_topk_decode(logits, seq_lens, indices, next_n, index_topk)
    for i in range(10):
        torch.ops.trtllm.indexer_topk_decode(logits, seq_lens, indices, next_n, index_topk)
    
    for i in range(5):
        cute_dsl_out_indices, cute_dsl_values = cute_dsl_topk_wrapper(logits, seq_lens, index_topk, next_n, return_val=False)
    for i in range(10):
        cute_dsl_out_indices, cute_dsl_values = cute_dsl_topk_wrapper(logits, seq_lens, index_topk, next_n, return_val=False)

    print("Finished perf test")


@pytest.mark.parametrize("batch_size", [1, 512, 2048])
@pytest.mark.parametrize("index_topk", [2048, 128])
@pytest.mark.parametrize("num_tokens", [4096, 8192])
def test_indexer_topk_prefill(batch_size, num_cols, index_topk, dry_run_iters=10, repeat_iters=100):
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)

    print("batch_size: ", batch_size)
    print("num_cols: ", num_cols)
    print("index_topk: ", index_topk)

    # Set input data
    row_starts = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    row_ends = torch.arange(1, batch_size + 1, device="cuda", dtype=torch.int32)
    print("row_starts: ", row_starts)
    print("row_ends: ", row_ends)
    # torch.fill_(row_ends, num_cols)
    # print("after row_starts: ", row_starts)
    # print("after row_ends: ", row_ends)

    logits = create_random_logits(row_starts, row_ends, torch.float32, 42)
    print("logits.shape: ", logits.shape)
    print("logits: ", logits)

    # Create output tensors
    indices = torch.empty((batch_size, index_topk), dtype=torch.int32, device="cuda")
    # print("indices.shape: ", indices.shape)

    # Run CUDA implementation
    torch.ops.trtllm.indexer_topk_prefill(logits, row_starts, row_ends, indices, index_topk)
    # print("indices: ", indices)

    # Run reference implementation
    torch_indices = logits.topk(min(index_topk, max(row_ends)), dim=-1)[1]
    mask_lo = torch_indices >= 0
    mask_hi = (torch_indices - (row_ends - row_starts)[:, None]) < 0
    mask = mask_lo & mask_hi
    torch_indices = torch_indices.masked_fill(~mask, -1)

    # Compare results
    assert compare_top_k_results(
        logits, indices, torch_indices, row_starts, row_ends, index_topk
    ), "CUDA top_k_per_row results don't match torch.topk"

    # from flashinfer.testing.utils import bench_gpu_time
    # import numpy as np
    # import flashinfer
    # measurements = bench_gpu_time(
    #         lambda: torch.ops.trtllm.indexer_topk_prefill(logits, row_starts, row_ends, indices, index_topk),
    #         enable_cupti=True,
    #         dry_run_iters=dry_run_iters,
    #         repeat_iters=repeat_iters,
    # )
    # print("trtllm indexer_topk_prefill time: ", np.median(measurements) * 1e3)

    # values, indices = flashinfer.top_k(logits, index_topk)

    # measurements = bench_gpu_time(
    #     lambda: flashinfer.top_k(logits, index_topk),
    #     enable_cupti=True,
    #     dry_run_iters=dry_run_iters,
    #     repeat_iters=repeat_iters,
    # )
    # print("flashinfer top_k time: ", np.median(measurements) * 1e3)

    print("PASSED")


# test_indexer_topk_prefill(256, 128, 4096)
# test_indexer_topk_decode(256, 1, 128, 4096)
# test_indexer_topk_decode(4, 1, 6, 10)
# print("--------------------------------")
# test_indexer_topk_decode(4, 2, 6, 10)
# print("--------------------------------")
# test_indexer_topk_decode(4, 3, 6, 10)

# if __name__ == "__main__":
#     batch_size = 256
#     num_cols = 4096
#     index_topk = 128
#     for batch_size in [1, 16, 256, 1024]:
#         for num_cols in [1024, 2048, 4096, 8192, 16384, 32768, 65536]:
#             for index_topk in [128, 256, 512, 1024]:
#                 if index_topk < num_cols:
#                     print(f"Testing batch_size: {batch_size}, num_cols: {num_cols}, index_topk: {index_topk}")
#                     test_indexer_topk_prefill(batch_size, num_cols, index_topk)

top_k = 2048

# dist = "normal"
# #pass
# for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
#     for next_n in [1, 2, 3]:
#         for num_tokens in [4096, 8192, 16384, 32768, 65536]:
#         # for num_tokens in [65536]:
#             test_indexer_topk_decode(batch_size, next_n, top_k, num_tokens, dist=dist)
#             print("--------------------------------")

# # pass
# dist = "uniform"
# for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
#     for next_n in [1, 2, 3]:
#         for num_tokens in [4096, 8192, 16384, 32768, 65536]:
#             test_indexer_topk_decode(batch_size, next_n, top_k, num_tokens, dist=dist)
#             print("--------------------------------")

dist="normal"
top_k=2048
num_tokens=65536
test_indexer_topk_decode(128, 3, top_k, num_tokens, dist=dist)
test_indexer_topk_decode(512, 1, top_k, num_tokens, dist=dist)
test_indexer_topk_decode(250, 2, top_k, num_tokens, dist=dist)
test_indexer_topk_decode(256, 3, top_k, num_tokens, dist=dist)
