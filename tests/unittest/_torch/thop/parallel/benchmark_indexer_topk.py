import pytest
from dataclasses import dataclass

import torch

# Import tensorrt_llm to load custom CUDA operators (indexer_topk_decode, indexer_topk_prefill)
import tensorrt_llm  # noqa: F401

# from quack.topk import topk as quack_topk
from flashinfer.testing.utils import bench_gpu_time
import numpy as np
import flashinfer
import os
from enum import Enum

import sys
# sys.path.append("/home/lmin/scratch/dkg-repo/dkg/cutlass_ir/compiler/python/examples/blackwell/sort")
sys.path.append("/home/lmin/scratch/dkg-repo/dkg/cutlass_ir/compiler/python/examples/blackwell/top_k")
sys.path.append("/home/lmin/scratch/dkg-repo/dkg/cutlass_ir/compiler/python/examples")
# from filter_top_k import cute_dsl_topk_wrapper
# from filter_top_k_remove_coarser_low_smem import cute_dsl_topk_wrapper
# TODO:
# from filter_top_k_remove_coarser_low_smem_sort import cute_dsl_topk_wrapper
# from filter_top_k_hybrid import cute_dsl_topk_wrapper
# from filter_top_k_padded import cute_dsl_topk_wrapper
from filter_top_k_dynamic_col import cute_dsl_topk_wrapper

class Distributation (Enum):
    """Distribution of the data."""
    normal = 1
    uniform = 2
    radix_adversal_20bit = 3
    # radix_adversal_8bit = 3
    # radix_adversal_11bit = 4
    # radix_adversal_16bit = 5
    # radix_adversal_22bit = 6

# only for float32
def create_radix_adversal_20bit_logits(num_rows: int, max_len: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    # M = 20
    # # float32 中 1.0 的 bit pattern
    # base_high_uint32 = torch.tensor(0x3f800000, dtype=torch.uint32)  # 0x3f800000 == 1065353216

    # # 创建 mask：低 (32 - M) 位为 1
    # low_mask = (1 << (32 - M)) - 1  # 0xFFF for M=20
    # low_mask = torch.tensor(low_mask, dtype=torch.uint32)

    # # 清除 base 的低 (32-M) 位，保留高 M 位
    # high_part = base_high_uint32 & (~low_mask)

    # # 生成随机低 (32-M) 位
    # low_bits = torch.randint(
    #     0, 
    #     1 << (32 - M), 
    #     size=(num_rows, max_len), 
    #     dtype=torch.int64
    # ).to(torch.uint32) & low_mask  # 确保不越界

    # # 合并
    # combined = high_part | low_bits

    # # 转为 float32，再转目标 dtype/device
    # logits = combined.view(torch.float32).to(dtype=dtype, device="cuda")

    M = 20
    num_low = 32 - M  # 12

    # 使用 1.0 的 bit pattern 作为高 M 位基础
    base_pattern = 0x3f800000  # float32(1.0)

    # 构造掩码：高 M 位保留，低 num_low 位清零
    high_mask = (0xFFFFFFFF << num_low) & 0xFFFFFFFF
    high_part = base_pattern & high_mask

    # 随机低 num_low 位
    low_max = 1 << num_low
    low_bits = torch.randint(0, low_max, (num_rows, max_len), dtype=torch.int64)
    low_bits = (low_bits & (low_max - 1)).to(torch.uint32)

    # 合并
    combined = torch.tensor(high_part, dtype=torch.uint32, device="cpu") | low_bits
    logits = combined.view(torch.float32).to(dtype=dtype, device=device)

    return logits

# only for float32
def check_radix_adversal_20bit_logits(logits: torch.Tensor, M: int = 20) -> bool:
    """Check if the logits are radix adversarial 20 bit."""
    # def get_high_bits(x, M):
    #     u = x.view(torch.uint32)
    #     return (u >> (32 - M)) & ((1 << M) - 1)
    def get_high_bits(logits, M):
        # 将 float32 reinterpret 为 int32（安全且支持位运算）
        u = logits.view(torch.int32)
        shift = 32 - M

        # 模拟逻辑右移（logical right shift）：
        # 因为 int32 >> 是算术右移，负数会补1，所以我们先转成非负的 uint32 等效值
        # 方法：将 u 视为 32 位无符号整数，用 int64 存储以避免符号问题
        u_uint32_equiv = u.to(torch.int64) & 0xFFFFFFFF  # 提升到 int64 并掩码为 32 位无符号

        # 执行逻辑右移
        high_bits = (u_uint32_equiv >> shift) & ((1 << M) - 1)

        # 返回类型可根据需要设为 int32 或 int64；通常 int32 足够
        return high_bits.to(torch.int32)

    high_bits = get_high_bits(logits, M)
    return torch.all(high_bits == high_bits[0, 0])


def create_random_logits(
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    dtype: torch.dtype,
    seed: int,
    distributation: Distributation = Distributation.normal,
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
    
    if distributation == Distributation.normal:
        # Generate random logits in range [0, 1)
        logits = torch.randn(num_rows, max_len, dtype=dtype, device="cuda")
    elif distributation == Distributation.uniform:
        logits = torch.rand(num_rows, max_len, dtype=dtype, device="cuda")
    elif distributation == Distributation.radix_adversal_20bit:
        # 设置高 20 位为：符号=0, 指数=127 (即 2^0 scale), 部分尾数=0
        # 二进制: 0 01111111 xxxxxxxx ...
        logits = create_radix_adversal_20bit_logits(num_rows, max_len, dtype, "cuda")
        if not check_radix_adversal_20bit_logits(logits):
            raise ValueError("Logits are not radix adversarial 20 bit. Please check the create_radix_adversal_20bit_logits function.")
    else:
        raise ValueError(f"Invalid distributation: {distributation}")

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
                print("  Different indices selected:")
                print(f"    Only in CUDA: {cuda_set - torch_set}")
                print(f"    Only in Torch: {torch_set - cuda_set}")

            return False

    return True


def flashinfer_set_topk_algo(algo: str):
    """Fixture to set and reset FLASHINFER_TOPK_ALGO environment variable."""
    original_value = os.environ.get("FLASHINFER_TOPK_ALGO", None)

    if algo == "auto":
        os.environ.pop("FLASHINFER_TOPK_ALGO", None)
    else:
        os.environ["FLASHINFER_TOPK_ALGO"] = algo
    return original_value

def flashinfer_restore_topk_algo(original_value: str):
    # Restore original value
    if original_value is None:
        os.environ.pop("FLASHINFER_TOPK_ALGO", None)
    else:
        os.environ["FLASHINFER_TOPK_ALGO"] = original_value


@dataclass
class BenchConfig:
    """Benchmark configuration."""

    Rows: int  # num_experts
    Cols: int  # num_tokens
    K: int  # top-k
    # dtype: torch.dtype = torch.bfloat16
    dtype: torch.dtype = torch.float32
    name: str = ""

def bench_single_config(config: BenchConfig, dry_run_iters=5, repeat_iters=100, distributation: Distributation = Distributation.normal):
    # cur_seed = 24
    cur_seed = 1111
    torch.manual_seed(cur_seed)
    torch.cuda.manual_seed(cur_seed)

    batch_size = config.Rows
    num_cols = config.Cols
    index_topk = config.K
    dtype = config.dtype

    results = {"config": config}
    print("bench_single_config config: ", config)

    # Set input data
    row_starts = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    row_ends = torch.arange(1, batch_size + 1, device="cuda", dtype=torch.int32)
    torch.fill_(row_ends, num_cols)
    # print("row_starts: ", row_starts)
    # print("row_ends: ", row_ends)

    # logits = create_random_logits(row_starts, row_ends, dtype, cur_seed, distributation)
    logits = torch.randn(batch_size, num_cols, dtype=dtype, device="cuda")
    # print("logits.shape: ", logits.shape)
    # print("logits: ", logits)

    # Create output tensors
    indices = torch.empty((batch_size, index_topk), dtype=torch.int32, device="cuda")
    # print("indices.shape: ", indices.shape)

    # # Run CUDA implementation
    # torch.ops.trtllm.indexer_topk_prefill(logits, row_starts, row_ends, indices, index_topk)
    # # print("indices: ", indices)

    # # Run reference implementation
    # torch_indices = logits.topk(min(index_topk, max(row_ends)), dim=-1)[1]
    # mask_lo = torch_indices >= 0
    # mask_hi = (torch_indices - (row_ends - row_starts)[:, None]) < 0
    # mask = mask_lo & mask_hi
    # torch_indices = torch_indices.masked_fill(~mask, -1)

    # # # Compare results
    # # assert compare_top_k_results(
    # #     logits, indices, torch_indices, row_starts, row_ends, index_topk
    # # ), "CUDA top_k_per_row results don't match torch.topk"

    """
    test_dsl_accuracy = True
    if test_dsl_accuracy and num_cols < 16384:
        dsl_out_indices, dsl_out_values = cute_dsl_topk_wrapper(logits, index_topk)
        original_value = flashinfer_set_topk_algo("filtered")
        fi_filter_out_values, fi_filter_out_indices = flashinfer.top_k(logits, index_topk, sorted=True)
        flashinfer_restore_topk_algo(original_value)

        sorted_dsl_out_values = torch.sort(dsl_out_values.cpu()).values
        sorted_fi_filter_out_values = torch.sort(fi_filter_out_values.cpu()).values

        assert torch.allclose(
               sorted_dsl_out_values,
               sorted_fi_filter_out_values,
               atol=1e-5,
        )
    """
    enable_cupti = True

    # # # # # # # """
    measurements = bench_gpu_time(
            lambda: torch.topk(logits, index_topk, dim=-1),
            enable_cupti=enable_cupti,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
    )
    # print("torch.topk time: ", np.median(measurements) * 1e3)
    results["torch_topk_us"] = np.median(measurements) * 1e3

    measurements = bench_gpu_time(
        lambda: cute_dsl_topk_wrapper(logits, index_topk, return_val=False),
        enable_cupti=enable_cupti,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
    )
    # print("cutlass filter top_k time: ", np.median(measurements) * 1e3)
    results["cute_dsl_filter_top_k_us"] = np.median(measurements) * 1e3
    
    # import numpy as np
    # logits_np = logits.detach().cpu().numpy()
    # # 2. 保存为 .npy 文件（推荐）
    # np.save('1_65536_2048_logits_debug.npy', logits_np)

    # torch.cuda.synchronize()
    # for i in range(1):
    #     dsl_out_indices, dsl_out_values = cute_dsl_topk_wrapper(logits, index_topk, return_values=False)
    # torch.cuda.synchronize()

    # torch.ops.trtllm.indexer_topk_prefill(logits, row_starts, row_ends, indices, index_topk)


    if dtype == torch.float32:
        measurements = bench_gpu_time(
                lambda: torch.ops.trtllm.indexer_topk_prefill(logits, row_starts, row_ends, indices, index_topk),
                enable_cupti=enable_cupti,
                dry_run_iters=dry_run_iters,
                repeat_iters=repeat_iters,
        )
        # print("trtllm indexer_topk_prefill time: ", np.median(measurements) * 1e3)
        results["trtllm_indexer_topk_prefill_us"] = np.median(measurements) * 1e3
 
    # # """
    measurements = bench_gpu_time(
        lambda: flashinfer.top_k(logits, index_topk),
        enable_cupti=enable_cupti,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
    )
    # print("flashinfer top_k time: ", np.median(measurements) * 1e3)
    results["flashinfer_top_k_us"] = np.median(measurements) * 1e3

    original_value = flashinfer_set_topk_algo("multi_cta")
    measurements = bench_gpu_time(
        lambda: flashinfer.top_k(logits, index_topk),
        enable_cupti=enable_cupti,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
    )
    # print("flashinfer multi_cta top_k time: ", np.median(measurements) * 1e3)
    results["flashinfer_multi_cta_top_k_us"] = np.median(measurements) * 1e3
    flashinfer_restore_topk_algo(original_value)


    original_value = flashinfer_set_topk_algo("filtered")
    measurements = bench_gpu_time(
        lambda: flashinfer.top_k(logits, index_topk),
        enable_cupti=enable_cupti,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
    )
    # print("flashinfer filtered top_k time: ", np.median(measurements) * 1e3)
    results["flashinfer_filtered_top_k_us"] = np.median(measurements) * 1e3
    flashinfer_restore_topk_algo(original_value)

    # if config.Cols <= 4096 and config.K <= 128 and config.Cols % 8 == 0 and config.K % 4 == 0:
    #     values, indices = quack_topk(logits, index_topk)
    #     measurements = bench_gpu_time(
    #         lambda: quack_topk(logits, index_topk),
    #         enable_cupti=True,
    #         dry_run_iters=dry_run_iters,
    #         repeat_iters=repeat_iters,
    #     )
    #     # print("quack_topk time: ", np.median(measurements) * 1e3)
    #     results["quack_topk_us"] = np.median(measurements) * 1e3
    # # """

    # # for ncu.
    # torch.ops.trtllm.indexer_topk_prefill(logits, row_starts, row_ends, indices, index_topk)
    # flashinfer.top_k(logits, index_topk)

    # torch.ops.trtllm.indexer_topk_prefill(logits, row_starts, row_ends, indices, index_topk)
    # flashinfer.top_k(logits, index_topk)

    return results


def print_results_table(results_list, title: str):
    """Print results in a formatted table."""
    print("\n" + "=" * 125)
    print(title)
    print("=" * 125)

    # Header
    header = f"{'Config':<25} |{'Torch':>12} | {'TRTLLM':>12} | {'FI':>12} | {'FI Multi-CTA':>12} | {'FI Filtered':>12} | {'Quack':>12} | {'Cute DSL':>12} | {'Best':>12} | {'Torch/Best':>10} | {'FI/TRTLLM':>10} | {'FI/Cute DSL':>10} | {'TRTLLM/DSL':>10}" 
    print(header)
    print("-" * 125)

    for res in results_list:
        cfg = res["config"]
        config_str = f"Rows={cfg.Rows},Cols={cfg.Cols},K={cfg.K}"
        if cfg.name:
            config_str = f"{cfg.name}"

        trtllm_us = res.get("trtllm_indexer_topk_prefill_us", float("nan"))
        fi_us = res.get("flashinfer_top_k_us", float("nan"))
        fi_multi_cta_us = res.get("flashinfer_multi_cta_top_k_us", float("nan"))
        fi_filtered_us = res.get("flashinfer_filtered_top_k_us", float("nan"))
        quack_us = res.get("quack_topk_us", float("nan"))
        torch_us = res.get("torch_topk_us", float("nan"))
        cute_dsl_us = res.get("cute_dsl_filter_top_k_us", float("nan"))

        # Find best
        times = {"Torch": torch_us}
        if not np.isnan(trtllm_us):
            times["TRTLLM"] = trtllm_us
        if not np.isnan(fi_us):
            times["FI"] = fi_us
        if not np.isnan(quack_us):
            times["Quack"] = quack_us
        if not np.isnan(fi_multi_cta_us):
            times["FI Multi-CTA"] = fi_multi_cta_us
        if not np.isnan(fi_filtered_us):
            times["FI Filtered"] = fi_filtered_us
        if not np.isnan(cute_dsl_us):
            times["Cute DSL"] = cute_dsl_us

        best_name = min(times, key=times.get)
        best_time = times[best_name]
        speedup = torch_us / best_time if best_time > 0 else 0
        trtllm_fi_speedup = fi_us / trtllm_us if trtllm_us > 0 else 0
        fi_cute_dsl_speedup = fi_filtered_us / cute_dsl_us if cute_dsl_us > 0 else 0
        trtllm_cute_dsl_speedup = trtllm_us / cute_dsl_us if cute_dsl_us > 0 else 0

        def fmt(val):
            return f"{val:>10.2f}us" if not np.isnan(val) else f"{'N/A':>12}"

        print(
            f"{config_str:<25} | {fmt(torch_us)} | {fmt(trtllm_us)} | {fmt(fi_us)} | {fmt(fi_multi_cta_us)} | {fmt(fi_filtered_us)} | {fmt(quack_us)} | {fmt(cute_dsl_us)} | {best_name:>12} | {speedup:>8.2f}x | {trtllm_fi_speedup:>8.2f}x | {fi_cute_dsl_speedup:>8.2f}x | {trtllm_cute_dsl_speedup:>8.2f}x" 
        )

def run_topk_benchmark(dtype=torch.float32, distributation: Distributation = Distributation.normal):
    """Benchmark DSA scenarios (small N, small K)."""
    configs = [
        # # Standard MoE models
        # BenchConfig(Rows=8, Cols=4096, K=2, name="Mixtral-8x7B", dtype=dtype),
        # BenchConfig(Rows=64, Cols=4096, K=4, name="Qwen-MoE", dtype=dtype),
        # BenchConfig(Rows=160, Cols=4096, K=6, name="DeepSeek-V2", dtype=dtype),
        # BenchConfig(Rows=256, Cols=4096, K=8, name="DeepSeek-V3", dtype=dtype),
        # # Varying batch sizes
        # BenchConfig(Rows=256, Cols=64, K=8, name="256-64-8", dtype=dtype),
        # BenchConfig(Rows=256, Cols=256, K=8, name="256-256-8", dtype=dtype),
        # BenchConfig(Rows=256, Cols=1024, K=8, name="256-1024-8", dtype=dtype),
        # BenchConfig(Rows=256, Cols=4096, K=8, name="256-4096-8", dtype=dtype),
        # BenchConfig(Rows=256, Cols=16384, K=8, name="256-16384-8", dtype=dtype),
        # BenchConfig(Rows=256, Cols=65536, K=8, name="256-65536-8", dtype=dtype),
        # # Varying K
        # BenchConfig(Rows=256, Cols=4096, K=1, name="256-4096-1", dtype=dtype),
        # BenchConfig(Rows=256, Cols=4096, K=2, name="256-4096-2", dtype=dtype),
        # BenchConfig(Rows=256, Cols=4096, K=4, name="256-4096-4", dtype=dtype),
        # BenchConfig(Rows=256, Cols=4096, K=8, name="256-4096-8", dtype=dtype),
        # BenchConfig(Rows=256, Cols=4096, K=16, name="256-4096-16", dtype=dtype),
        # K=2048
        BenchConfig(Rows=1, Cols=4096, K=2048, name="1-4096-2048", dtype=dtype),
        BenchConfig(Rows=1, Cols=8192, K=2048, name="1-8192-2048", dtype=dtype),
        BenchConfig(Rows=1, Cols=16384, K=2048, name="1-16384-2048", dtype=dtype),
        BenchConfig(Rows=1, Cols=32768, K=2048, name="1-32768-2048", dtype=dtype),
        BenchConfig(Rows=1, Cols=65536, K=2048, name="1-65536-2048", dtype=dtype),
        # # Note: trtllm accuracy failed for Cols=131072
        BenchConfig(Rows=1, Cols=131072, K=2048, name="1-131072-2048", dtype=dtype),
        BenchConfig(Rows=1, Cols=262144, K=2048, name="1-262144-2048", dtype=dtype),
        BenchConfig(Rows=16, Cols=4096, K=2048, name="16-4096-2048", dtype=dtype),
        BenchConfig(Rows=16, Cols=8192, K=2048, name="16-8192-2048", dtype=dtype),
        BenchConfig(Rows=16, Cols=16384, K=2048, name="16-16384-2048", dtype=dtype),
        BenchConfig(Rows=16, Cols=32768, K=2048, name="16-32768-2048", dtype=dtype),
        BenchConfig(Rows=16, Cols=65536, K=2048, name="16-65536-2048", dtype=dtype),
        BenchConfig(Rows=16, Cols=131072, K=2048, name="16-131072-2048", dtype=dtype),
        BenchConfig(Rows=16, Cols=262144, K=2048, name="16-262144-2048", dtype=dtype),
        BenchConfig(Rows=128, Cols=4096, K=2048, name="128-4096-2048", dtype=dtype),
        BenchConfig(Rows=128, Cols=8192, K=2048, name="128-8192-2048", dtype=dtype),
        BenchConfig(Rows=128, Cols=16384, K=2048, name="128-16384-2048", dtype=dtype),
        BenchConfig(Rows=128, Cols=32768, K=2048, name="128-32768-2048", dtype=dtype),
        BenchConfig(Rows=128, Cols=65536, K=2048, name="128-65536-2048", dtype=dtype),
        BenchConfig(Rows=128, Cols=131072, K=2048, name="128-131072-2048", dtype=dtype),
        BenchConfig(Rows=128, Cols=262144, K=2048, name="128-262144-2048", dtype=dtype),
        BenchConfig(Rows=256, Cols=4096, K=2048, name="256-4096-2048", dtype=dtype),
        BenchConfig(Rows=256, Cols=8192, K=2048, name="256-8192-2048", dtype=dtype),
        BenchConfig(Rows=256, Cols=16384, K=2048, name="256-16384-2048", dtype=dtype),
        BenchConfig(Rows=256, Cols=32768, K=2048, name="256-32768-2048", dtype=dtype),
        BenchConfig(Rows=256, Cols=65536, K=2048, name="256-65536-2048", dtype=dtype),
        BenchConfig(Rows=256, Cols=131072, K=2048, name="256-131072-2048", dtype=dtype),
        BenchConfig(Rows=256, Cols=262144, K=2048, name="256-262144-2048", dtype=dtype),
        BenchConfig(Rows=512, Cols=4096, K=2048, name="512-4096-2048", dtype=dtype),
        BenchConfig(Rows=512, Cols=8192, K=2048, name="512-8192-2048", dtype=dtype),
        BenchConfig(Rows=512, Cols=16384, K=2048, name="512-16384-2048", dtype=dtype),
        BenchConfig(Rows=512, Cols=32768, K=2048, name="512-32768-2048", dtype=dtype),
        BenchConfig(Rows=512, Cols=65536, K=2048, name="512-65536-2048", dtype=dtype),
        BenchConfig(Rows=512, Cols=131072, K=2048, name="512-131072-2048", dtype=dtype),
        BenchConfig(Rows=512, Cols=262144, K=2048, name="512-262144-2048", dtype=dtype),
        BenchConfig(Rows=1024, Cols=4096, K=2048, name="1024-4096-2048", dtype=dtype),
        BenchConfig(Rows=1024, Cols=8192, K=2048, name="1024-8192-2048", dtype=dtype),
        BenchConfig(Rows=1024, Cols=16384, K=2048, name="1024-16384-2048", dtype=dtype),
        BenchConfig(Rows=1024, Cols=32768, K=2048, name="1024-32768-2048", dtype=dtype),
        BenchConfig(Rows=1024, Cols=65536, K=2048, name="1024-65536-2048", dtype=dtype),
        BenchConfig(Rows=1024, Cols=131072, K=2048, name="1024-131072-2048", dtype=dtype),
        BenchConfig(Rows=1024, Cols=262144, K=2048, name="1024-262144-2048", dtype=dtype),
        BenchConfig(Rows=2048, Cols=4096, K=2048, name="2048-4096-2048", dtype=dtype),
        BenchConfig(Rows=2048, Cols=8192, K=2048, name="2048-8192-2048", dtype=dtype),
        BenchConfig(Rows=2048, Cols=16384, K=2048, name="2048-16384-2048", dtype=dtype),
        BenchConfig(Rows=2048, Cols=32768, K=2048, name="2048-32768-2048", dtype=dtype),
        BenchConfig(Rows=2048, Cols=65536, K=2048, name="2048-65536-2048", dtype=dtype),
        BenchConfig(Rows=2048, Cols=131072, K=2048, name="2048-131072-2048", dtype=dtype),
        BenchConfig(Rows=2048, Cols=262144, K=2048, name="2048-262144-2048", dtype=dtype),
    ]

    results = [bench_single_config(cfg, distributation=distributation) for cfg in configs]
    print_results_table(
        results,
        "Top-K Benchmark (K=2048), dtype=" + str(dtype) + ", distributation=" + str(distributation),
    )


# test_indexer_topk_prefill(256, 128, 4096)

if __name__ == "__main__":
    # trtllm only supports float32
    run_topk_benchmark(dtype=torch.float32, distributation=Distributation.normal)
    # run_topk_benchmark(dtype=torch.float32, distributation=Distributation.uniform)
    # run_topk_benchmark(dtype=torch.float32, distributation=Distributation.radix_adversal_20bit)

    # run_topk_benchmark(dtype=torch.bfloat16, distributation=Distributation.normal)
    # run_topk_benchmark(dtype=torch.float16, distributation=Distributation.normal)
