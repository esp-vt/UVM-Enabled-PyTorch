"""
Benchmark comparing UVM (cudaMallocManaged) vs Pinned Memory (cudaHostAlloc)

This script compares the performance of CUDA Unified Virtual Memory (UVM)
against traditional pinned memory for tensor allocations in PyTorch.

Usage:
    # Run comparison (launches both modes in subprocesses)
    python benchmarks/memory/uvm_pinned_comparison.py

    # Run single mode directly
    python benchmarks/memory/uvm_pinned_comparison.py --mode pinned
    python benchmarks/memory/uvm_pinned_comparison.py --mode uvm

Environment variable to switch modes:
    PYTORCH_CUDA_ALLOC_CONF="pinned_use_cuda_malloc_managed:True"   # UVM mode
    PYTORCH_CUDA_ALLOC_CONF="pinned_use_cuda_malloc_managed:False"  # Pinned mode (default)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any


def check_cuda_available():
    """Check if CUDA is available."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("CUDA is not available. This benchmark requires a CUDA-enabled GPU.")
            sys.exit(1)
        return True
    except ImportError:
        print("PyTorch is not installed.")
        sys.exit(1)


def benchmark_allocation(sizes: list[int], iterations: int = 50) -> dict[int, dict[str, float]]:
    """Benchmark pinned memory allocation times."""
    import torch

    results = {}
    for size in sizes:
        times = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            t = torch.empty(size, pin_memory=True)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
            del t
            # Clear the allocator cache to force new allocations
            torch._C._host_emptyCache()

        results[size] = {
            'mean_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5
        }
    return results


def benchmark_transfer_to_gpu(sizes: list[int], iterations: int = 50) -> dict[int, dict[str, float]]:
    """Benchmark CPU->GPU transfer times with pinned memory."""
    import torch

    results = {}
    for size in sizes:
        times = []
        for _ in range(iterations):
            cpu_tensor = torch.randn(size, pin_memory=True)
            torch.cuda.synchronize()

            start = time.perf_counter()
            gpu_tensor = cpu_tensor.cuda(non_blocking=True)
            torch.cuda.synchronize()
            end = time.perf_counter()

            times.append((end - start) * 1000)  # Convert to ms
            del cpu_tensor, gpu_tensor

        results[size] = {
            'mean_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'bandwidth_gbps': (size * 4 / 1e9) / (sum(times) / len(times) / 1000)  # GB/s for float32
        }
    return results


def benchmark_transfer_to_cpu(sizes: list[int], iterations: int = 50) -> dict[int, dict[str, float]]:
    """Benchmark GPU->CPU transfer times to pinned memory."""
    import torch

    results = {}
    for size in sizes:
        times = []
        for _ in range(iterations):
            gpu_tensor = torch.randn(size, device='cuda')
            cpu_tensor = torch.empty(size, pin_memory=True)
            torch.cuda.synchronize()

            start = time.perf_counter()
            cpu_tensor.copy_(gpu_tensor, non_blocking=True)
            torch.cuda.synchronize()
            end = time.perf_counter()

            times.append((end - start) * 1000)  # Convert to ms
            del cpu_tensor, gpu_tensor

        results[size] = {
            'mean_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'bandwidth_gbps': (size * 4 / 1e9) / (sum(times) / len(times) / 1000)  # GB/s for float32
        }
    return results


def benchmark_dataloader(batch_size: int = 64, num_batches: int = 50) -> dict[str, float]:
    """Benchmark DataLoader with pin_memory."""
    import torch
    import torch.utils.data

    # Create dummy dataset (simulating image data)
    dataset = torch.utils.data.TensorDataset(
        torch.randn(500, 3, 224, 224),
        torch.randint(0, 1000, (500,))
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2,
        shuffle=True
    )

    times = []
    for i, (data, target) in enumerate(loader):
        if i >= num_batches:
            break
        torch.cuda.synchronize()
        start = time.perf_counter()
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'samples_per_sec': batch_size / (sum(times) / len(times) / 1000)
    }


def get_host_memory_stats() -> dict[str, Any]:
    """Get host allocator statistics."""
    import torch
    stats = torch.cuda.memory.host_memory_stats()
    return {
        'alloc_time_total_ms': stats.get('host_alloc_time.total', 0) / 1000,
        'alloc_time_count': stats.get('host_alloc_time.count', 0),
        'free_time_total_ms': stats.get('host_free_time.total', 0) / 1000,
        'free_time_count': stats.get('host_free_time.count', 0),
        'allocated_bytes_current': stats.get('allocated_bytes.current', 0),
        'allocated_bytes_peak': stats.get('allocated_bytes.peak', 0),
    }


def run_benchmarks(mode_name: str) -> dict[str, Any]:
    """Run all benchmarks and collect results."""
    import torch

    print(f"\n{'='*60}")
    print(f"Running benchmarks in {mode_name} mode")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    # Test sizes: 1KB, 1MB, 10MB, 100MB (in float32 elements)
    sizes = [256, 256*1024, 256*1024*10, 256*1024*100]
    size_names = ['1KB', '1MB', '10MB', '100MB']

    print("1. Allocation benchmark...")
    alloc_results = benchmark_allocation(sizes, iterations=30)

    print("2. CPU->GPU transfer benchmark...")
    to_gpu_results = benchmark_transfer_to_gpu(sizes, iterations=30)

    print("3. GPU->CPU transfer benchmark...")
    to_cpu_results = benchmark_transfer_to_cpu(sizes, iterations=30)

    print("4. DataLoader benchmark...")
    dataloader_results = benchmark_dataloader(batch_size=32, num_batches=30)

    print("5. Collecting host memory stats...")
    host_stats = get_host_memory_stats()

    results = {
        'mode': mode_name,
        'device': torch.cuda.get_device_name(0),
        'sizes': dict(zip(size_names, sizes)),
        'allocation': {name: alloc_results[size] for name, size in zip(size_names, sizes)},
        'transfer_to_gpu': {name: to_gpu_results[size] for name, size in zip(size_names, sizes)},
        'transfer_to_cpu': {name: to_cpu_results[size] for name, size in zip(size_names, sizes)},
        'dataloader': dataloader_results,
        'host_stats': host_stats
    }

    return results


def print_results(results: dict[str, Any]):
    """Print benchmark results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Results for {results['mode']} mode")
    print(f"{'='*60}")

    print("\nAllocation Times (ms):")
    print(f"{'Size':<10} {'Mean':>10} {'Min':>10} {'Max':>10} {'Std':>10}")
    print("-" * 50)
    for size_name, data in results['allocation'].items():
        print(f"{size_name:<10} {data['mean_ms']:>10.3f} {data['min_ms']:>10.3f} "
              f"{data['max_ms']:>10.3f} {data['std_ms']:>10.3f}")

    print("\nCPU->GPU Transfer (ms / GB/s):")
    print(f"{'Size':<10} {'Mean':>10} {'Bandwidth':>12}")
    print("-" * 35)
    for size_name, data in results['transfer_to_gpu'].items():
        print(f"{size_name:<10} {data['mean_ms']:>10.3f} {data['bandwidth_gbps']:>10.2f} GB/s")

    print("\nGPU->CPU Transfer (ms / GB/s):")
    print(f"{'Size':<10} {'Mean':>10} {'Bandwidth':>12}")
    print("-" * 35)
    for size_name, data in results['transfer_to_cpu'].items():
        print(f"{size_name:<10} {data['mean_ms']:>10.3f} {data['bandwidth_gbps']:>10.2f} GB/s")

    print("\nDataLoader Performance:")
    dl = results['dataloader']
    print(f"  Mean transfer time: {dl['mean_ms']:.3f} ms")
    print(f"  Throughput: {dl['samples_per_sec']:.1f} samples/sec")

    print("\nHost Memory Stats:")
    hs = results['host_stats']
    print(f"  Total alloc time: {hs['alloc_time_total_ms']:.3f} ms ({hs['alloc_time_count']} calls)")
    print(f"  Total free time: {hs['free_time_total_ms']:.3f} ms ({hs['free_time_count']} calls)")
    print(f"  Peak allocated: {hs['allocated_bytes_peak'] / 1024 / 1024:.2f} MB")


def print_comparison(pinned_results: dict, uvm_results: dict):
    """Print comparison between pinned and UVM results."""
    print(f"\n{'='*70}")
    print("COMPARISON: Pinned Memory vs UVM (cudaMallocManaged)")
    print(f"{'='*70}")

    print("\nAllocation Time Comparison (lower is better):")
    print(f"{'Size':<10} {'Pinned (ms)':>12} {'UVM (ms)':>12} {'Speedup':>10}")
    print("-" * 50)
    for size_name in pinned_results['allocation']:
        pinned_time = pinned_results['allocation'][size_name]['mean_ms']
        uvm_time = uvm_results['allocation'][size_name]['mean_ms']
        speedup = pinned_time / uvm_time if uvm_time > 0 else float('inf')
        winner = "UVM" if speedup > 1 else "Pinned"
        print(f"{size_name:<10} {pinned_time:>12.3f} {uvm_time:>12.3f} {speedup:>8.2f}x ({winner})")

    print("\nCPU->GPU Transfer Comparison:")
    print(f"{'Size':<10} {'Pinned (GB/s)':>14} {'UVM (GB/s)':>12} {'Speedup':>10}")
    print("-" * 50)
    for size_name in pinned_results['transfer_to_gpu']:
        pinned_bw = pinned_results['transfer_to_gpu'][size_name]['bandwidth_gbps']
        uvm_bw = uvm_results['transfer_to_gpu'][size_name]['bandwidth_gbps']
        speedup = uvm_bw / pinned_bw if pinned_bw > 0 else float('inf')
        winner = "UVM" if speedup > 1 else "Pinned"
        print(f"{size_name:<10} {pinned_bw:>14.2f} {uvm_bw:>12.2f} {speedup:>8.2f}x ({winner})")

    print("\nGPU->CPU Transfer Comparison:")
    print(f"{'Size':<10} {'Pinned (GB/s)':>14} {'UVM (GB/s)':>12} {'Speedup':>10}")
    print("-" * 50)
    for size_name in pinned_results['transfer_to_cpu']:
        pinned_bw = pinned_results['transfer_to_cpu'][size_name]['bandwidth_gbps']
        uvm_bw = uvm_results['transfer_to_cpu'][size_name]['bandwidth_gbps']
        speedup = uvm_bw / pinned_bw if pinned_bw > 0 else float('inf')
        winner = "UVM" if speedup > 1 else "Pinned"
        print(f"{size_name:<10} {pinned_bw:>14.2f} {uvm_bw:>12.2f} {speedup:>8.2f}x ({winner})")

    print("\nDataLoader Comparison:")
    pinned_throughput = pinned_results['dataloader']['samples_per_sec']
    uvm_throughput = uvm_results['dataloader']['samples_per_sec']
    speedup = uvm_throughput / pinned_throughput if pinned_throughput > 0 else float('inf')
    winner = "UVM" if speedup > 1 else "Pinned"
    print(f"  Pinned: {pinned_throughput:.1f} samples/sec")
    print(f"  UVM:    {uvm_throughput:.1f} samples/sec")
    print(f"  Speedup: {speedup:.2f}x ({winner})")


def run_single_mode(mode: str):
    """Run benchmarks in a single mode and output JSON results."""
    check_cuda_available()
    results = run_benchmarks(mode.upper())
    print_results(results)
    # Output JSON for parsing by parent process
    print("\n---JSON_RESULTS_START---")
    print(json.dumps(results))
    print("---JSON_RESULTS_END---")


def run_comparison():
    """Run both modes and compare results."""
    check_cuda_available()

    print("Running UVM vs Pinned Memory Comparison Benchmark")
    print("=" * 60)

    # Run pinned memory mode
    print("\nLaunching Pinned Memory benchmark...")
    env_pinned = os.environ.copy()
    env_pinned["PYTORCH_CUDA_ALLOC_CONF"] = "pinned_use_cuda_malloc_managed:False"
    result_pinned = subprocess.run(
        [sys.executable, __file__, "--mode", "pinned"],
        capture_output=True,
        text=True,
        env=env_pinned
    )

    if result_pinned.returncode != 0:
        print(f"Error running pinned benchmark:\n{result_pinned.stderr}")
        sys.exit(1)

    # Parse pinned results
    pinned_output = result_pinned.stdout
    print(pinned_output.split("---JSON_RESULTS_START---")[0])
    pinned_json = pinned_output.split("---JSON_RESULTS_START---")[1].split("---JSON_RESULTS_END---")[0].strip()
    pinned_results = json.loads(pinned_json)

    # Run UVM mode
    print("\nLaunching UVM benchmark...")
    env_uvm = os.environ.copy()
    env_uvm["PYTORCH_CUDA_ALLOC_CONF"] = "pinned_use_cuda_malloc_managed:True"
    result_uvm = subprocess.run(
        [sys.executable, __file__, "--mode", "uvm"],
        capture_output=True,
        text=True,
        env=env_uvm
    )

    if result_uvm.returncode != 0:
        print(f"Error running UVM benchmark:\n{result_uvm.stderr}")
        sys.exit(1)

    # Parse UVM results
    uvm_output = result_uvm.stdout
    print(uvm_output.split("---JSON_RESULTS_START---")[0])
    uvm_json = uvm_output.split("---JSON_RESULTS_START---")[1].split("---JSON_RESULTS_END---")[0].strip()
    uvm_results = json.loads(uvm_json)

    # Print comparison
    print_comparison(pinned_results, uvm_results)


def main():
    parser = argparse.ArgumentParser(description="UVM vs Pinned Memory Benchmark")
    parser.add_argument(
        "--mode",
        choices=["pinned", "uvm", "compare"],
        default="compare",
        help="Benchmark mode: 'pinned' for cudaHostAlloc, 'uvm' for cudaMallocManaged, 'compare' for both"
    )
    args = parser.parse_args()

    if args.mode == "compare":
        run_comparison()
    else:
        run_single_mode(args.mode)


if __name__ == "__main__":
    main()
