"""benchmark_parallel_loader.py.

Benchmark script comparing Loader vs ParallelLoader performance.

This script creates temporary test files and measures the loading time for both the standard Loader and the ParallelLoader to demonstrate the performance improvement when loading multiple files.

Last Updated: December 27, 2025
Last Tested: December 27, 2025
"""

import shutil
import tempfile
import time
from pathlib import Path

from lexos.io.loader import Loader
from lexos.io.parallel_loader import ParallelLoader


def create_test_files(
    num_files: int, file_size: str = "small"
) -> tuple[Path, list[Path]]:
    """Create temporary test files.

    Args:
        num_files: Number of files to create
        file_size: Size of files - "small", "medium", or "large"

    Returns:
        Tuple of (temp_dir, list of file paths)
    """
    temp_dir = Path(tempfile.mkdtemp())
    test_files = []

    # Determine content size
    if file_size == "small":
        lines = 10
    elif file_size == "medium":
        lines = 100
    elif file_size == "large":
        lines = 1000
    else:
        lines = 10

    content = "\n".join([f"Line {i}: This is some test content." for i in range(lines)])

    for i in range(num_files):
        test_file = temp_dir / f"test_file_{i:04d}.txt"
        test_file.write_text(f"File {i}\n{content}")
        test_files.append(test_file)

    return temp_dir, test_files


def benchmark_loader(loader_class, test_files, **kwargs):
    """Benchmark a loader class.

    Args:
        loader_class: The loader class to benchmark
        test_files: List of files to load
        **kwargs: Additional arguments for the loader

    Returns:
        Tuple of (elapsed_time, loader_instance)
    """
    start = time.time()
    loader = loader_class(**kwargs)
    loader.load(test_files)
    elapsed = time.time() - start
    return elapsed, loader


def run_benchmark(num_files: int = 50, file_size: str = "small"):
    """Run benchmark comparing Loader and ParallelLoader.

    Args:
        num_files: Number of files to create and load
        file_size: Size of files - "small", "medium", or "large"
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {num_files} {file_size} files")
    print(f"{'=' * 60}\n")

    # Create test files
    print(f"Creating {num_files} test files...")
    temp_dir, test_files = create_test_files(num_files, file_size)
    print(f"Created files in {temp_dir}\n")

    # Benchmark standard Loader
    print("Testing standard Loader...")
    time1, loader1 = benchmark_loader(Loader, test_files)
    print(f"  Time: {time1:.3f}s")
    print(f"  Loaded: {len(loader1.texts)} texts")
    print(f"  Errors: {len(loader1.errors)}\n")

    # Benchmark ParallelLoader with progress bar
    print("Testing ParallelLoader (with progress bar)...")
    time2, loader2 = benchmark_loader(
        ParallelLoader, test_files, show_progress=True, max_workers=8
    )
    print(f"  Time: {time2:.3f}s")
    print(f"  Loaded: {len(loader2.texts)} texts")
    print(f"  Errors: {len(loader2.errors)}\n")

    # Benchmark ParallelLoader without progress bar (for pure performance)
    print("Testing ParallelLoader (no progress bar)...")
    time3, loader3 = benchmark_loader(
        ParallelLoader, test_files, show_progress=False, max_workers=8
    )
    print(f"  Time: {time3:.3f}s")
    print(f"  Loaded: {len(loader3.texts)} texts")
    print(f"  Errors: {len(loader3.errors)}\n")

    # Calculate speedup
    speedup = time1 / time3
    print(f"Results:")
    print(f"  Speedup (vs standard Loader): {speedup:.2f}x")
    print(f"  Parallel overhead (progress bar): {(time2 - time3):.3f}s\n")

    # Cleanup
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary files\n")


if __name__ == "__main__":
    # Run benchmarks with different file counts
    run_benchmark(num_files=20, file_size="small")
    run_benchmark(num_files=50, file_size="small")
    run_benchmark(num_files=100, file_size="small")

    print(f"\n{'=' * 60}")
    print("Benchmark complete!")
    print(f"{'=' * 60}\n")
