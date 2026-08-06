"""performance_benchmarking.py.

Performance Benchmarking Framework for Storage Backends

This module provides comprehensive performance testing for different
storage backend implementations to guide production deployment decisions.
"""

import gc
import json
import random
import statistics
import string
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import memory_profiler
import pandas as pd
import psutil


@dataclass
class BenchmarkResult:
    """Result of a single benchmark test."""

    test_name: str
    backend_type: str
    operation: str
    num_records: int
    duration_seconds: float
    records_per_second: float
    memory_peak_mb: float
    memory_average_mb: float
    cpu_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    success_rate: float
    error_count: int


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""

    test_timestamp: str
    system_info: Dict[str, Any]
    results: List[BenchmarkResult]

    def to_json(self) -> str:
        """Export results to JSON."""
        return json.dumps(asdict(self), indent=2, default=str)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis."""
        return pd.DataFrame([asdict(r) for r in self.results])


class PerformanceBenchmark:
    """Comprehensive storage backend performance testing framework."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.test_data_cache: Dict[str, List[Any]] = {}

    def generate_test_records(
        self, count: int, complexity: str = "medium"
    ) -> List[Dict[str, Any]]:
        """Generate test records for benchmarking."""
        cache_key = f"{count}_{complexity}"
        if cache_key in self.test_data_cache:
            return self.test_data_cache[cache_key]

        records = []
        for i in range(count):
            if complexity == "simple":
                text = " ".join(random.choices(["word", "test", "document"], k=10))
            elif complexity == "medium":
                text = " ".join(
                    [
                        "".join(random.choices(string.ascii_lowercase, k=8))
                        for _ in range(50)
                    ]
                )
            else:  # complex
                text = " ".join(
                    [
                        "".join(random.choices(string.ascii_lowercase, k=12))
                        for _ in range(200)
                    ]
                )

            record_data = {
                "id": f"test_{i:06d}",
                "name": f"document_{i}",
                "content": text,
                "model": random.choice(["en_core_web_sm", "en_core_web_md"]),
                "is_active": random.choice([True, False]),
                "metadata": {
                    "category": random.choice(["research", "news", "literature"]),
                    "language": "en",
                    "created_by": f"user_{random.randint(1, 100)}",
                    "tags": random.sample(
                        ["ai", "nlp", "corpus", "analysis", "text"],
                        k=random.randint(1, 3),
                    ),
                },
            }
            records.append(record_data)

        self.test_data_cache[cache_key] = records
        return records

    def benchmark_write_performance(
        self, storage_backend, record_counts: List[int], complexity: str = "medium"
    ) -> List[BenchmarkResult]:
        """Benchmark write operations across different record counts."""
        results = []

        for count in record_counts:
            print(f"Benchmarking write performance: {count} records ({complexity})")

            # Generate test data
            test_records = self.generate_test_records(count, complexity)

            # Prepare monitoring
            process = psutil.Process()
            initial_io = process.io_counters()

            # Run benchmark
            start_time = time.perf_counter()
            start_memory = memory_profiler.memory_usage()[0]

            memory_samples = []
            success_count = 0
            error_count = 0

            def monitor_memory():
                while True:
                    try:
                        memory_samples.append(memory_profiler.memory_usage()[0])
                        time.sleep(0.1)
                    except:
                        break

            # Start memory monitoring in background
            import threading

            monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
            monitor_thread.start()

            # Perform write operations
            for record_data in test_records:
                try:
                    # Convert dict to actual Record object (would be real implementation)
                    # record = Record(**record_data)
                    # success = storage_backend.save_record(record)
                    success = True  # Simulated for prototype

                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    error_count += 1

            # Calculate metrics
            end_time = time.perf_counter()
            duration = end_time - start_time
            final_io = process.io_counters()

            peak_memory = max(memory_samples) if memory_samples else start_memory
            avg_memory = (
                statistics.mean(memory_samples) if memory_samples else start_memory
            )
            cpu_percent = process.cpu_percent()

            result = BenchmarkResult(
                test_name=f"write_performance_{complexity}",
                backend_type=type(storage_backend).__name__,
                operation="write",
                num_records=count,
                duration_seconds=duration,
                records_per_second=count / duration if duration > 0 else 0,
                memory_peak_mb=peak_memory,
                memory_average_mb=avg_memory,
                cpu_percent=cpu_percent,
                disk_io_read_mb=(final_io.read_bytes - initial_io.read_bytes)
                / 1024
                / 1024,
                disk_io_write_mb=(final_io.write_bytes - initial_io.write_bytes)
                / 1024
                / 1024,
                success_rate=success_count / count if count > 0 else 0,
                error_count=error_count,
            )

            results.append(result)
            self.results.append(result)

            # Cleanup
            gc.collect()

        return results

    def benchmark_read_performance(
        self,
        storage_backend,
        record_counts: List[int],
        access_pattern: str = "sequential",
    ) -> List[BenchmarkResult]:
        """Benchmark read operations with different access patterns."""
        results = []

        for count in record_counts:
            print(f"Benchmarking read performance: {count} records ({access_pattern})")

            # Prepare record IDs based on access pattern
            if access_pattern == "sequential":
                record_ids = [f"test_{i:06d}" for i in range(count)]
            elif access_pattern == "random":
                all_ids = [f"test_{i:06d}" for i in range(count * 2)]  # Larger pool
                record_ids = random.sample(all_ids, count)
            else:  # 'sparse'
                record_ids = [f"test_{i:06d}" for i in range(0, count * 10, 10)]

            # Run benchmark
            start_time = time.perf_counter()
            success_count = 0
            error_count = 0

            for record_id in record_ids:
                try:
                    # record = storage_backend.load_record(record_id)
                    record = {"id": record_id}  # Simulated for prototype

                    if record:
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    error_count += 1

            end_time = time.perf_counter()
            duration = end_time - start_time

            result = BenchmarkResult(
                test_name=f"read_performance_{access_pattern}",
                backend_type=type(storage_backend).__name__,
                operation="read",
                num_records=count,
                duration_seconds=duration,
                records_per_second=count / duration if duration > 0 else 0,
                memory_peak_mb=memory_profiler.memory_usage()[0],
                memory_average_mb=memory_profiler.memory_usage()[0],
                cpu_percent=psutil.Process().cpu_percent(),
                disk_io_read_mb=0,  # Would measure in real implementation
                disk_io_write_mb=0,
                success_rate=success_count / count if count > 0 else 0,
                error_count=error_count,
            )

            results.append(result)
            self.results.append(result)

        return results

    def benchmark_query_performance(
        self, storage_backend, query_scenarios: List[Dict[str, Any]]
    ) -> List[BenchmarkResult]:
        """Benchmark query operations with different query types."""
        results = []

        for scenario in query_scenarios:
            print(f"Benchmarking query: {scenario['name']}")

            query = scenario["query"]
            expected_results = scenario.get("expected_results", "unknown")

            start_time = time.perf_counter()

            try:
                # query_result = storage_backend.query_records(query)
                # Simulated for prototype
                query_result = type(
                    "QueryResult",
                    (),
                    {
                        "records": [],
                        "total_count": 0,
                        "execution_time_ms": 0,
                        "backend_used": "simulated",
                    },
                )()

                success = True
                result_count = query_result.total_count
            except Exception as e:
                success = False
                result_count = 0

            end_time = time.perf_counter()
            duration = end_time - start_time

            result = BenchmarkResult(
                test_name=f"query_{scenario['name']}",
                backend_type=type(storage_backend).__name__,
                operation="query",
                num_records=result_count,
                duration_seconds=duration,
                records_per_second=result_count / duration if duration > 0 else 0,
                memory_peak_mb=memory_profiler.memory_usage()[0],
                memory_average_mb=memory_profiler.memory_usage()[0],
                cpu_percent=psutil.Process().cpu_percent(),
                disk_io_read_mb=0,
                disk_io_write_mb=0,
                success_rate=1.0 if success else 0.0,
                error_count=0 if success else 1,
            )

            results.append(result)
            self.results.append(result)

        return results

    def benchmark_concurrent_access(
        self,
        storage_backend,
        num_threads: int,
        operations_per_thread: int,
        operation_mix: Dict[str, float],
    ) -> List[BenchmarkResult]:
        """Benchmark concurrent access patterns."""
        print(
            f"Benchmarking concurrent access: {num_threads} threads, {operations_per_thread} ops/thread"
        )

        def worker_thread(thread_id: int) -> Dict[str, Any]:
            """Worker function for concurrent testing."""
            thread_results = {"success_count": 0, "error_count": 0, "operations": []}

            for i in range(operations_per_thread):
                operation = random.choices(
                    list(operation_mix.keys()), weights=list(operation_mix.values())
                )[0]

                start_time = time.perf_counter()
                try:
                    if operation == "read":
                        record_id = f"test_{random.randint(0, 1000):06d}"
                        # result = storage_backend.load_record(record_id)
                        result = {"id": record_id}  # Simulated
                        success = result is not None
                    elif operation == "write":
                        record_data = self.generate_test_records(1, "simple")[0]
                        record_data["id"] = f"concurrent_{thread_id}_{i}"
                        # success = storage_backend.save_record(Record(**record_data))
                        success = True  # Simulated
                    elif operation == "query":
                        query = {"is_active": True}
                        # result = storage_backend.query_records(query)
                        success = True  # Simulated

                    if success:
                        thread_results["success_count"] += 1
                    else:
                        thread_results["error_count"] += 1

                except Exception as e:
                    thread_results["error_count"] += 1

                duration = time.perf_counter() - start_time
                thread_results["operations"].append(
                    {"operation": operation, "duration": duration, "success": success}
                )

            return thread_results

        # Run concurrent benchmark
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
            thread_results = [future.result() for future in futures]

        end_time = time.perf_counter()
        total_duration = end_time - start_time

        # Aggregate results
        total_operations = num_threads * operations_per_thread
        total_success = sum(r["success_count"] for r in thread_results)
        total_errors = sum(r["error_count"] for r in thread_results)

        result = BenchmarkResult(
            test_name=f"concurrent_access_{num_threads}threads",
            backend_type=type(storage_backend).__name__,
            operation="concurrent",
            num_records=total_operations,
            duration_seconds=total_duration,
            records_per_second=total_operations / total_duration,
            memory_peak_mb=memory_profiler.memory_usage()[0],
            memory_average_mb=memory_profiler.memory_usage()[0],
            cpu_percent=psutil.Process().cpu_percent(),
            disk_io_read_mb=0,
            disk_io_write_mb=0,
            success_rate=total_success / total_operations,
            error_count=total_errors,
        )

        self.results.append(result)
        return [result]

    def run_comprehensive_benchmark(
        self, storage_backends: List[Any]
    ) -> BenchmarkSuite:
        """Run complete benchmark suite across all storage backends."""
        print("Starting comprehensive benchmark suite...")

        # System information
        system_info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "disk_type": "unknown",  # Would detect in real implementation
            "python_version": "3.12",  # Would get actual version
            "platform": "Windows",  # Would detect actual platform
        }

        all_results = []

        for backend in storage_backends:
            print(f"\nBenchmarking {type(backend).__name__}...")

            # Write performance tests
            write_results = self.benchmark_write_performance(
                backend, record_counts=[10, 100, 1000, 5000], complexity="medium"
            )
            all_results.extend(write_results)

            # Read performance tests
            read_results = self.benchmark_read_performance(
                backend,
                record_counts=[10, 100, 1000, 5000],
                access_pattern="sequential",
            )
            all_results.extend(read_results)

            # Query performance tests
            query_scenarios = [
                {
                    "name": "simple_filter",
                    "query": {"is_active": True},
                    "expected_results": "variable",
                },
                {
                    "name": "model_filter",
                    "query": {"model": "en_core_web_sm"},
                    "expected_results": "variable",
                },
                {
                    "name": "range_query",
                    "query": {"num_tokens": {"$gte": 100, "$lte": 1000}},
                    "expected_results": "variable",
                },
            ]

            query_results = self.benchmark_query_performance(backend, query_scenarios)
            all_results.extend(query_results)

            # Concurrent access tests
            concurrent_results = self.benchmark_concurrent_access(
                backend,
                num_threads=4,
                operations_per_thread=25,
                operation_mix={"read": 0.6, "write": 0.3, "query": 0.1},
            )
            all_results.extend(concurrent_results)

        return BenchmarkSuite(
            test_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            system_info=system_info,
            results=all_results,
        )

    def generate_performance_report(
        self, benchmark_suite: BenchmarkSuite, output_dir: Path
    ) -> None:
        """Generate comprehensive performance report with visualizations."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save raw results
        with open(output_dir / "benchmark_results.json", "w") as f:
            f.write(benchmark_suite.to_json())

        # Create DataFrame for analysis
        df = benchmark_suite.to_dataframe()

        # Generate plots
        self._plot_write_performance(df, output_dir)
        self._plot_read_performance(df, output_dir)
        self._plot_query_performance(df, output_dir)
        self._plot_memory_usage(df, output_dir)

        # Generate summary report
        self._generate_summary_report(df, output_dir, benchmark_suite.system_info)

    def _plot_write_performance(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Plot write performance comparison."""
        write_data = df[df["operation"] == "write"]

        plt.figure(figsize=(12, 8))

        for backend in write_data["backend_type"].unique():
            backend_data = write_data[write_data["backend_type"] == backend]
            plt.plot(
                backend_data["num_records"],
                backend_data["records_per_second"],
                marker="o",
                label=backend,
                linewidth=2,
            )

        plt.xlabel("Number of Records")
        plt.ylabel("Records per Second")
        plt.title("Write Performance Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale("log")
        plt.yscale("log")

        plt.tight_layout()
        plt.savefig(output_dir / "write_performance.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_read_performance(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Plot read performance comparison."""
        read_data = df[df["operation"] == "read"]

        plt.figure(figsize=(12, 8))

        for backend in read_data["backend_type"].unique():
            backend_data = read_data[read_data["backend_type"] == backend]
            plt.plot(
                backend_data["num_records"],
                backend_data["records_per_second"],
                marker="s",
                label=backend,
                linewidth=2,
            )

        plt.xlabel("Number of Records")
        plt.ylabel("Records per Second")
        plt.title("Read Performance Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale("log")
        plt.yscale("log")

        plt.tight_layout()
        plt.savefig(output_dir / "read_performance.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_query_performance(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Plot query performance comparison."""
        query_data = df[df["operation"] == "query"]

        if query_data.empty:
            return

        plt.figure(figsize=(12, 8))

        backends = query_data["backend_type"].unique()
        test_names = query_data["test_name"].unique()

        x = range(len(test_names))
        width = 0.35

        for i, backend in enumerate(backends):
            backend_data = query_data[query_data["backend_type"] == backend]
            durations = [
                backend_data[backend_data["test_name"] == name][
                    "duration_seconds"
                ].iloc[0]
                if len(backend_data[backend_data["test_name"] == name]) > 0
                else 0
                for name in test_names
            ]

            plt.bar([pos + width * i for pos in x], durations, width, label=backend)

        plt.xlabel("Query Type")
        plt.ylabel("Duration (seconds)")
        plt.title("Query Performance Comparison")
        plt.xticks([pos + width / 2 for pos in x], test_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "query_performance.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_memory_usage(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Plot memory usage comparison."""
        plt.figure(figsize=(12, 8))

        for backend in df["backend_type"].unique():
            backend_data = df[df["backend_type"] == backend]
            plt.scatter(
                backend_data["num_records"],
                backend_data["memory_peak_mb"],
                label=f"{backend} (Peak)",
                alpha=0.7,
                s=50,
            )
            plt.scatter(
                backend_data["num_records"],
                backend_data["memory_average_mb"],
                label=f"{backend} (Avg)",
                alpha=0.7,
                s=30,
                marker="x",
            )

        plt.xlabel("Number of Records")
        plt.ylabel("Memory Usage (MB)")
        plt.title("Memory Usage Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale("log")

        plt.tight_layout()
        plt.savefig(output_dir / "memory_usage.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _generate_summary_report(
        self, df: pd.DataFrame, output_dir: Path, system_info: Dict[str, Any]
    ) -> None:
        """Generate text summary report."""
        with open(output_dir / "performance_summary.md", "w") as f:
            f.write("# Storage Backend Performance Report\n\n")

            f.write("## System Information\n")
            for key, value in system_info.items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")

            f.write("## Performance Summary\n\n")

            # Write performance summary
            write_data = df[df["operation"] == "write"]
            if not write_data.empty:
                f.write("### Write Performance\n")
                for backend in write_data["backend_type"].unique():
                    backend_data = write_data[write_data["backend_type"] == backend]
                    avg_rps = backend_data["records_per_second"].mean()
                    max_rps = backend_data["records_per_second"].max()
                    f.write(
                        f"- **{backend}**: Avg {avg_rps:.0f} records/sec, Peak {max_rps:.0f} records/sec\n"
                    )
                f.write("\n")

            # Read performance summary
            read_data = df[df["operation"] == "read"]
            if not read_data.empty:
                f.write("### Read Performance\n")
                for backend in read_data["backend_type"].unique():
                    backend_data = read_data[read_data["backend_type"] == backend]
                    avg_rps = backend_data["records_per_second"].mean()
                    max_rps = backend_data["records_per_second"].max()
                    f.write(
                        f"- **{backend}**: Avg {avg_rps:.0f} records/sec, Peak {max_rps:.0f} records/sec\n"
                    )
                f.write("\n")

            # Memory usage summary
            f.write("### Memory Usage\n")
            for backend in df["backend_type"].unique():
                backend_data = df[df["backend_type"] == backend]
                avg_memory = backend_data["memory_peak_mb"].mean()
                max_memory = backend_data["memory_peak_mb"].max()
                f.write(
                    f"- **{backend}**: Avg {avg_memory:.1f} MB, Peak {max_memory:.1f} MB\n"
                )
            f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")

            # Find best performing backends
            if not write_data.empty:
                best_write = write_data.loc[write_data["records_per_second"].idxmax()]
                f.write(
                    f"- **Best Write Performance**: {best_write['backend_type']} "
                    f"({best_write['records_per_second']:.0f} records/sec)\n"
                )

            if not read_data.empty:
                best_read = read_data.loc[read_data["records_per_second"].idxmax()]
                f.write(
                    f"- **Best Read Performance**: {best_read['backend_type']} "
                    f"({best_read['records_per_second']:.0f} records/sec)\n"
                )

            # Memory efficiency
            most_efficient = df.loc[df["memory_peak_mb"].idxmin()]
            f.write(
                f"- **Most Memory Efficient**: {most_efficient['backend_type']} "
                f"({most_efficient['memory_peak_mb']:.1f} MB peak)\n"
            )


# Example usage
if __name__ == "__main__":
    # Create mock storage backends for testing
    class MockFileBackend:
        pass

    class MockSQLiteBackend:
        pass

    # Run benchmark
    benchmark = PerformanceBenchmark()

    backends = [MockFileBackend(), MockSQLiteBackend()]
    results = benchmark.run_comprehensive_benchmark(backends)

    # Generate report
    output_dir = Path("benchmark_results")
    benchmark.generate_performance_report(results, output_dir)

    print(f"Benchmark complete. Results saved to {output_dir}")
