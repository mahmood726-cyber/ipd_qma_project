"""
IPD-QMA Performance Benchmarks

This module provides comprehensive performance benchmarks for the IPD-QMA package.
It measures execution time across various configurations and dataset sizes.

Usage:
    python benchmarks/benchmark_ipd_qma.py

The benchmarks will:
1. Test various dataset sizes (small, medium, large)
2. Compare parallel vs sequential processing
3. Measure bootstrap scalability
4. Track memory usage (optional)
5. Generate performance reports

Performance Targets (from plan):
- 10 studies, 1000 bootstrap: < 30 seconds
- 50 studies, 500 bootstrap: < 2 minutes
- Memory usage: < 500MB for 100 studies
"""

import time
import gc
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Callable
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipd_qma import IPDQMA, IQMAConfig

# Optional memory profiling
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.elapsed_time = None
        self.memory_mb = None
        self.success = False
        self.error = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'config': self.config,
            'elapsed_time': self.elapsed_time,
            'memory_mb': self.memory_mb,
            'success': self.success,
            'error': self.error
        }


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for IPD-QMA.

    Tests performance across various scenarios:
    - Different numbers of studies
    - Different bootstrap sample sizes
    - Different sample sizes per study
    - Parallel vs sequential processing
    """

    def __init__(self, output_dir: str = "benchmark_results"):
        self.results: List[BenchmarkResult] = []
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _generate_test_data(
        self,
        n_studies: int,
        n_per_group: int,
        effect_size: float = 0.5,
        scale_multiplier: float = 1.5
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate synthetic test data for benchmarking.

        Parameters
        ----------
        n_studies : int
            Number of studies to generate
        n_per_group : int
            Sample size per group (control and treatment)
        effect_size : float
            Mean shift for treatment group
        scale_multiplier : float
            Variance multiplier for treatment group (creates heterogeneity)

        Returns
        -------
        list of tuples
            List of (control, treatment) arrays for each study
        """
        np.random.seed(42)  # Reproducible data
        studies = []

        for i in range(n_studies):
            # Base parameters with slight variation across studies
            base_scale = np.random.uniform(0.8, 1.2)
            base_mean = np.random.uniform(-0.2, 0.2)

            # Control group
            control = np.random.exponential(base_scale, n_per_group) + base_mean - 1

            # Treatment group: larger variance + location shift
            # This creates quantile-dependent effects
            variance_multiplier = scale_multiplier * np.random.uniform(0.9, 1.1)
            treatment = (np.random.exponential(base_scale, n_per_group) - 1) * variance_multiplier + effect_size

            studies.append((control, treatment))

        return studies

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        return None

    def _run_benchmark(
        self,
        name: str,
        func: Callable,
        config: Dict
    ) -> BenchmarkResult:
        """
        Run a single benchmark.

        Parameters
        ----------
        name : str
            Benchmark name
        func : callable
            Function to benchmark
        config : dict
            Configuration parameters

        Returns
        -------
        BenchmarkResult
            Result of the benchmark
        """
        result = BenchmarkResult(name, config)

        # Force garbage collection before benchmark
        gc.collect()

        # Get initial memory
        initial_memory = self._get_memory_usage()

        # Run benchmark with timing
        start_time = time.time()
        try:
            func()
            result.elapsed_time = time.time() - start_time
            result.success = True

            # Get peak memory
            peak_memory = self._get_memory_usage()
            if initial_memory is not None and peak_memory is not None:
                result.memory_mb = peak_memory - initial_memory

        except Exception as e:
            result.elapsed_time = time.time() - start_time
            result.success = False
            result.error = str(e)

        return result

    def run_all_benchmarks(self) -> None:
        """Run all benchmarks in the suite."""
        print("=" * 70)
        print("IPD-QMA Performance Benchmark Suite")
        print("=" * 70)
        print()

        # Benchmark 1: Scalability with number of studies
        print("[1/5] Benchmarking: Scalability with number of studies...")
        for n_studies in [5, 10, 20, 50]:
            studies = self._generate_test_data(n_studies, 100)

            def run():
                config = IQMAConfig(
                    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
                    n_bootstrap=500,
                    use_random_effects=True,
                    show_progress=False
                )
                analyzer = IPDQMA(config)
                analyzer.fit(studies)

            result = self._run_benchmark(
                f"{n_studies} studies",
                run,
                {'n_studies': n_studies, 'n_bootstrap': 500, 'n_per_group': 100}
            )
            self.results.append(result)

        # Benchmark 2: Scalability with bootstrap samples
        print("[2/5] Benchmarking: Scalability with bootstrap samples...")
        for n_boot in [100, 500, 1000, 2000]:
            studies = self._generate_test_data(10, 100)

            def run():
                config = IQMAConfig(
                    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
                    n_bootstrap=n_boot,
                    use_random_effects=True,
                    show_progress=False
                )
                analyzer = IPDQMA(config)
                analyzer.fit(studies)

            result = self._run_benchmark(
                f"{n_boot} bootstrap samples",
                run,
                {'n_studies': 10, 'n_bootstrap': n_boot, 'n_per_group': 100}
            )
            self.results.append(result)

        # Benchmark 3: Scalability with sample size
        print("[3/5] Benchmarking: Scalability with sample size per study...")
        for n_per_group in [50, 100, 200, 500]:
            studies = self._generate_test_data(10, n_per_group)

            def run():
                config = IQMAConfig(
                    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
                    n_bootstrap=500,
                    use_random_effects=True,
                    show_progress=False
                )
                analyzer = IPDQMA(config)
                analyzer.fit(studies)

            result = self._run_benchmark(
                f"{n_per_group} per group",
                run,
                {'n_studies': 10, 'n_bootstrap': 500, 'n_per_group': n_per_group}
            )
            self.results.append(result)

        # Benchmark 4: Parallel vs Sequential processing
        print("[4/5] Benchmarking: Parallel vs Sequential processing...")
        studies = self._generate_test_data(20, 100)

        # Sequential
        def run_sequential():
            config = IQMAConfig(
                quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
                n_bootstrap=1500,  # Above threshold for parallel
                n_workers=1,  # Sequential
                show_progress=False
            )
            analyzer = IPDQMA(config)
            analyzer.fit(studies)

        result_seq = self._run_benchmark(
            "Sequential (n_workers=1)",
            run_sequential,
            {'n_studies': 20, 'n_bootstrap': 1500, 'n_workers': 1}
        )
        self.results.append(result_seq)

        # Parallel
        def run_parallel():
            config = IQMAConfig(
                quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
                n_bootstrap=1500,  # Above threshold for parallel
                n_workers=None,  # Auto (use all CPUs)
                show_progress=False
            )
            analyzer = IPDQMA(config)
            analyzer.fit(studies)

        result_par = self._run_benchmark(
            "Parallel (n_workers=auto)",
            run_parallel,
            {'n_studies': 20, 'n_bootstrap': 1500, 'n_workers': 'auto'}
        )
        self.results.append(result_par)

        # Benchmark 5: Different quantile configurations
        print("[5/5] Benchmarking: Different quantile configurations...")
        for n_quantiles in [3, 5, 9, 19]:
            quantiles = np.linspace(0.05, 0.95, n_quantiles).tolist()
            studies = self._generate_test_data(10, 100)

            def run():
                config = IQMAConfig(
                    quantiles=quantiles,
                    n_bootstrap=500,
                    use_random_effects=True,
                    show_progress=False
                )
                analyzer = IPDQMA(config)
                analyzer.fit(studies)

            result = self._run_benchmark(
                f"{n_quantiles} quantiles",
                run,
                {'n_studies': 10, 'n_bootstrap': 500, 'n_quantiles': n_quantiles}
            )
            self.results.append(result)

    def print_results(self) -> None:
        """Print benchmark results in a formatted table."""
        print()
        print("=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)
        print()

        # Group results by category
        categories = {
            'Studies': [r for r in self.results if 'studies' in r.name],
            'Bootstrap': [r for r in self.results if 'bootstrap' in r.name],
            'Sample Size': [r for r in self.results if 'per group' in r.name],
            'Processing': [r for r in self.results if 'Sequential' in r.name or 'Parallel' in r.name],
            'Quantiles': [r for r in self.results if 'quantiles' in r.name]
        }

        for category, results in categories.items():
            if not results:
                continue
            print(f"{category}:")
            print("-" * 70)
            for r in results:
                status = "[OK]" if r.success else "[FAIL]"
                time_str = f"{r.elapsed_time:.2f}s" if r.elapsed_time else "N/A"
                mem_str = f"{r.memory_mb:.1f}MB" if r.memory_mb else "N/A"
                print(f"  {status} {r.name:25} | Time: {time_str:10} | Memory: {mem_str:10}")
                if r.error:
                    print(f"      Error: {r.error}")
            print()

        # Performance targets check
        print("=" * 70)
        print("PERFORMANCE TARGETS")
        print("=" * 70)
        print()

        targets = [
            ("10 studies, 1000 bootstrap", 30, {'n_studies': 10, 'n_bootstrap': 1000}),
            ("50 studies, 500 bootstrap", 120, {'n_studies': 50, 'n_bootstrap': 500}),
        ]

        all_passed = True
        for target_name, max_seconds, config in targets:
            matching = [r for r in self.results
                       if r.config.get('n_studies') == config.get('n_studies') and
                          r.config.get('n_bootstrap') == config.get('n_bootstrap')]
            if matching and matching[0].success:
                elapsed = matching[0].elapsed_time
                passed = elapsed <= max_seconds
                all_passed = all_passed and passed
                status = "[OK] PASS" if passed else "[FAIL] FAIL"
                print(f"  {status}  {target_name}: {elapsed:.2f}s (target: {max_seconds}s)")
            else:
                print(f"  ?      {target_name}: Not tested")

        print()
        if all_passed:
            print("  [OK] All performance targets met!")
        else:
            print("  [FAIL] Some performance targets not met")

        print()

    def save_results(self) -> None:
        """Save benchmark results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"benchmark_{timestamp}.json")

        output = {
            'timestamp': datetime.now().isoformat(),
            'results': [r.to_dict() for r in self.results]
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to: {filename}")


def run_quick_benchmark():
    """Run a quick benchmark for basic validation."""
    print("Running quick IPD-QMA benchmark...")
    print()

    suite = BenchmarkSuite()

    # Quick test: 10 studies, 500 bootstrap
    studies_data = []
    np.random.seed(42)
    for i in range(10):
        control = np.random.exponential(1, 100) - 1
        treatment = (np.random.exponential(1, 100) - 1) * 2 + 0.5
        studies_data.append((control, treatment))

    start = time.time()
    config = IQMAConfig(
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
        n_bootstrap=500,
        use_random_effects=True,
        show_progress=False
    )
    analyzer = IPDQMA(config)
    results = analyzer.fit(studies_data)
    elapsed = time.time() - start

    print(f"[OK] Completed in {elapsed:.2f} seconds")
    print(f"  - {len(studies_data)} studies")
    print(f"  - {config.n_bootstrap} bootstrap samples")
    print(f"  - {len(config.quantiles)} quantiles")
    print(f"  - Model: {results['model_type']}")
    print()

    if elapsed < 10:
        print("[OK] Performance: Excellent")
    elif elapsed < 30:
        print("[OK] Performance: Good")
    else:
        print("[!] Performance: Consider optimization")

    return elapsed


def main():
    """Main entry point for benchmark suite."""
    import argparse

    parser = argparse.ArgumentParser(description="IPD-QMA Performance Benchmarks")
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark only')
    parser.add_argument('--output', default='benchmark_results', help='Output directory for results')

    args = parser.parse_args()

    if args.quick:
        run_quick_benchmark()
    else:
        suite = BenchmarkSuite(output_dir=args.output)
        suite.run_all_benchmarks()
        suite.print_results()
        suite.save_results()


if __name__ == "__main__":
    main()
