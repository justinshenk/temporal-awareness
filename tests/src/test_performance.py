"""Performance benchmarks and device comparison tests.

These tests are slow because they:
- Run multiple iterations for timing accuracy
- Compare CPU vs GPU/MPS performance
- Test memory usage

Skip with: pytest --skip-slow
"""

import gc
import time

import numpy as np
import pytest
import torch

from src.models.model_runner import ModelRunner, ModelBackend


TEST_MODEL = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def transformerlens_runner():
    """Load model with TransformerLens backend once per module."""
    runner = ModelRunner(TEST_MODEL, backend=ModelBackend.TRANSFORMERLENS)
    yield runner
    del runner
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks across backends and devices."""

    @staticmethod
    def _measure_generation(runner, prompt, max_tokens=10, n_runs=2):
        """Measure generation time and return stats."""
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            runner.generate(prompt, max_new_tokens=max_tokens, temperature=0.0)
            times.append(time.perf_counter() - start)
        return {"mean": np.mean(times), "std": np.std(times), "min": min(times)}

    @staticmethod
    def _measure_activation_capture(runner, prompt, layer, n_runs=2):
        """Measure activation capture time."""
        names_filter = lambda n: n == f"blocks.{layer}.hook_resid_post"
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            runner.run_with_cache(prompt, names_filter)
            times.append(time.perf_counter() - start)
        return {"mean": np.mean(times), "std": np.std(times), "min": min(times)}

    @staticmethod
    def _get_memory_usage():
        """Get current memory usage in MB."""
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def test_backend_generation_speed(self, transformerlens_runner):
        """Benchmark generation speed for TransformerLens backend."""
        prompt = "The quick brown fox"
        stats = self._measure_generation(transformerlens_runner, prompt)
        print(
            f"\nTransformerLens generation: {stats['mean']:.3f}s +/- {stats['std']:.3f}s"
        )
        assert stats["mean"] < 30, "Generation too slow"

    def test_activation_capture_speed(self, transformerlens_runner):
        """Benchmark activation capture speed."""
        prompt = "Testing activation capture performance"
        layer = transformerlens_runner.n_layers // 2
        stats = self._measure_activation_capture(transformerlens_runner, prompt, layer)
        print(f"\nActivation capture: {stats['mean']:.3f}s +/- {stats['std']:.3f}s")
        assert stats["mean"] < 10, "Activation capture too slow"

    def test_memory_usage_during_generation(self, transformerlens_runner):
        """Check memory usage doesn't explode during generation."""
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        mem_before = self._get_memory_usage()
        prompt = "Memory test " * 5

        for _ in range(3):
            transformerlens_runner.generate(prompt, max_new_tokens=20, temperature=0.0)

        gc.collect()
        mem_after = self._get_memory_usage()
        mem_increase = mem_after - mem_before

        print(
            f"\nMemory: before={mem_before:.1f}MB, after={mem_after:.1f}MB, increase={mem_increase:.1f}MB"
        )
        assert mem_increase < 500, f"Memory increased by {mem_increase:.1f}MB"


@pytest.mark.slow
class TestDeviceComparison:
    """Compare CPU vs GPU/MPS performance."""

    @staticmethod
    def _get_available_device():
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def test_device_comparison_generation(self):
        """Compare generation speed across devices."""
        prompt = "The quick brown fox"
        results = {}

        # Always test CPU
        cpu_runner = ModelRunner(
            TEST_MODEL, backend=ModelBackend.TRANSFORMERLENS, device="cpu"
        )
        times = []
        for _ in range(2):
            start = time.perf_counter()
            cpu_runner.generate(prompt, max_new_tokens=10, temperature=0.0)
            times.append(time.perf_counter() - start)
        results["cpu"] = np.mean(times)
        del cpu_runner
        gc.collect()

        # Test GPU/MPS if available
        accel_device = self._get_available_device()
        if accel_device != "cpu":
            accel_runner = ModelRunner(
                TEST_MODEL, backend=ModelBackend.TRANSFORMERLENS, device=accel_device
            )
            times = []
            for _ in range(2):
                start = time.perf_counter()
                accel_runner.generate(prompt, max_new_tokens=10, temperature=0.0)
                times.append(time.perf_counter() - start)
            results[accel_device] = np.mean(times)
            del accel_runner
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print("\nGeneration speed comparison:")
        for device, time_s in results.items():
            print(f"  {device}: {time_s:.3f}s")

        if accel_device != "cpu":
            speedup = results["cpu"] / results[accel_device]
            print(f"  Speedup ({accel_device} vs cpu): {speedup:.2f}x")

    def test_device_comparison_activation_capture(self):
        """Compare activation capture speed across devices."""
        prompt = "Testing activation capture"
        results = {}

        def measure(runner):
            layer = runner.n_layers // 2
            names_filter = lambda n: n == f"blocks.{layer}.hook_resid_post"
            times = []
            for _ in range(2):
                start = time.perf_counter()
                runner.run_with_cache(prompt, names_filter)
                times.append(time.perf_counter() - start)
            return np.mean(times)

        # CPU
        cpu_runner = ModelRunner(
            TEST_MODEL, backend=ModelBackend.TRANSFORMERLENS, device="cpu"
        )
        results["cpu"] = measure(cpu_runner)
        del cpu_runner
        gc.collect()

        # GPU/MPS
        accel_device = self._get_available_device()
        if accel_device != "cpu":
            accel_runner = ModelRunner(
                TEST_MODEL, backend=ModelBackend.TRANSFORMERLENS, device=accel_device
            )
            results[accel_device] = measure(accel_runner)
            del accel_runner
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print("\nActivation capture speed comparison:")
        for device, time_s in results.items():
            print(f"  {device}: {time_s:.3f}s")

        if accel_device != "cpu":
            speedup = results["cpu"] / results[accel_device]
            print(f"  Speedup ({accel_device} vs cpu): {speedup:.2f}x")
