from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
import helion.language as hl


class TestRandom(RefEagerTestBase, TestCase):
    def test_hl_rand_1d(self):
        @helion.kernel
        def rand_kernel_tiled_1d(x: torch.Tensor, seed: int) -> torch.Tensor:
            output = torch.zeros_like(x)
            (m,) = x.shape
            for (tile_m,) in hl.tile([m]):
                output[tile_m] = hl.rand([tile_m], seed=seed)
            return output

        x_small = torch.ones(128, device=DEVICE)
        _, output = code_and_output(rand_kernel_tiled_1d, (x_small, 42))
        _, output2 = code_and_output(rand_kernel_tiled_1d, (x_small, 1337))

        self.assertFalse(
            torch.allclose(output, output2),
            "Different seeds should produce different outputs",
        )

        code3, output3 = code_and_output(rand_kernel_tiled_1d, (x_small, 42))
        self.assertTrue(
            torch.allclose(output, output3),
            "Same seed should produce identical outputs",
        )

        # Check that all values are in [0, 1) range
        self.assertTrue(torch.all(output >= 0.0), "All values should be >= 0")
        self.assertTrue(torch.all(output < 1.0), "All values should be < 1")

        self.assertExpectedJournal(code3)

    def test_hl_rand_2d(self):
        @helion.kernel
        def rand_kernel_tiled_2d(x: torch.Tensor, seed: int) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                output[tile_m, tile_n] = hl.rand([tile_m, tile_n], seed=seed)
            return output

        x_small = torch.ones(128, 128, device=DEVICE)
        _, output = code_and_output(rand_kernel_tiled_2d, (x_small, 42))
        _, output2 = code_and_output(rand_kernel_tiled_2d, (x_small, 1337))

        self.assertFalse(
            torch.allclose(output, output2),
            "Different seeds should produce different outputs",
        )

        code3, output3 = code_and_output(rand_kernel_tiled_2d, (x_small, 42))
        self.assertTrue(
            torch.allclose(output, output3),
            "Same seed should produce identical outputs",
        )

        self.assertTrue(torch.all(output >= 0.0), "All values should be >= 0")
        self.assertTrue(torch.all(output < 1.0), "All values should be < 1")
        self.assertExpectedJournal(code3)

    def test_hl_rand_3d(self):
        @helion.kernel
        def rand_kernel_tiled_3d(x: torch.Tensor, seed: int) -> torch.Tensor:
            output = torch.zeros_like(x)
            b, m, n = x.shape
            for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
                output[tile_b, tile_m, tile_n] = hl.rand(
                    [tile_b, tile_m, tile_n], seed=seed
                )
            return output

        x_small = torch.ones(16, 32, 64, device=DEVICE)
        _, output = code_and_output(rand_kernel_tiled_3d, (x_small, 42))
        _, output2 = code_and_output(rand_kernel_tiled_3d, (x_small, 1337))

        self.assertFalse(
            torch.allclose(output, output2),
            "Different seeds should produce different outputs",
        )

        code3, output3 = code_and_output(rand_kernel_tiled_3d, (x_small, 42))
        self.assertTrue(
            torch.allclose(output, output3),
            "Same seed should produce identical outputs",
        )

        self.assertTrue(torch.all(output >= 0.0), "All values should be >= 0")
        self.assertTrue(torch.all(output < 1.0), "All values should be < 1")

        # Check distribution properties
        mean_val = output.mean().item()
        self.assertTrue(
            0.4 < mean_val < 0.6,
            f"Mean {mean_val:.3f} should be around 0.5 for uniform distribution",
        )
        self.assertExpectedJournal(code3)

    def test_hl_rand_block_size_determinism(self):
        @helion.kernel
        def rand_kernel_2d(x: torch.Tensor, seed: int) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                output[tile_m, tile_n] = hl.rand([tile_m, tile_n], seed=seed)
            return output

        x = torch.ones(128, 256, device=DEVICE)
        seed = 42

        _, output_32_32 = code_and_output(
            rand_kernel_2d, (x, seed), block_sizes=[32, 32]
        )
        _, output_64_64 = code_and_output(
            rand_kernel_2d, (x, seed), block_sizes=[64, 64]
        )
        _, output_128_128 = code_and_output(
            rand_kernel_2d, (x, seed), block_sizes=[128, 128]
        )
        _, output_16_32 = code_and_output(
            rand_kernel_2d, (x, seed), block_sizes=[16, 32]
        )

        torch.testing.assert_close(
            output_32_32,
            output_64_64,
            msg="rand should be deterministic across different block sizes (32x32 vs 64x64)",
        )
        torch.testing.assert_close(
            output_32_32,
            output_128_128,
            msg="rand should be deterministic across different block sizes (32x32 vs 128x128)",
        )
        torch.testing.assert_close(
            output_32_32,
            output_16_32,
            msg="rand should be deterministic across different block sizes (32x32 vs 16x32)",
        )

        self.assertTrue(torch.all(output_32_32 >= 0.0))
        self.assertTrue(torch.all(output_32_32 < 1.0))

    def test_hl_rand_uniqueness_distribution(self):
        @helion.kernel
        def rand_kernel(x: torch.Tensor, seed: int) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                output[tile_m, tile_n] = hl.rand([tile_m, tile_n], seed=seed)
            return output

        x = torch.ones(256, 256, device=DEVICE)
        seed = 1337

        _, output = code_and_output(rand_kernel, (x, seed))

        sorted_values = torch.sort(output.flatten()).values

        unique_values = torch.unique(sorted_values)
        total_values = output.numel()
        uniqueness_ratio = len(unique_values) / total_values

        self.assertGreater(
            uniqueness_ratio,
            0.99,
            f"Expected >99% unique values, got {uniqueness_ratio:.4f}",
        )

        n_quartile = total_values // 4
        q1_val = sorted_values[n_quartile].item()
        q2_val = sorted_values[2 * n_quartile].item()
        q3_val = sorted_values[3 * n_quartile].item()

        self.assertTrue(
            0.2 < q1_val < 0.3, f"First quartile {q1_val:.3f} should be around 0.25"
        )
        self.assertTrue(
            0.45 < q2_val < 0.55, f"Median {q2_val:.3f} should be around 0.5"
        )
        self.assertTrue(
            0.7 < q3_val < 0.8, f"Third quartile {q3_val:.3f} should be around 0.75"
        )

    def test_hl_rand_non_tiled_dimensions(self):
        @helion.kernel
        def rand_kernel_partial_tile(x: torch.Tensor, seed: int) -> torch.Tensor:
            output = torch.zeros_like(x)
            m, n, k = x.shape
            k = hl.specialize(k)
            for tile_m, tile_n in hl.tile([m, n]):
                output[tile_m, tile_n, :] = hl.rand([tile_m, tile_n, k], seed=seed)
            return output

        x = torch.ones(64, 64, 8, device=DEVICE)
        seed = 1337

        _, output = code_and_output(rand_kernel_partial_tile, (x, seed))

        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output < 1.0))

        code2, output2 = code_and_output(rand_kernel_partial_tile, (x, seed))
        torch.testing.assert_close(output, output2, msg="it should deterministic")

        self.assertExpectedJournal(code2)


if __name__ == "__main__":
    unittest.main()
