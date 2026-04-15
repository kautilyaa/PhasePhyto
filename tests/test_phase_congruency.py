"""Tests for the Log-Gabor filter bank and Phase Congruency extractor."""

import math

import pytest
import torch

from phasephyto.models.phase_congruency import LogGaborFilterBank, PhaseCongruencyExtractor


@pytest.fixture
def filter_bank():
    return LogGaborFilterBank(image_size=(224, 224), num_scales=4, num_orientations=6)


@pytest.fixture
def pc_extractor():
    return PhaseCongruencyExtractor(num_scales=4, num_orientations=6)


class TestLogGaborFilterBank:
    def test_filter_bank_shape(self, filter_bank):
        """Filter bank should have num_scales * num_orientations filters."""
        assert filter_bank.filter_bank.shape[0] == 24  # 4 * 6
        assert filter_bank.filter_bank.shape[1] == 224
        assert filter_bank.filter_bank.shape[2] == 113  # 224 // 2 + 1

    def test_filters_are_buffers_not_parameters(self, filter_bank):
        """Filters must be register_buffer, not nn.Parameter."""
        param_names = {n for n, _ in filter_bank.named_parameters()}
        assert "filter_bank" not in param_names
        assert "sign_mask" not in param_names

    def test_zero_trainable_params(self, filter_bank):
        """Filter bank should have zero trainable parameters."""
        n_params = sum(p.numel() for p in filter_bank.parameters() if p.requires_grad)
        assert n_params == 0

    def test_forward_shapes(self, filter_bank):
        """Forward should return even and odd responses with correct shapes."""
        x = torch.randn(2, 1, 224, 224)
        even, odd = filter_bank(x)
        assert even.shape == (2, 24, 224, 224)
        assert odd.shape == (2, 24, 224, 224)

    def test_dc_component_zero(self, filter_bank):
        """DC component of each filter should be zero."""
        for i in range(filter_bank.filter_bank.shape[0]):
            assert filter_bank.filter_bank[i, 0, 0].item() == 0.0

    def test_amplitude_invariance(self, filter_bank):
        """PC(image) should equal PC(image * k) for positive k."""
        pc_ext = PhaseCongruencyExtractor(num_scales=4, num_orientations=6)
        x = torch.rand(1, 1, 224, 224) * 0.5 + 0.1  # avoid near-zero

        # Original
        even1, odd1 = filter_bank(x)
        pc1 = pc_ext(even1, odd1)

        # Scaled by k=3.0
        even2, odd2 = filter_bank(x * 3.0)
        pc2 = pc_ext(even2, odd2)

        for key in ["pc_magnitude", "phase_symmetry", "oriented_energy"]:
            diff = (pc1[key] - pc2[key]).abs().max().item()
            assert diff < 1e-3, f"{key} not invariant: max diff = {diff}"

    def test_amplitude_invariance_multiple_k(self, filter_bank):
        """Test invariance across several scaling factors."""
        pc_ext = PhaseCongruencyExtractor(num_scales=4, num_orientations=6)
        x = torch.rand(1, 1, 224, 224) * 0.5 + 0.1

        even_ref, odd_ref = filter_bank(x)
        pc_ref = pc_ext(even_ref, odd_ref)

        for k in [0.1, 0.5, 2.0, 5.0, 10.0]:
            even_k, odd_k = filter_bank(x * k)
            pc_k = pc_ext(even_k, odd_k)
            for key in ["pc_magnitude", "phase_symmetry", "oriented_energy"]:
                diff = (pc_ref[key] - pc_k[key]).abs().max().item()
                assert diff < 5e-3, f"k={k}, {key}: max diff = {diff}"

    def test_orientation_filter_selectivity(self, filter_bank):
        """Horizontal-frequency stripes should peak in the 0-degree filter bin."""
        width = filter_bank.image_size[1]
        height = filter_bank.image_size[0]
        x_coord = torch.linspace(0, 8 * math.pi, width).view(1, 1, 1, width)
        stripes = ((torch.sin(x_coord) + 1.0) / 2.0).expand(1, 1, height, width)

        even, odd = filter_bank(stripes)
        amplitude = torch.sqrt(even**2 + odd**2)
        orientation_energy = amplitude.view(
            1,
            filter_bank.num_scales,
            filter_bank.num_orientations,
            height,
            width,
        ).mean(dim=(0, 1, 3, 4))

        assert orientation_energy.argmax().item() == 0
        assert orientation_energy[0] > orientation_energy[1] * 1.5


class TestPhaseCongruencyExtractor:
    def test_output_shapes(self, filter_bank, pc_extractor):
        """All PC maps should be (B, 1, H, W) with correct dimensions."""
        x = torch.randn(2, 1, 224, 224)
        even, odd = filter_bank(x)
        maps = pc_extractor(even, odd)

        for key in ["pc_magnitude", "phase_symmetry", "oriented_energy"]:
            assert maps[key].shape == (2, 1, 224, 224), f"{key} shape mismatch"

    def test_output_range(self, filter_bank, pc_extractor):
        """All maps should be normalised to [0, 1]."""
        x = torch.randn(2, 1, 224, 224)
        even, odd = filter_bank(x)
        maps = pc_extractor(even, odd)

        for key in ["pc_magnitude", "phase_symmetry", "oriented_energy"]:
            assert maps[key].min() >= 0.0 - 1e-6, f"{key} below 0"
            assert maps[key].max() <= 1.0 + 1e-6, f"{key} above 1"

    def test_step_edge_response(self, filter_bank, pc_extractor):
        """PC magnitude should peak at a step edge."""
        x = torch.zeros(1, 1, 224, 224)
        x[:, :, :, 112:] = 1.0  # vertical edge at center

        even, odd = filter_bank(x)
        maps = pc_extractor(even, odd)

        pc_mag = maps["pc_magnitude"][0, 0]
        # Peak should be near column 112
        col_max = pc_mag.mean(dim=0).argmax().item()
        assert abs(col_max - 112) < 15, f"Edge peak at col {col_max}, expected ~112"

    def test_phase_symmetry_circle_boundary(self, filter_bank, pc_extractor):
        """Phase symmetry should respond more on a ring boundary than flat regions."""
        height, width = filter_bank.image_size
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        radius = torch.sqrt((xx - width // 2) ** 2 + (yy - height // 2) ** 2)
        circle = ((radius > 45) & (radius < 55)).float().unsqueeze(0).unsqueeze(0)

        even, odd = filter_bank(circle)
        maps = pc_extractor(even, odd)

        phase_symmetry = maps["phase_symmetry"][0, 0]
        boundary = (radius > 43) & (radius < 57)
        interior = radius < 30
        background = (radius > 80) & (radius < 95)

        boundary_response = phase_symmetry[boundary].mean()
        flat_response = torch.maximum(
            phase_symmetry[interior].mean(),
            phase_symmetry[background].mean(),
        )
        assert boundary_response > flat_response * 4

    def test_no_nan_in_outputs(self, filter_bank, pc_extractor):
        """Outputs should never contain NaN."""
        x = torch.randn(2, 1, 224, 224)
        even, odd = filter_bank(x)
        maps = pc_extractor(even, odd)
        for key, val in maps.items():
            assert not torch.isnan(val).any(), f"NaN found in {key}"
