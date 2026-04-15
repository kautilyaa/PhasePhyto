"""End-to-end model tests: forward pass, backward pass, parameter counts."""

import pytest
import torch

from phasephyto.models.cross_attention import StructuralSemanticFusion
from phasephyto.models.pc_encoder import PCEncoder
from phasephyto.models.phasephyto import PhasePhyto


@pytest.fixture
def model():
    """Create a small PhasePhyto model for testing (ViT backbone)."""
    return PhasePhyto(
        num_classes=10,
        backbone_name="vit_base_patch16_224",
        fusion_dim=256,
        pc_scales=4,
        pc_orientations=6,
        image_size=(224, 224),
        pretrained_backbone=False,  # skip download in tests
    )


class TestPCEncoder:
    def test_output_shape(self):
        enc = PCEncoder(in_channels=3, fusion_dim=256, spatial_size=7)
        x = torch.randn(2, 3, 224, 224)
        tokens = enc(x)
        assert tokens.shape == (2, 49, 256)

    def test_num_tokens(self):
        enc = PCEncoder(spatial_size=7)
        assert enc.num_tokens == 49


class TestCrossAttention:
    def test_output_shape(self):
        fusion = StructuralSemanticFusion(fusion_dim=256, num_heads=4)
        q = torch.randn(2, 49, 256)   # structural tokens
        kv = torch.randn(2, 196, 256)  # semantic tokens (ViT)
        fused, attn = fusion(q, kv, return_attention=True)
        assert fused.shape == (2, 256)
        assert attn.shape == (2, 49, 196)

    def test_attention_sums_to_one(self):
        fusion = StructuralSemanticFusion(fusion_dim=256, num_heads=4)
        q = torch.randn(2, 49, 256)
        kv = torch.randn(2, 196, 256)
        _, attn = fusion(q, kv, return_attention=True)
        # Attention weights along key dimension should sum to ~1
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    def test_fusion_parameter_count(self):
        """Fusion module should stay in the lightweight ~331K param range."""
        fusion = StructuralSemanticFusion(fusion_dim=256, num_heads=4)
        n_params = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
        # Allow range: 150K-350K
        assert 150_000 < n_params < 350_000, f"Fusion params: {n_params}"


class TestPhasePhytoModel:
    def test_forward_pass(self, model):
        """Full forward pass should produce logits of correct shape."""
        x_rgb = torch.randn(2, 3, 224, 224)
        output = model(x_rgb)
        assert "logits" in output
        assert output["logits"].shape == (2, 10)

    def test_forward_with_clahe(self, model):
        """Forward with separate CLAHE input should work."""
        x_rgb = torch.randn(2, 3, 224, 224)
        x_clahe = torch.randn(2, 3, 224, 224)
        output = model(x_rgb, x_clahe=x_clahe)
        assert output["logits"].shape == (2, 10)

    def test_return_maps(self, model):
        """return_maps=True should include PC maps in output."""
        x = torch.randn(1, 3, 224, 224)
        output = model(x, return_maps=True)
        assert "pc_magnitude" in output
        assert "phase_symmetry" in output
        assert "oriented_energy" in output
        assert output["pc_magnitude"].shape == (1, 1, 224, 224)

    def test_return_attention(self, model):
        """return_attention=True should include attention weights."""
        x = torch.randn(1, 3, 224, 224)
        output = model(x, return_attention=True)
        assert "attn_weights" in output

    def test_backward_pass(self, model):
        """Gradients should flow through the entire model."""
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        loss = output["logits"].sum()
        loss.backward()

        # Check gradients exist for key learnable modules
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_eval_mode(self, model):
        """Model should work in eval mode with no_grad."""
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output["logits"].shape == (1, 10)

    def test_parameter_count_breakdown(self, model):
        """Parameter count should be reported correctly."""
        counts = model.count_parameters()
        assert counts["filter_bank"] == 0  # non-learnable
        assert counts["pc_extractor"] == 0  # non-learnable
        assert counts["total"] > 0
        assert "backbone" in counts
        assert "fusion" in counts

    def test_rgb_to_gray_conversion(self, model):
        """Grayscale conversion should produce (B, 1, H, W)."""
        x = torch.randn(2, 3, 224, 224)
        gray = model._rgb_to_gray(x)
        assert gray.shape == (2, 1, 224, 224)
