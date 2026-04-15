"""
Differentiable Log-Gabor filter bank and Phase Congruency computation.

Implements the core physics-based feature extraction for PhasePhyto.
All filters are pre-computed and registered as non-learnable buffers.
Computation is performed entirely in the frequency domain via FFT for efficiency.

The key mathematical property exploited: PC(image) = PC(image * k) for any k > 0,
providing amplitude (illumination) invariance by construction.

Reference:
    Kovesi, P. (1999). "Image Features from Phase Congruency."
    Vidyarthi et al. (2024). "PhaseHisto" - medical histopathology predecessor.
"""

import math
from typing import cast

import torch
import torch.nn as nn


class LogGaborFilterBank(nn.Module):
    """Pre-computed Log-Gabor filter bank in the frequency domain.

    Generates ``num_scales * num_orientations`` filters and stores them as
    ``register_buffer`` (zero trainable parameters).  Forward pass applies all
    filters via rfft2 pointwise multiplication + irfft2.

    Args:
        image_size: (H, W) spatial dimensions of input images.
        num_scales: Number of wavelet scales.
        num_orientations: Number of filter orientations.
        min_wavelength: Wavelength of smallest-scale filter (pixels).
        mult: Scaling factor between successive filter scales.
        sigma_on_f: Ratio of the standard deviation of the Gaussian describing
            the Log-Gabor filter's transfer function in the frequency domain
            to the filter center frequency.
        d_theta_on_sigma: Scaling factor for angular bandwidth.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        num_scales: int = 4,
        num_orientations: int = 6,
        min_wavelength: float = 3.0,
        mult: float = 2.1,
        sigma_on_f: float = 0.55,
        d_theta_on_sigma: float = 1.2,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.num_orientations = num_orientations
        self.image_size = image_size

        H, W = image_size

        # --- Frequency coordinate grids (rfft2 layout) ---
        # Match the unshifted layout returned by torch.fft.rfft2: DC is at [0, 0],
        # vertical frequencies wrap after the Nyquist bin, and horizontal frequencies
        # are the non-negative half spectrum.  Using fftshifted coordinates here would
        # multiply each FFT bin by a filter for the wrong frequency.
        u = torch.fft.fftfreq(H, dtype=torch.float32)
        v = torch.fft.rfftfreq(W, dtype=torch.float32)

        grid_u, grid_v = torch.meshgrid(u, v, indexing="ij")  # (H, W//2+1)

        radius = torch.sqrt(grid_u**2 + grid_v**2)
        radius = torch.clamp(radius, min=1e-8)  # avoid log(0)
        theta = torch.atan2(grid_u, grid_v)

        # Angular Gaussian bandwidth
        d_theta = math.pi / num_orientations / d_theta_on_sigma

        # --- Build filter bank ---
        filters = []
        sign_masks = []
        for s in range(num_scales):
            wavelength = min_wavelength * (mult**s)
            fo = 1.0 / wavelength  # centre frequency for this scale

            # Radial Log-Gabor component
            log_ratio = torch.log(radius / fo)
            log_sigma = math.log(sigma_on_f)
            radial = torch.exp(-0.5 * (log_ratio**2) / (log_sigma**2))
            radial[0, 0] = 0.0  # zero DC component

            for o in range(num_orientations):
                angle = o * math.pi / num_orientations

                # Angular Gaussian component (handles wraparound)
                ds = torch.sin(theta - angle)
                dc = torch.cos(theta - angle)
                d_angle = torch.abs(torch.atan2(ds, dc))
                angular = torch.exp(-0.5 * (d_angle**2) / (d_theta**2))

                filters.append(radial * angular)

                # Odd-symmetric quadrature responses use an oriented Hilbert phase
                # shift.  The sign is taken along the current filter orientation.
                projection = grid_v * math.cos(angle) + grid_u * math.sin(angle)
                sign_masks.append(torch.sign(projection))

        # (num_scales * num_orientations, H, W//2+1)
        self.register_buffer("filter_bank", torch.stack(filters, dim=0))
        self.register_buffer("sign_mask", torch.stack(sign_masks, dim=0))

    @property
    def num_filters(self) -> int:
        return self.num_scales * self.num_orientations

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply filter bank to grayscale input via FFT.

        Args:
            x: (B, 1, H, W) grayscale image tensor.

        Returns:
            even: (B, N_filters, H, W) even-symmetric (real) responses.
            odd:  (B, N_filters, H, W) odd-symmetric (quadrature) responses.
        """
        # x -> frequency domain  (B, 1, H, W//2+1) complex
        X_freq = torch.fft.rfft2(x)
        filter_bank = cast(torch.Tensor, self.get_buffer("filter_bank"))
        sign_mask = cast(torch.Tensor, self.get_buffer("sign_mask"))

        # Broadcast multiply: (B, 1, H, W//2+1) * (1, N, H, W//2+1)
        filtered = X_freq * filter_bank.unsqueeze(0)  # (B, N, H, W//2+1)

        # Even (real) responses via inverse FFT
        even = torch.fft.irfft2(filtered, s=self.image_size)  # (B, N, H, W)

        # Odd (quadrature) responses via an oriented Hilbert-like transform.
        hilbert = filtered * (-1j * sign_mask.unsqueeze(0))
        odd = torch.fft.irfft2(hilbert, s=self.image_size)  # (B, N, H, W)

        return even, odd


class PhaseCongruencyExtractor(nn.Module):
    """Compute Phase Congruency structural maps from filter bank responses.

    Produces three complementary, illumination-invariant structural maps:

    * **PC Magnitude** -- scale-invariant edges and boundaries.
    * **Phase Symmetry** -- symmetric morphological structures (circles, pores).
    * **Oriented Energy** -- directional texture patterns (veins, hyphae).

    Args:
        num_scales: Must match ``LogGaborFilterBank.num_scales``.
        num_orientations: Must match ``LogGaborFilterBank.num_orientations``.
        noise_threshold_k: Multiplier for median-based noise threshold.
        epsilon: Numerical stability constant (>= 1e-6 to avoid NaN grads).
        cutoff: Sigmoid cutoff for frequency-spread weighting (0-1).
        gain: Sigmoid gain for frequency-spread weighting.
    """

    def __init__(
        self,
        num_scales: int = 4,
        num_orientations: int = 6,
        noise_threshold_k: float = 3.0,
        epsilon: float = 1e-6,
        cutoff: float = 0.5,
        gain: float = 10.0,
    ):
        super().__init__()
        self.ns = num_scales
        self.no = num_orientations
        self.noise_k = noise_threshold_k
        self.eps = epsilon
        self.cutoff = cutoff
        self.gain = gain

    def _frequency_spread_weight(self, amplitude: torch.Tensor) -> torch.Tensor:
        """Weighting based on how many scales contribute at each pixel.

        Args:
            amplitude: (B, no, ns, H, W)

        Returns:
            weight: (B, no, H, W) in [0, 1]
        """
        sum_amp = amplitude.sum(dim=2)  # (B, no, H, W)
        max_amp = amplitude.max(dim=2).values  # (B, no, H, W)
        width = (
            sum_amp / max_amp.clamp_min(torch.finfo(amplitude.dtype).tiny) - 1.0
        ) / (self.ns - 1.0)
        weight = 1.0 / (1.0 + torch.exp(self.gain * (self.cutoff - width)))
        return weight

    def forward(
        self, even: torch.Tensor, odd: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute PC maps.

        Args:
            even: (B, ns*no, H, W) even-symmetric responses.
            odd:  (B, ns*no, H, W) odd-symmetric responses.

        Returns:
            dict with ``'pc_magnitude'``, ``'phase_symmetry'``,
            ``'oriented_energy'`` -- each (B, 1, H, W), normalised to [0, 1].
        """
        B, _, H, W = even.shape
        ns, no = self.ns, self.no

        # Reshape: (B, ns, no, H, W) -> (B, no, ns, H, W)
        even = even.view(B, ns, no, H, W).permute(0, 2, 1, 3, 4)
        odd = odd.view(B, ns, no, H, W).permute(0, 2, 1, 3, 4)

        # Per-filter amplitude: (B, no, ns, H, W)
        amplitude = torch.sqrt((even**2 + odd**2).clamp_min(torch.finfo(even.dtype).tiny))

        # Noise threshold from finest scale (highest frequency = most noise)
        noise_amp = amplitude[:, :, 0, :, :]  # (B, no, H, W)
        noise_median = noise_amp.flatten(2).median(dim=-1).values  # (B, no)
        T = self.noise_k * noise_median.unsqueeze(-1).unsqueeze(-1)  # (B, no, 1, 1)

        # Frequency spread weighting
        W_spread = self._frequency_spread_weight(amplitude)  # (B, no, H, W)

        # Sum of amplitudes across scales per orientation
        sum_A = amplitude.sum(dim=2)  # (B, no, H, W)

        # ---- PC Magnitude ----
        sum_even = even.sum(dim=2)  # (B, no, H, W)
        sum_odd = odd.sum(dim=2)

        energy = torch.sqrt((sum_even**2 + sum_odd**2).clamp_min(torch.finfo(even.dtype).tiny))
        pc_orient = (
            W_spread
            * torch.clamp(energy - T, min=0)
            / sum_A.clamp_min(torch.finfo(sum_A.dtype).tiny)
        )
        pc_magnitude = pc_orient.max(dim=1).values.unsqueeze(1)  # (B, 1, H, W)

        # ---- Phase Symmetry ----
        abs_even_sum = even.abs().sum(dim=2)  # (B, no, H, W)
        abs_odd_sum = odd.abs().sum(dim=2)
        sym_energy = W_spread * torch.clamp(abs_even_sum - abs_odd_sum - T, min=0)
        phase_symmetry = sym_energy / sum_A.clamp_min(torch.finfo(sum_A.dtype).tiny)
        phase_symmetry = phase_symmetry.max(dim=1).values.unsqueeze(1)

        # ---- Oriented Energy ----
        angles = torch.linspace(0, math.pi, no + 1, device=even.device)[:-1]
        cos_w = angles.cos().view(1, no, 1, 1)
        sin_w = angles.sin().view(1, no, 1, 1)
        oe_x = (energy * cos_w).sum(dim=1, keepdim=True)
        oe_y = (energy * sin_w).sum(dim=1, keepdim=True)
        oriented_energy = torch.sqrt((oe_x**2 + oe_y**2).clamp_min(torch.finfo(even.dtype).tiny))

        # Suppress circular boundary artifacts introduced by FFT filtering.  A hard
        # step image has a true interior edge and a periodic wrap edge at the image
        # border; the taper keeps the structural map focused on in-frame content.
        def _edge_taper(
            height: int,
            width: int,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            taper_y = torch.ones(height, device=device, dtype=dtype)
            taper_x = torch.ones(width, device=device, dtype=dtype)
            for taper in (taper_y, taper_x):
                ramp_width = max(1, taper.numel() // 16)
                ramp = torch.linspace(0.0, 1.0, ramp_width, device=device, dtype=dtype)
                taper[:ramp_width] = ramp
                taper[-ramp_width:] = ramp.flip(0)
            return taper_y.view(1, 1, height, 1) * taper_x.view(1, 1, 1, width)

        taper = _edge_taper(H, W, even.device, even.dtype)
        pc_magnitude = pc_magnitude * taper
        phase_symmetry = phase_symmetry * taper
        oriented_energy = oriented_energy * taper

        # Normalise each map to [0, 1] per sample
        def _norm(t: torch.Tensor) -> torch.Tensor:
            flat = t.flatten(1)
            lo = flat.min(dim=1).values.view(B, 1, 1, 1)
            hi = flat.max(dim=1).values.view(B, 1, 1, 1)
            return (t - lo) / (hi - lo).clamp_min(torch.finfo(t.dtype).tiny)

        return {
            "pc_magnitude": _norm(pc_magnitude),      # (B, 1, H, W)
            "phase_symmetry": _norm(phase_symmetry),   # (B, 1, H, W)
            "oriented_energy": _norm(oriented_energy),  # (B, 1, H, W)
        }
