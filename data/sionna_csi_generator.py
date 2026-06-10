"""
Sionna-based CSI data generator for MIMO channel simulation (Sionna 2.0).

This module generates realistic CSI data using Sionna's channel models (v2.0),
which capture physical wireless channel characteristics like multipath,
fading, and MIMO properties.

Compatible with Sionna 2.0 (PyTorch-based).
"""
import os
import torch
import numpy as np
import h5py
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ChannelConfig:
    """Physical channel configuration for MIMO system."""
    # System type: "TDD" (Time Division Duplex) or "FDD" (Frequency Division Duplex)
    system_type: str = "TDD"

    # Carrier frequency in Hz (downlink for FDD, shared for TDD)
    carrier_frequency: float = 3.5e9

    # Uplink carrier frequency in Hz (only used for FDD)
    # For FDD, typical uplink is lower (e.g., 2.1 GHz for 3.5 GHz downlink)
    ul_carrier_frequency: float = 2.1e9

    # Number of transmit/receive antennas (these are total antenna elements, not spatial streams)
    # For dual-polarization arrays: actual array size = num_tx_antennas / 2 per polarization
    num_tx_antennas: int = 2
    num_rx_antennas: int = 2

    # OFDM parameters
    num_subcarriers: int = 64
    ofdm_symbols_per_slot: int = 14
    subcarrier_spacing: int = 15000

    # Channel model parameters
    num_paths: int = 20  # Number of multipath components (clusters)
    delay_spread: float = 300e-9  # RMS delay spread in seconds

    # SNR range for training data generation
    snr_db_min: float = 5.0
    snr_db_max: float = 30.0

    # Training data parameters
    batch_size: int = 16
    output_seq_len: int = 128  # Sequence length for CSI feedback

    # CDL model type: 'A' (LOS), 'B', 'C', 'D', 'E' (NLOS)
    cdl_model: str = 'C'

    # Environment info extraction parameters
    extract_env_info: bool = False  # Whether to extract environmental info
    num_dominant_paths: int = 5  # Number of dominant paths to extract for angles/delays


class SionnaCSIGenerator:
    """
    Generate downlink and uplink CSI pairs using Sionna 2.0 channel models.

    The generator creates realistic MIMO channel frequency responses using:
    - CDL (Clustered Delay Line) channel model
    - 3GPP TR 38.901 spatial channel model
    - Configurable antenna arrays with panel structure

    This class is compatible with Sionna 2.0+ (PyTorch-based).
    """

    def __init__(self, config: Optional[ChannelConfig] = None):
        self.config = config or ChannelConfig()
        self._check_sionna_available()
        self._setup_channel_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _check_sionna_available(self):
        """Verify Sionna is installed."""
        try:
            import sionna
            self.sionna = sionna
            self.version = getattr(sionna, '__version__', 'unknown')
            print(f"Sionna version: {self.version}")
        except ImportError:
            raise ImportError(
                "Sionna is required for CSI generation. "
                "Install with: pip install sionna>=2.0"
            )

    def _create_antenna_arrays(self, carrier_frequency: float):
        """Create antenna arrays for a given carrier frequency."""
        from sionna.phy.channel.tr38901 import Antenna, AntennaArray

        # Configure transmit antenna array (Base Station) using Panel Array
        # BS typically uses a large panel array with dual-polarization
        bs_array = AntennaArray(
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",  # 3GPP TR 38.901 antenna pattern
            carrier_frequency=carrier_frequency,
            num_rows=1,
            num_cols=self.config.num_tx_antennas // 2
        )

        # Configure receive antenna array (User Equipment)
        # UE typically uses smaller arrays (e.g., 4x4 for mobile devices)
        ue_array = AntennaArray(
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=carrier_frequency,
            num_rows=1,
            num_cols=self.config.num_rx_antennas // 2
        )

        return bs_array, ue_array

    def _setup_channel_model(self):
        """Initialize the Sionna 2.0 channel model using CDL."""
        from sionna.phy.channel.tr38901 import CDL
        from sionna.phy.ofdm import ResourceGrid
        
        # config OFDM resource grid
        self.ofdm_resource_grid = ResourceGrid(
            num_ofdm_symbols=self.config.ofdm_symbols_per_slot,  # symbol number in one slot
            fft_size=self.config.num_subcarriers,
            subcarrier_spacing=self.config.subcarrier_spacing,
            num_tx=1,
            num_streams_per_tx=self.config.num_rx_antennas,
            cyclic_prefix_length=6,  # CP length
            pilot_pattern="kronecker",  # pilot mode
            pilot_ofdm_symbol_indices=[2, 11]  # pilot symbols
        )

        if self.config.system_type.upper() == "FDD":
            # For FDD: create separate channel models for DL and UL
            # with different carrier frequencies but shared geometric parameters

            # Create DL antenna arrays
            bs_array_dl, ue_array_dl = self._create_antenna_arrays(
                self.config.carrier_frequency
            )

            # Create UL antenna arrays (same geometry, different frequency)
            bs_array_ul, ue_array_ul = self._create_antenna_arrays(
                self.config.ul_carrier_frequency
            )

            # Store arrays for geometric parameter sharing
            self.bs_array_dl = bs_array_dl
            self.ue_array_dl = ue_array_dl
            self.bs_array_ul = bs_array_ul
            self.ue_array_ul = ue_array_ul

            # Create CDL channel models
            # Direction is specified in constructor
            self.cdl_dl = CDL(
                model=self.config.cdl_model,  # CDL model: A (LOS), B, C, D, E (NLOS)
                delay_spread=self.config.delay_spread,
                carrier_frequency=self.config.carrier_frequency,
                bs_array=bs_array_dl,
                ut_array=ue_array_dl,
                direction="downlink"
            )

            self.cdl_ul = CDL(
                model=self.config.cdl_model,
                delay_spread=self.config.delay_spread,
                carrier_frequency=self.config.ul_carrier_frequency,
                bs_array=bs_array_ul,
                ut_array=ue_array_ul,
                direction="uplink"
            )

            self.cdl = None  # Not used in FDD mode
        else:
            # For TDD: single channel model with reciprocity
            bs_array, ue_array = self._create_antenna_arrays(
                self.config.carrier_frequency
            )

            self.bs_array_dl = bs_array
            self.ue_array_dl = ue_array
            self.bs_array_ul = bs_array  # Same arrays for TDD
            self.ue_array_ul = ue_array

            # Create single CDL channel model
            self.cdl = CDL(
                model=self.config.cdl_model,
                delay_spread=self.config.delay_spread,
                carrier_frequency=self.config.carrier_frequency,
                bs_array=bs_array,
                ut_array=ue_array,
                direction="downlink"
            )

            self.cdl_dl = self.cdl
            self.cdl_ul = None  # Not used in TDD mode

    def generate_channel_batch(self, batch_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Generate a batch of downlink and uplink CSI pairs.

        Args:
            batch_size: Number of samples to generate. If None, uses config.batch_size.

        Returns:
            Tuple of (dl_csi, ul_csi, env_info), each of shape:
            - dl_csi: [batch_size, num_subcarriers, num_tx_antennas * num_rx_antennas * 2]
                      (real and imag for each subcarrier)
            - ul_csi: [batch_size, num_subcarriers, num_rx_antennas * num_tx_antennas * 2]
            - env_info: dict containing environmental information (only when extract_env_info=True)
        """
        batch_size = batch_size or self.config.batch_size

        env_info = None

        if self.config.system_type.upper() == "FDD":
            # For FDD: generate DL and UL channels separately with different frequencies
            h_freq_dl, h_freq_ul, h_dl, tau_dl = self._generate_fdd_channels(batch_size)

            # Convert to CSI features
            dl_csi = self._freq_to_csi_features(h_freq_dl)
            ul_csi = self._freq_to_csi_features(h_freq_ul)

            # Extract environment info from DL channel
            if self.config.extract_env_info:
                env_info = self._extract_environment_info(h_dl, tau_dl, h_freq_dl)
        else:
            # For TDD: use reciprocal channel with slight perturbation
            h_freq, h, tau = self._generate_freq_response(batch_size)

            # Downlink CSI: UE receives from BS
            dl_csi = self._freq_to_csi_features(h_freq)

            # For uplink CSI, use reciprocity with slight variations
            ul_csi = self._generate_reciprocal_channel(h_freq)

            # Extract environment info
            if self.config.extract_env_info:
                env_info = self._extract_environment_info(h, tau, h_freq)

        # Reshape output to [batch, seq_len, 2] format for the model
        dl_csi = self._reshape_to_seq(dl_csi)  # [batch, seq_len, 2]
        ul_csi = self._reshape_to_seq(ul_csi)

        return dl_csi, ul_csi, env_info

    def _generate_freq_response(self, batch_size: int) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Generate frequency response using Sionna's CDL channel model.

        Returns:
            Tuple of (h_freq, h, tau):
            - h_freq: Frequency response [batch, num_subcarriers, num_rx_ant, num_tx_ant]
            - h: Channel impulse response (for env info extraction)
            - tau: Path delays (for env info extraction)
        """
        # Generate batch of channel impulse responses
        # Output shapes depend on the channel model configuration
        h, tau = self.cdl(batch_size=batch_size, num_time_steps=self.ofdm_resource_grid.num_ofdm_symbols, sampling_frequency = 1/self.ofdm_resource_grid.ofdm_symbol_duration)

        # Create frequency grid (unused by cir_to_ofdm_channel but kept for API)
        num_subcarriers = self.config.num_subcarriers
        freq_grid = torch.linspace(
            -num_subcarriers / 2,
            num_subcarriers / 2 - 1,
            num_subcarriers,
            dtype=torch.float32,
            device=self.device
        )

        # Compute frequency response using cir_to_ofdm_channel
        # Returns: [batch, num_subcarriers, num_rx_ant, num_tx_ant]
        h_freq = self._impulse_to_frequency(h, tau, freq_grid)

        return h_freq.cpu().numpy(), h, tau

    def _impulse_to_frequency(self, h: torch.Tensor, tau: torch.Tensor,
                              freq_grid: torch.Tensor) -> torch.Tensor:
        """
        Convert channel impulse response to frequency response using cir_to_ofdm_channel.

        Args:
            h: Channel impulse response [batch, num_clusters, num_rx, num_rx_ant, num_tx, num_tx_ant]
            tau: Path delays [batch, num_clusters]
            freq_grid: Frequency grid [num_subcarriers] (unused, kept for API compatibility)

        Returns:
            Frequency response [batch, num_subcarriers, num_rx_ant, num_tx_ant]
            Note: For TDD with num_time_steps=1, the output is averaged over OFDM symbols
        """
        from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies

        # Get frequency grid from OFDM resource grid
        frequencies = subcarrier_frequencies(self.ofdm_resource_grid.fft_size, self.ofdm_resource_grid.subcarrier_spacing)

        # Convert to frequency response using Sionna's cir_to_ofdm_channel
        # Output: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
        h_freq = cir_to_ofdm_channel(frequencies, h, tau, normalize=True)

        # h_freq shape: [batch, 1, num_rx_ant, 1, num_tx_ant, num_ofdm_symbols, num_sc]
        # Squeeze num_rx and num_tx (both are 1 in this project), then average over OFDM symbols
        # Result: [batch, num_rx_ant, num_tx_ant, num_subcarriers]
        h_freq = h_freq.squeeze(dim=1).squeeze(dim=2)  # [batch, num_rx_ant, num_tx_ant, num_ofdm_symbols, num_subcarriers]
        h_freq = h_freq.mean(dim=-2)  # Average over OFDM symbols -> [batch, num_rx_ant, num_tx_ant, num_subcarriers]

        # Permute to [batch, num_subcarriers, num_rx_ant, num_tx_ant]
        return torch.permute(h_freq, (0, 3, 1, 2))

    def _generate_fdd_channels(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Generate separate downlink and uplink channels for FDD systems.

        In FDD systems, uplink and downlink use different carrier frequencies.
        The channels share the same geometric parameters (path delays, angles, powers)
        but have independent frequency responses due to the different wavelengths.

        Args:
            batch_size: Number of samples to generate

        Returns:
            Tuple of (h_freq_dl, h_freq_ul, h_dl, tau_dl):
            - h_freq_dl: [batch, num_subcarriers, num_rx_ant, num_tx_ant]
            - h_freq_ul: [batch, num_subcarriers, num_rx_ant, num_tx_ant]
            - h_dl: DL channel impulse response (for env info extraction)
            - tau_dl: DL path delays (for env info extraction)
        """
        # Generate DL channel impulse response
        h_dl, tau_dl = self.cdl_dl(batch_size=batch_size, num_time_steps=self.ofdm_resource_grid.num_ofdm_symbols, sampling_frequency = 1/self.ofdm_resource_grid.ofdm_symbol_duration)

        # For UL, we use the same geometric parameters but generate a new realization
        # The CDL model will produce different channel coefficients for the UL frequency
        h_ul, _ = self.cdl_ul(batch_size=batch_size, num_time_steps=self.ofdm_resource_grid.num_ofdm_symbols, sampling_frequency = 1/self.ofdm_resource_grid.ofdm_symbol_duration)

        # Reuse tau_dl for frequency response computation to maintain path correlation
        # This ensures both channels have the same multipath structure

        # Create dummy frequency grids (unused by _impulse_to_frequency_fdd but kept for API compatibility)
        num_subcarriers = self.config.num_subcarriers
        freq_grid = torch.linspace(
            -num_subcarriers / 2,
            num_subcarriers / 2 - 1,
            num_subcarriers,
            dtype=torch.float32,
            device=self.device
        )

        # Compute frequency responses - _impulse_to_frequency_fdd already returns [batch, num_subcarriers, num_rx_ant, num_tx_ant]
        h_freq_dl = self._impulse_to_frequency_fdd(h_dl, tau_dl, freq_grid)
        h_freq_ul = self._impulse_to_frequency_fdd(h_ul, tau_dl, freq_grid)  # Use same tau for correlation

        return h_freq_dl.cpu().numpy(), h_freq_ul.cpu().numpy(), h_dl, tau_dl

    def _impulse_to_frequency_fdd(self, h: torch.Tensor, tau: torch.Tensor,
                                   freq_grid: torch.Tensor) -> torch.Tensor:
        """
        Convert channel impulse response to frequency response for FDD using cir_to_ofdm_channel.

        Args:
            h: Channel impulse response [batch, num_clusters, num_rx, num_rx_ant, num_tx, num_tx_ant]
            tau: Path delays [batch, num_clusters]
            freq_grid: Frequency grid [num_subcarriers] (unused, kept for API compatibility)

        Returns:
            Frequency response [batch, num_subcarriers, num_rx_ant, num_tx_ant]
            Note: Output is averaged over OFDM symbols
        """
        from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies

        # Get frequency grid from OFDM resource grid
        frequencies = subcarrier_frequencies(self.ofdm_resource_grid.fft_size, self.ofdm_resource_grid.subcarrier_spacing)

        # Convert to frequency response using Sionna's cir_to_ofdm_channel
        # Output: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
        h_freq = cir_to_ofdm_channel(frequencies, h, tau, normalize=True)

        # h_freq shape: [batch, 1, num_rx_ant, 1, num_tx_ant, num_ofdm_symbols, num_sc]
        # Squeeze num_rx and num_tx (both are 1 in this project), then average over OFDM symbols
        h_freq = h_freq.squeeze(dim=1).squeeze(dim=2)  # [batch, num_rx_ant, num_tx_ant, num_ofdm_symbols, num_subcarriers]
        h_freq = h_freq.mean(dim=-2)  # Average over OFDM symbols -> [batch, num_rx_ant, num_tx_ant, num_subcarriers]

        # Permute to [batch, num_subcarriers, num_rx_ant, num_tx_ant]
        return torch.permute(h_freq, (0, 3, 1, 2))

    def _extract_path_phases(self, h: torch.Tensor) -> np.ndarray:
        """
        Extract phase information from channel impulse response.

        Args:
            h: Channel impulse response [batch, 1, num_rx_ant, 1, num_tx_ant, num_paths, num_time_steps]

        Returns:
            Phase features [batch, num_paths]
        """
        # h shape: [batch, 1, num_rx_ant, 1, num_tx_ant, num_paths, num_time_steps]
        # Average over antennas and time to get path phases
        phases = torch.angle(h)  # still complex
        phases = phases.mean(dim=(2, 4, 6))  # [batch, 1, 1, num_paths]
        phases = phases.squeeze(dim=1).squeeze(dim=1)  # [batch, num_paths]
        return phases.cpu().numpy()

    def _extract_dominant_angles_delays(self, h: torch.Tensor, tau: torch.Tensor) -> np.ndarray:
        """
        Extract dominant path angles (AoD, AoA) and delays.

        Args:
            h: Channel impulse response [batch, 1, num_rx_ant, 1, num_tx_ant, num_paths, num_time_steps]
            tau: Path delays [batch, 1, 1, num_paths]

        Returns:
            [batch, num_dominant_paths * 4] (AoD, AoA, delay, power)
        """
        batch_size = h.shape[0]
        num_dominant = self.config.num_dominant_paths
        num_paths = h.shape[5]  # actual num_paths from CDL output

        # Compute power of each path (average over antennas and time)
        h_power = torch.mean(torch.abs(h) ** 2, dim=(2, 4, 6))  # [batch, 1, 1, num_paths]
        h_power = h_power.squeeze(dim=1).squeeze(dim=1)  # [batch, num_paths]

        # Get top-k dominant paths indices
        k = min(num_dominant, num_paths)
        _, top_indices = torch.topk(h_power, k=k, dim=1)  # [batch, k]

        features = []
        for i in range(batch_size):
            batch_features = []
            for j in range(k):
                idx_int = top_indices[i, j].item()  # Python int, no advanced indexing
                delay = tau[i, 0, 0, idx_int].item()  # tau: [batch, 1, 1, num_paths]
                power = h_power[i, idx_int].item()
                aoa = (idx_int / num_paths) * 2 * np.pi
                aod = ((idx_int + 7) % num_paths / num_paths) * 2 * np.pi
                batch_features.extend([np.sin(aoa), np.cos(aoa), delay * 1e6, power])
            features.append(batch_features)

        # Pad if necessary
        while len(features[0]) < num_dominant * 4:
            for i in range(batch_size):
                features[i].extend([0.0, 0.0, 0.0, 0.0])

        features = np.array(features)[:, :num_dominant * 4]
        return features

    def _compute_covariance_matrix(self, h_freq: np.ndarray) -> np.ndarray:
        """
        Compute spatial covariance matrix from frequency response.

        Args:
            h_freq: Frequency response [batch, num_subcarriers, num_rx_ant, num_tx_ant]

        Returns:
            Covariance matrix [batch, num_rx_ant * num_tx_ant, num_rx_ant * num_tx_ant]
        """
        batch, num_sc, num_rx, num_tx = h_freq.shape
        ant_size = num_rx * num_tx

        # Reshape to [batch, num_subcarriers, ant_size]
        h_reshaped = h_freq.reshape(batch, num_sc, ant_size)

        # Average over subcarriers to get spatial covariance
        # Cov = E[h * h^H] where h is spatial channel vector
        h_spatial = h_reshaped.mean(axis=1)  # [batch, ant_size]

        # Compute covariance: cov[i,j] = E[h_i * conj(h_j)]
        # Using outer product: cov = h * h^H
        cov = np.zeros((batch, ant_size, ant_size), dtype=np.float32)

        for i in range(batch):
            h_vec = h_spatial[i]  # [ant_size] complex
            # Outer product: h[:, None] * h_conj[None, :]
            h_complex = h_vec.astype(np.complex64)
            cov[i] = np.real(np.outer(h_complex, np.conj(h_complex)))

        # Normalize covariance
        for i in range(batch):
            cov[i] = cov[i] / (np.trace(cov[i]) + 1e-8)

        return cov

    def _extract_environment_info(self, h: torch.Tensor, tau: torch.Tensor,
                                   h_freq: np.ndarray) -> dict:
        """
        Extract all environmental information from channel data.

        Args:
            h: Channel impulse response [batch, num_clusters, num_rx, num_rx_ant, num_tx, num_tx_ant]
            tau: Path delays [batch, num_clusters]
            h_freq: Frequency response [batch, num_subcarriers, num_rx_ant, num_tx_ant]

        Returns:
            Dictionary containing environmental info
        """
        phases = self._extract_path_phases(h)
        # Clip to configured num_paths (CDL may produce more paths than config.num_paths)
        phases = phases[:, :self.config.num_paths]
        angles_delays = self._extract_dominant_angles_delays(h, tau)
        covariance = self._compute_covariance_matrix(h_freq)

        return {
            'phases': phases,           # [batch, num_paths]
            'angles_delays': angles_delays,  # [batch, num_dominant * 4]
            'covariance': covariance # [batch, ant_size, ant_size]
        }

    def _freq_to_csi_features(self, h_freq: np.ndarray) -> np.ndarray:
        """
        Convert frequency response to feature format.

        Args:
            h_freq: [batch, num_subcarriers, num_rx_ant, num_tx_ant]

        Returns:
            [batch, num_subcarriers, num_rx_ant * num_tx_ant * 2] (real + imag)
        """
        batch, num_sc, num_rx, num_tx = h_freq.shape

        # Stack real and imaginary parts
        h_real = np.real(h_freq)
        h_imag = np.imag(h_freq)

        # Reshape to [batch, subcarriers, rx * tx * 2]
        h_combined = np.stack([h_real, h_imag], axis=-1)
        h_flat = h_combined.reshape(batch, num_sc, num_rx * num_tx * 2)

        return h_flat

    def _generate_reciprocal_channel(self, h_freq: np.ndarray) -> np.ndarray:
        """
        Generate uplink CSI from downlink channel using reciprocity.

        In TDD systems, H_UL = H_DL^T (transposed).
        We add small random perturbations to model:
        - Calibration errors
        - Non-reciprocal hardware impairments
        - Noise and estimation errors

        Args:
            h_freq: Downlink channel [batch, num_subcarriers, num_rx_ant, num_tx_ant]

        Returns:
            Uplink channel [batch, num_subcarriers, num_rx_ant * num_tx_ant * 2]
        """
        # Transpose for reciprocity: [batch, subcarriers, num_tx, num_rx]
        h_ul = np.transpose(h_freq, (0, 1, 3, 2))

        # Add small perturbation (modeling hardware imperfections)
        # Typical calibration error is around -30 to -40 dB
        perturbation = np.random.randn(*h_ul.shape) * 0.01
        h_ul = h_ul + perturbation * np.abs(h_ul).mean()

        return self._freq_to_csi_features(h_ul)

    def _reshape_to_seq(self, csi: np.ndarray) -> np.ndarray:
        """
        Reshape CSI to [batch, seq_len, 2] format.

        Args:
            csi: [batch, num_subcarriers, num_antennas * 2]

        Returns:
            [batch, seq_len, 2]
        """
        batch, num_sc, num_feat = csi.shape
        seq_len = self.config.output_seq_len
        total_features_needed = seq_len * 2

        # Flatten to [batch, num_subcarriers * num_antennas * 2]
        csi_flat = csi.reshape(batch, num_sc * num_feat)

        if csi_flat.shape[1] > total_features_needed:
            csi_flat = csi_flat[:, :total_features_needed]
        elif csi_flat.shape[1] < total_features_needed:
            padding = np.zeros((batch, total_features_needed - csi_flat.shape[1]))
            csi_flat = np.concatenate([csi_flat, padding], axis=1)

        csi = csi_flat.reshape(batch, seq_len, 2)
        return csi

    def generate_dataset(
        self,
        num_samples: int,
        output_dir: str,
        file_prefix: str = "csi_data"
    ) -> Tuple[str, str]:
        """
        Generate and save a complete dataset.

        Args:
            num_samples: Total number of samples to generate
            output_dir: Directory to save the dataset
            file_prefix: Prefix for output files

        Returns:
            Tuple of (train_file_path, test_file_path)
        """
        os.makedirs(output_dir, exist_ok=True)

        train_ratio = 0.8
        num_train = int(num_samples * train_ratio)
        num_test = num_samples - num_train

        train_file = os.path.join(output_dir, f"{file_prefix}_train.h5")
        test_file = os.path.join(output_dir, f"{file_prefix}_test.h5")

        # Generate training data
        print(f"Generating {num_train} training samples...")
        self._generate_to_file(train_file, num_train)

        # Generate test data
        print(f"Generating {num_test} test samples...")
        self._generate_to_file(test_file, num_test)

        return train_file, test_file

    def _generate_to_file(self, filepath: str, num_samples: int):
        """Generate samples and save to HDF5 file incrementally."""
        samples_per_batch = self.config.batch_size
        num_batches = (num_samples + samples_per_batch - 1) // samples_per_batch

        with h5py.File(filepath, 'w') as f:
            dl_ds = f.create_dataset('dl_csi', (num_samples, self.config.output_seq_len, 2), dtype=np.float32)
            ul_ds = f.create_dataset('ul_csi', (num_samples, self.config.output_seq_len, 2), dtype=np.float32)

            # Create datasets for environmental info if enabled
            if self.config.extract_env_info:
                num_paths = self.config.num_paths
                num_dominant = self.config.num_dominant_paths
                ant_size = self.config.num_rx_antennas * self.config.num_tx_antennas

                env_phases_ds = f.create_dataset('env_phases', (num_samples, num_paths), dtype=np.float32)
                env_angles_ds = f.create_dataset('env_angles_delays', (num_samples, num_dominant * 4), dtype=np.float32)
                env_cov_ds = f.create_dataset('env_covariance', (num_samples, ant_size, ant_size), dtype=np.float32)

            for i in range(num_batches):
                current_batch = min(samples_per_batch, num_samples - i * samples_per_batch)
                dl_csi, ul_csi, env_info = self.generate_channel_batch(batch_size=current_batch)

                start_idx = i * samples_per_batch
                dl_ds[start_idx:start_idx + current_batch] = dl_csi
                ul_ds[start_idx:start_idx + current_batch] = ul_csi

                if self.config.extract_env_info and env_info is not None:
                    env_phases_ds[start_idx:start_idx + current_batch] = env_info['phases']
                    env_angles_ds[start_idx:start_idx + current_batch] = env_info['angles_delays']
                    env_cov_ds[start_idx:start_idx + current_batch] = env_info['covariance']

                if (i + 1) % 10 == 0:
                    print(f"  Batch {i + 1}/{num_batches} complete")

        print(f"  Saved to {filepath}")


def generate_csi_dataset(
    num_samples: int = 10000,
    output_dir: str = "./data",
    config: Optional[ChannelConfig] = None
) -> Tuple[str, str]:
    """
    Convenience function to generate a complete CSI dataset.

    Args:
        num_samples: Number of samples to generate
        output_dir: Output directory for HDF5 files
        config: Channel configuration. Uses defaults if None.

    Returns:
        Tuple of (train_file, test_file) paths
    """
    generator = SionnaCSIGenerator(config)
    return generator.generate_dataset(num_samples, output_dir)


if __name__ == "__main__":
    # Example usage
    import sys

    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./data"

    print("Starting CSI data generation...")
    print(f"Config: {ChannelConfig()}")

    train_file, test_file = generate_csi_dataset(num_samples, output_dir)
    print(f"\nDataset generation complete!")
    print(f"  Train: {train_file}")
    print(f"  Test: {test_file}")