"""
Sionna-based CSI data generator for MIMO channel simulation.

This module generates realistic CSI data using Sionna's channel models,
which capture physical wireless channel characteristics like multipath,
fading, and MIMO properties.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import tensorflow as tf
import numpy as np
import h5py
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ChannelConfig:
    """Physical channel configuration for MIMO system."""
    # Carrier frequency in Hz
    carrier_frequency: float = 3.5e9

    # Number of transmit/receive antennas
    num_tx_antennas: int = 64
    num_rx_antennas: int = 16

    # OFDM parameters
    num_subcarriers: int = 128
    ofdm_symbols_per_slot: int = 14

    # Channel model parameters
    num_paths: int = 20  # Number of multipath components
    delay_spread: float = 300e-9  # RMS delay spread in seconds

    # SNR range for training data generation
    snr_db_min: float = 5.0
    snr_db_max: float = 30.0

    # Training data parameters
    batch_size: int = 32
    output_seq_len: int = 128  # Sequence length for CSI feedback


class SionnaCSIGenerator:
    """
    Generate downlink and uplink CSI pairs using Sionna channel models.

    The generator creates realistic MIMO channel frequency responses that
    capture:
    - Multipath fading
    - Spatial correlation
    - Delay spread effects
    - Frequency selectivity
    """

    def __init__(self, config: Optional[ChannelConfig] = None):
        self.config = config or ChannelConfig()
        self._check_sionna_available()
        self._setup_channel_model()

    def _check_sionna_available(self):
        """Verify Sionna is installed."""
        try:
            import sionna
            self.sionna = sionna
        except ImportError:
            raise ImportError(
                "Sionna is required for CSI generation. "
                "Install with: pip install sionna"
            )

    def _setup_channel_model(self):
        """Initialize the Sionna channel model."""
        from sionna.channel import (
            gen_single_sector_bs_ue_channels,
            AntennaArray,
            COST
        )

        # Configure transmit antenna array (Base Station)
        self.bs_array = AntennaArray(
            num_rows=self.config.num_tx_antennas // 8,
            num_cols=8,
            polarization="dual",
            pattern="tr38901",
            carrier_frequency=self.config.carrier_frequency
        )

        # Configure receive antenna array (User Equipment)
        self.ue_array = AntennaArray(
            num_rows=4,
            num_cols=4,
            polarization="dual",
            pattern="tr38901",
            carrier_frequency=self.config.carrier_frequency
        )

    def generate_channel_batch(self, batch_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a batch of downlink and uplink CSI pairs.

        Args:
            batch_size: Number of samples to generate. If None, uses config.batch_size.

        Returns:
            Tuple of (dl_csi, ul_csi), each of shape:
            - dl_csi: [batch_size, num_subcarriers, num_tx_antennas * num_rx_antennas * 2]
                      (real and imag for each subcarrier)
            - ul_csi: [batch_size, num_subcarriers, num_rx_antennas * num_tx_antennas * 2]
        """
        from sionna.channel import gen_single_sector_bs_ue_channels
        import sionna.constants as const

        batch_size = batch_size or self.config.batch_size

        # Generate channel frequency response
        # Shape: [batch_size, num_subcarriers, num_rx, num_tx]
        h_freq = self._generate_freq_response(batch_size)

        # Downlink CSI: UE receives from BS
        # Reshape to [batch, subcarriers, rx_antennas * tx_antennas * 2]
        dl_csi = self._freq_to_csi_features(h_freq)

        # For uplink CSI, we simulate a reciprocal channel
        # In TDD systems, uplink and downlink channels are related by reciprocity
        # We add slight variations to model practical impairments
        ul_csi = self._generate_reciprocal_channel(h_freq)

        # Reshape output to [batch, seq_len, 2] format for the model
        dl_csi = self._reshape_to_seq(dl_csi)  # [batch, seq_len, 2]
        ul_csi = self._reshape_to_seq(ul_csi)

        return dl_csi, ul_csi

    def _generate_freq_response(self, batch_size: int):
        """Generate frequency response using Sionna's channel model."""
        from sionna.channel import gen_single_sector_bs_ue_channels

        # Generate channels for batch_size users
        # Output shape: [batch_size, num_subcarriers, num_rx, num_tx]
        h_freq = gen_single_sector_bs_ue_channels(
            self.bs_array,
            self.ue_array,
            self.config.num_subcarriers,
            self.config.num_paths,
            batch_size=batch_size,
            rng=None
        )

        return h_freq.numpy()

    def _freq_to_csi_features(self, h_freq: np.ndarray) -> np.ndarray:
        """
        Convert frequency response to feature format.

        Args:
            h_freq: [batch, num_subcarriers, num_rx, num_tx]

        Returns:
            [batch, num_subcarriers, num_rx * num_tx * 2] (real + imag)
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
            h_freq: Downlink channel [batch, num_subcarriers, num_rx, num_tx]

        Returns:
            Uplink channel [batch, num_subcarriers, num_rx * num_tx * 2]
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
            csi: [batch, num_subcarriers * num_antennas * 2]

        Returns:
            [batch, seq_len, 2]
        """
        batch, features = csi.shape
        seq_len = self.config.output_seq_len

        # If features > seq_len * 2, we need to reduce
        # This can happen when num_subcarriers * num_antennas * 2 > seq_len
        total_features_needed = seq_len * 2

        if features > total_features_needed:
            # Reduce via averaging or truncation
            # Take the first N features (e.g., dominant paths)
            csi = csi[:, :total_features_needed]
        elif features < total_features_needed:
            # Pad with zeros
            padding = np.zeros((batch, total_features_needed - features))
            csi = np.concatenate([csi, padding], axis=1)

        # Reshape to [batch, seq_len, 2]
        csi = csi.reshape(batch, seq_len, 2)

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
        """Generate samples and save to HDF5 file."""
        samples_per_batch = self.config.batch_size
        num_batches = (num_samples + samples_per_batch - 1) // samples_per_batch

        dl_csi_list = []
        ul_csi_list = []

        for i in range(num_batches):
            current_batch = min(samples_per_batch, num_samples - i * samples_per_batch)
            dl_csi, ul_csi = self.generate_channel_batch(batch_size=current_batch)
            dl_csi_list.append(dl_csi)
            ul_csi_list.append(ul_csi)

            if (i + 1) % 10 == 0:
                print(f"  Batch {i + 1}/{num_batches} complete")

        dl_csi_all = np.concatenate(dl_csi_list, axis=0)[:num_samples]
        ul_csi_all = np.concatenate(ul_csi_list, axis=0)[:num_samples]

        with h5py.File(filepath, 'w') as f:
            f.create_dataset('dl_csi', data=dl_csi_all)
            f.create_dataset('ul_csi', data=ul_csi_all)

        print(f"  Saved to {filepath}")
        print(f"  Shape: dl_csi={dl_csi_all.shape}, ul_csi={ul_csi_all.shape}")


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
