import mne
from pathlib import Path


def load_eeg(file_path):
    """
    Load an EEG recording from an EDF file.
    """

    file_path = Path(file_path)

    print(f"Loading EEG file: {file_path}")

    raw = mne.io.read_raw_edf(
        file_path,
        preload=False,
        verbose=False
    )

    print("\nEEG Information")
    print("-------------------------")
    print(f"Number of channels : {len(raw.ch_names)}")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Recording duration: {raw.times[-1]:.2f} seconds")
    print(f"Number of samples : {raw.n_times}")

    print("\nChannel names:")
    for channel in raw.ch_names:
        print(f" - {channel}")

    return raw


if __name__ == "__main__":

    eeg_path = Path(
        "datasets/raw/epilepsy/chb01_01.edf"
    )

    raw = load_eeg(eeg_path)