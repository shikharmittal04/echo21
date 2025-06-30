import urllib.request
from pathlib import Path

DATA_FILENAME = "f_coll_idm.npz"
DATA_URL = "https://github.com/shikharmittal04/echo21/blob/master/.echo21/f_coll_idm.npz"

def get_data_path():
    """Ensure the required data file exists and return its path."""
    data_dir = Path.home() / ".echo21"
    data_dir.mkdir(parents=True, exist_ok=True)

    data_file = data_dir / DATA_FILENAME
    if not data_file.exists():
        print(f"Downloading {DATA_FILENAME} to {data_file}")
        urllib.request.urlretrieve(DATA_URL, data_file)
        print("Download complete.")
    return str(data_file)


DATA_PATH = get_data_path()
print(DATA_PATH)
# Expose DATA_PATH to all modules that import echo21
__all__ = ["DATA_PATH"]