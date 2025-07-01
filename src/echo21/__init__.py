import requests
from pathlib import Path

DATA_FILENAME = "f_coll_idm.npz"
url = "https://raw.githubusercontent.com/shikharmittal04/echo21/master/.echo21/f_coll_idm.npz"

def get_data_path():
    """Ensure the required data file exists and return its path."""
    data_dir = Path.home() / ".echo21"
    data_dir.mkdir(parents=True, exist_ok=True)

    data_file = data_dir / DATA_FILENAME
    if not data_file.exists():
        #print(f"Downloading {DATA_FILENAME} to {data_file}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(data_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return str(data_file)


DATA_PATH = get_data_path()

# Expose DATA_PATH to all modules that import echo21
__all__ = ["DATA_PATH"]