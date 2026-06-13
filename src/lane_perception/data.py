"""KITTI sequence acquisition: download, extract, and enumerate frames."""

import urllib.request
import zipfile

from . import config


def download(url, dest):
    """Idempotent download to `dest` (a Path)."""
    if dest.exists():
        print(f"  already have {dest.name}")
        return

    print(f"  downloading {dest.name}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  done ({dest.stat().st_size / 1e6:.1f} MB)")


def prepare_sequence(seq_key):
    """Download + extract a KITTI sequence and return its sorted frame paths.

    `seq_key` is a key into config.SEQUENCES (e.g. 'drive_0002').
    """
    config.ensure_dirs()
    meta = config.SEQUENCES[seq_key]
    seq_name = meta["seq_name"]
    calib_day = meta["calib_day"]

    sync_zip = config.DATA_DIR / f"{seq_name}_sync.zip"
    calib_zip = config.DATA_DIR / f"{calib_day}_calib.zip"

    download(f"{config.KITTI_BASE_URL}/{seq_name}/{seq_name}_sync.zip", sync_zip)
    download(f"{config.KITTI_BASE_URL}/{calib_day}_calib.zip", calib_zip)

    for z in (sync_zip, calib_zip):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(config.DATA_DIR)

    img_dir = config.DATA_DIR / calib_day / f"{seq_name}_sync" / "image_02" / "data"
    frame_paths = sorted(img_dir.glob("*.png"))
    print(f"{seq_key}: {len(frame_paths)} frames")
    return frame_paths
