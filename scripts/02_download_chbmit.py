"""
02_download_chbmit.py
---------------------
Download the CHB-MIT recordings we need from PhysioNet.

Patient chb01 has 42 one-hour recordings, but only SEVEN of them contain
a seizure. Downloading all 42 would cost ~1.7 GB and add nothing but more
non-seizure background, which we already have far too much of. So we grab:

    - the summary file (the seizure annotations)
    - the 7 recordings that actually contain seizures
    - chb01_01.edf, one clean recording with no seizure at all

That last one is deliberate: having a recording with zero seizures lets us
measure how many false alarms the detector raises on a completely normal
hour of EEG, which is the number a clinician would actually care about.

Total download: roughly 340 MB.

Files are saved under:
    data/raw/chbmit/chb01/

Run from the project root:
    python scripts/02_download_chbmit.py
"""

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

# Base URL for the CHB-MIT Scalp EEG Database on PhysioNet
BASE_URL = "https://physionet.org/files/chbmit/1.0.0"

# ---------------------------------------------------------------------
# Which files to fetch
# ---------------------------------------------------------------------
#
# The 7 chb01 recordings containing seizures, per chb01-summary.txt:
#     chb01_03  seizure 2996-3036 s   (40 s)
#     chb01_04  seizure 1467-1494 s   (27 s)
#     chb01_15  seizure 1732-1772 s   (40 s)
#     chb01_16  seizure 1015-1066 s   (51 s)
#     chb01_18  seizure 1720-1810 s   (90 s)
#     chb01_21  seizure  327-420  s   (93 s)
#     chb01_26  seizure 1862-1963 s  (101 s)
#                              total  442 s of seizure
#
# Add more files here later if you want to extend the dataset. Anything
# you add is picked up automatically by scripts 03 and 04.

SEIZURE_FILES = [
    "chb01/chb01_03.edf",
    "chb01/chb01_04.edf",
    "chb01/chb01_15.edf",
    "chb01/chb01_16.edf",
    "chb01/chb01_18.edf",
    "chb01/chb01_21.edf",
    "chb01/chb01_26.edf",
]

# Recordings with no seizure at all, used to measure the false-alarm rate.
CLEAN_FILES = [
    "chb01/chb01_01.edf",
]

FILES = ["chb01/chb01-summary.txt"] + SEIZURE_FILES + CLEAN_FILES

# Local folder where raw EEG files will be saved
RAW_DIR = Path("data/raw/chbmit")


def main():
    print("=" * 60)
    print("02_download_chbmit.py -- fetching CHB-MIT recordings")
    print("=" * 60)
    print(f"{len(FILES)} files to fetch (~340 MB total).")
    print("This can take several minutes on a slow connection.\n")

    downloaded = 0
    skipped = 0
    failed = []

    for i, relative_path in enumerate(FILES, start=1):
        url = f"{BASE_URL}/{relative_path}"
        destination = RAW_DIR / relative_path

        # Create parent folders if they do not exist yet
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Skip anything already on disk, so re-running is cheap and safe.
        if destination.exists() and destination.stat().st_size > 0:
            size_mb = destination.stat().st_size / 1e6
            print(f"[{i}/{len(FILES)}] already have {destination.name} ({size_mb:.1f} MB)")
            skipped += 1
            continue

        print(f"[{i}/{len(FILES)}] downloading {destination.name} ...", end=" ", flush=True)
        try:
            # Download to a temporary name first. If the connection drops
            # halfway, we do not leave a truncated file that later looks
            # complete to the "already have" check above.
            temp_path = destination.with_suffix(destination.suffix + ".part")
            urlretrieve(url, temp_path)
            temp_path.replace(destination)

            size_mb = destination.stat().st_size / 1e6
            print(f"done ({size_mb:.1f} MB)")
            downloaded += 1

        except (HTTPError, URLError, OSError) as exc:
            print(f"FAILED ({exc})")
            failed.append(relative_path)
            # Clean up any partial file so the next run retries properly.
            temp_path = destination.with_suffix(destination.suffix + ".part")
            if temp_path.exists():
                temp_path.unlink()

    # --- report -------------------------------------------------------
    print()
    print(f"Downloaded : {downloaded}")
    print(f"Already had: {skipped}")

    if failed:
        print(f"Failed     : {len(failed)}")
        for name in failed:
            print(f"   - {name}")
        print("\nRe-run this script to retry the failed files.")
    else:
        print("\nAll files present. Next: python scripts/03_parse_summary.py")


if __name__ == "__main__":
    main()