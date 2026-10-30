"""Download and extract the LIMUC dataset (Polat et al., CC BY 4.0) from Zenodo.

Images are extracted to ./dataset/limuc/patient_based_classified_images,
which is the path referenced by the split CSVs in ./dataset/limuc/5fold.
"""
import hashlib
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/5827695/files/patient_based_classified_images.zip?download=1"
MD5 = "6d99c6b4f9b9bf3e210964d16439de42"

root = Path("./dataset/limuc")
zip_path = root / "patient_based_classified_images.zip"
image_dir = root / "patient_based_classified_images"

if not zip_path.exists():
    print(f"Downloading {URL} (1.9 GB) ...")
    urllib.request.urlretrieve(URL, zip_path)

md5 = hashlib.md5()
with open(zip_path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        md5.update(chunk)
if md5.hexdigest() != MD5:
    raise RuntimeError(f"MD5 mismatch for {zip_path}; delete it and run again.")

if not image_dir.exists():
    print(f"Extracting to {image_dir} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(root)

print("Done.")
