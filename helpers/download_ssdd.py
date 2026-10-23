from pathlib import Path
from huggingface_hub import HfApi, snapshot_download


def download_ssdd():
    PROJECT = Path.cwd()
    DOWNLOAD_DIR = PROJECT / "SSDD_data"

    repo_id = "dronefreak/SSDD"

    # Capture the exact repository version for reproducibility.
    revision = HfApi().dataset_info(repo_id).sha

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=DOWNLOAD_DIR,
        allow_patterns=["README.md", "data/*"],
    )

    DATA_ROOT = DOWNLOAD_DIR / "data"

    print("Dataset location:", DATA_ROOT)
    print("Dataset revision:", revision)