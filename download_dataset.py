import kaggle
import os

# ==================================================
# 1. Configuration
# ==================================================

# All the dataset "slugs" to download.
DATASETS_TO_DOWNLOAD = [
    # downloaded: "blastnet/channelflow-dns-re544-seq-p000-020",
    "blastnet/channelflow-dns-re544-seq-p021-041",
    "blastnet/channelflow-dns-re544-seq-p042-062",
    # "blastnet/channelflow-dns-re544-seq-p063-083",
]

# The destination folder of datasets.
DESTINATION_FOLDER = "/data1/wangteng/raw"

# ==================================================
# 2. Download & Unzip Function
# ==================================================

def download_and_unzip(dataset_slug, destination):
    """
    Downloads a single dataset, shows progress, and auto-unzips.
    """
    print(f"\n{'=' * 50}")
    print(f"Processing dataset: {dataset_slug}")
    print(f"Target directory: {destination}")
    print(f"{'=' * 50}")

    try:
        kaggle.api.dataset_download_files(
            dataset_slug, 
            path=destination, 
            unzip=True, 
            quiet=False,
        )
        print(f"\nSuccessfully downloaded and unzipped: {dataset_slug}")

    except kaggle.rest.ApiException as e:
        if "404 - Not Found" in str(e):
            print(f"Error: Dataset not found. Check the slug: {dataset_slug}")
        else:
            print(f"API Error (e.g., rate limit or permissions): {e}")
    except Exception as e:
        print(f"An unexpected error occurred for {dataset_slug}: {e}")

# ==================================================
# 3. Main Execution
# ==================================================

if __name__ == "__main__": 

    if not os.path.exists(DESTINATION_FOLDER):
        print(f"Creating directory: {DESTINATION_FOLDER}")
        os.makedirs(DESTINATION_FOLDER, exist_ok=True)

    # Loop through and download all datasets.
    print(f"\nStarting batch download of {len(DATASETS_TO_DOWNLOAD)} dataset(s)...")

    for slug in DATASETS_TO_DOWNLOAD:
        dataset_specific_folder = os.path.join(DESTINATION_FOLDER, slug.split('/')[-1])
        if not os.path.exists(dataset_specific_folder):
            os.makedirs(dataset_specific_folder, exist_ok=True)

        download_and_unzip(slug, dataset_specific_folder)

    print("\nBatch download process finished.")