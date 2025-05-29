import os
import requests

# Create folders if they don't exist
os.makedirs("data/with_mask", exist_ok=True)
os.makedirs("data/without_mask", exist_ok=True)

# GitHub base raw URL
# https://raw.githubusercontent.com
BASE_URL = (
    "https://raw.githubusercontent.com/prajnasb/observations/master/experiements/data"
)

# Example filenames – add more or automate if needed
with_mask_files = [
    f"augmented_image_{i}.jpg" for i in range(1, 481)
]  # Adjust as per actual repo
without_mask_files = [f"{i}.jpg" for i in range(1, 480)]


def download_images(category, file_list):
    for file_name in file_list:
        url = f"{BASE_URL}/{category}/{file_name}"
        save_path = f"data/{category}/{file_name}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded: {save_path}")
            else:
                print(f"Failed to download: {url}")
        except Exception as e:
            print(f"Error for {file_name}: {e}")


# Download both sets
download_images("with_mask", with_mask_files)
# download_images("without_mask", without_mask_files)
