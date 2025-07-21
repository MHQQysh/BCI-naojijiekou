import requests
import os



def download_bci2a_files(base_url, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    print(f"Files will be saved to: {os.path.abspath(output_directory)}\n")


    subjects = range(1, 10) # Subjects 1 to 9
    session_types = ['T', 'E'] # Training (T) and Evaluation (E)
    downloaded_count = 0
    failed_downloads = []



    for sub_num in subjects:
        for session_type in session_types:
            # Format the filename (e.g., A01T.mat, A02E.mat)
            file_name = f"A{sub_num:02d}{session_type}.mat"
            full_url = os.path.join(base_url, file_name) # Using os.path.join for URL segments
            output_path = os.path.join(output_directory, file_name)

            print(f"Attempting to download: {file_name}")
            try:
                # Use stream=True for potentially large files and iterate over content
                response = requests.get(full_url, stream=True)
                response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"  --> Successfully downloaded {file_name}")
                downloaded_count += 1

            except requests.exceptions.RequestException as e:
                print(f"  --> ERROR downloading {file_name}: {e}")
                failed_downloads.append(file_name)
            except Exception as e:
                print(f"  --> An unexpected error occurred with {file_name}: {e}")
                failed_downloads.append(file_name)

    print(f"\n--- Download Summary ---")
    print(f"Total files attempted: {len(subjects) * len(session_types)}")
    print(f"Successfully downloaded: {downloaded_count}")
    if failed_downloads:
        print(f"Failed to download: {len(failed_downloads)} files.")
        print("Please check these files/URLs:")
        for f in failed_downloads:
            print(f"- {f}")
    else:
        print("All specified files downloaded successfully!")







if __name__ == "__main__":
    # --- Configuration ---
    # The base URL for the BCI2a dataset files
    # Note: Make sure the URL ends with a '/' if it represents a directory.
    BCI2A_BASE_URL = "http://bnci-horizon-2020.eu/database/data-sets/001-2014/"


    # The directory where you want to save the downloaded .mat files.
    # IMPORTANT: Change this path to your desired location!
    # Example for Linux/macOS:
    # DOWNLOAD_DIR = "/home/shihongyuan/bci/EEG-ATCNet/data/BCI2a_raw_mat/"
    # Example for Windows:
    # DOWNLOAD_DIR = "C:/Users/YourUsername/Documents/EEG_Data/BCI2a_raw_mat/"
    DOWNLOAD_DIR = os.path.join(os.getcwd(), "data") # Saves to a new folder in your current working directory



    # --- Run the download ---
    download_bci2a_files(BCI2A_BASE_URL, DOWNLOAD_DIR)

    # After successful download, remember to update your `data_path`
    # in `main_TrainValTest.py` to point to this `DOWNLOAD_DIR`.
    # E.g., data_path = "/home/shihongyuan/bci/EEG-ATCNet/data/BCI2a_raw_mat/"
    # or data_path = "C:/Users/YourUsername/Documents/EEG_Data/BCI2a_raw_mat/"