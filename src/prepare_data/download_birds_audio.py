import os
import requests
import pandas as pd

# Xeno-canto API endpoint
def download_recordings(query, save_path):
    page = 1
    missing_files = []  # List to store IDs without recording files

    while True:
        # Get data from Xeno-canto API
        url = f"{base_url}?query={query}&page={page}&key={key}"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to retrieve data from Xeno-canto API. {url} returned status code {response.status_code}")
            break
        
        data = response.json()
        recordings = data.get("recordings", [])
        
        # If there are no recordings, break the loop
        if not recordings:
            print("All recordings have been downloaded.")
            break

        # Download each recording
        for recording in recordings:
            file_url = recording.get('file')  
            if not file_url:  
                print(f"Recording with ID {recording.get('id')} does not have a file URL. Saving ID.")
                missing_files.append(recording.get('id'))  
                continue
            
            file_name = os.path.join(save_path, f"{recording['id']}.mp3")
            
            # Download file MP3
            print(f"Download file: {file_url}")
            try:
                with requests.get(file_url, stream=True) as file_response:
                    if file_response.status_code == 200:
                        with open(file_name, "wb") as file:
                            for chunk in file_response.iter_content(chunk_size=1024):
                                file.write(chunk)
                    else:
                        print(f"Fail to download file: {file_url}")
            except requests.exceptions.RequestException as e:
                print(f"Error while downloading {file_url}: {e}")

        # Check if there are more pages to fetch
        if page >= data.get("numPages", 0):
            break
        page += 1

    # Save ID without recording to a CSV file
    if missing_files:
        missing_files_path = os.path.join(save_path, "id_without_recording.csv")
        pd.DataFrame({"id": missing_files}).to_csv(missing_files_path, index=False)
        print(f"ID without recording saved to {missing_files_path}")


if __name__ == "__main__":
    # Xeno-canto API key (replace with your actual key)
    key="PUT_YOUR_API_KEY_HERE"
    base_url = "https://www.xeno-canto.org/api/3/recordings"

    # Indonesian birds query and download
    indonesian_birds_query = "cnt:Indonesia+grp:birds"
    save_path = r"./data/indonesian_birds"
    os.makedirs(save_path, exist_ok=True)
    download_recordings(indonesian_birds_query, save_path)
  
    # Malaysian birds query and download
    query = "cnt:Malaysia+grp:birds"
    save_path = r"./data/malaysian_birds" 
    os.makedirs(save_path, exist_ok=True)
    download_recordings(query, save_path)


