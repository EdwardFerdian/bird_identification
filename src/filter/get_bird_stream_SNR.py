import librosa
import numpy as np
import os

def calculate_approx_snr(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    
    # 1. Compute the Root Mean Square (RMS) energy for short frames
    # Use a hop_length that captures short bird chirps (e.g., 20ms)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    
    # 2. Convert to decibels so the math is easier
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # 3. Estimate Noise Floor (10th percentile) and Signal (95th percentile)
    noise_floor = np.percentile(rms_db, 10)
    signal_peak = np.percentile(rms_db, 95)
    
    # 4. The "Conventional" SNR
    snr = signal_peak - noise_floor
    
    # 5. BONUS: Temporal Entropy (Birds are "bursty", noise is "flat")
    # A low entropy means the energy is concentrated in a few bursts (birds)
    energy_dist = rms / np.sum(rms)
    entropy = -np.sum(energy_dist * np.log2(energy_dist + 1e-10))
    
    return snr, entropy


if __name__ == "__main__":
    split_dir = r"./bird_mixture/test_scenario2"

    feature_list = []
    filenames = []
    best_files = []
    # crawl through all wav files in that folder
    for root, dirs, files in os.walk(split_dir):
        # get snr for every files and every 4 files, get the highest snr
        file_list = [f for f in files if f.endswith(".wav")]
        file_list.sort()  # Ensure consistent ordering

        for i in range(0, len(file_list), 4):
            group = file_list[i:i+4]
            max_snr = float('-inf')
            best_file = None
            for filename in group:
                filepath = os.path.join(root, filename)
                snr, entropy = calculate_approx_snr(filepath)
                if snr > max_snr:
                    max_snr = snr
                    best_file = filename

            feature_list.append([max_snr, 0])  # Use 0 for entropy since we're not tracking it per group
            filenames.append(os.path.join(root, best_file))
            best_files.append(os.path.join(root, best_file))
 
    # Convert list to a 2D array (Samples x Features)
    X = np.array(feature_list)
 
 
    with open(os.path.join(split_dir, "snr_entropy_list_best.txt"), "w") as f:
        f.write("Filename\tSNR(dB)\tEntropy\n")
        for name, (snr, entropy) in zip(filenames, feature_list):
            f.write(f"{name}\t{snr:.2f}\t{entropy:.4f}\n")
    
    # save the best files to a txt file
    with open(os.path.join(split_dir, "best_files.txt"), "w") as f:
        for name in filenames:
            f.write(f"{name}\n")
