import librosa
import soundfile as sf
import numpy as np
import os
import random
import librosa.display
import matplotlib.pyplot as plt

def plot_spectrogram(y, sr, save_dir, title):
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()

    # save at that folder
    print(f"Saving spectrogram to {save_dir}/{title}.png")
    plt.savefig(save_dir + f"/{title}.png")
    plt.close()
    # plt.show()
    
if __name__ == "__main__":
    split_dir = r"./bird_mixture/test_scenario2"

    # crawl through all wav files in that folder
    for root, dirs, files in os.walk(split_dir):
        for filename in files:
            if filename.endswith(".wav"):
                filepath = os.path.join(root, filename)
                # print(f"Processing {filepath}")
                y, sr = librosa.load(filepath, sr=22050)
                plot_spectrogram(y, sr, root, filename)
