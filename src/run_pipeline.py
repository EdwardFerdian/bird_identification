# Pipeline execution for bird identification

from asyncio import streams, subprocess

import librosa
import numpy as np
import tensorflow as tf

from filter.get_bird_stream_SNR import calculate_approx_snr


def run_pipeline(model, audio_filepath):
    """Main pipeline to process audio and identify birds"""
    
    # Step 1: Load audio stream
    # audio_stream = load_audio_stream(audio_filepath)
    
    # Step 2: Run sound separation
    separated_sounds = run_sound_separation(audio_filepath)
    
    # Step 3: Get highest SNR from separated audio
    highest_snr_audio = select_highest_snr(separated_sounds)
    
    # Step 4: Run prediction on best quality audio
    bird_predictions = run_prediction(highest_snr_audio)
    
    # Step 5: Return results
    return bird_predictions


def load_audio_stream(audio_filepath):
    """Load audio file or stream"""
    return librosa.load(audio_filepath, sr=22050)  # Example using librosa, adjust as needed


def run_sound_separation(audio_filepath):
    """Run sound separation model on audio stream"""
    # call the mixit model
    streams = []
    
    # run bash script from python
    subprocess.run(["python", "sound-separation/models/tools/process_wav.py", "--input", audio_filepath, "--output", "path/to/separated/"])
    # python sound-separation/models/tools/process_wav.py --input path/to/raw_audio.wav --output path/to/separated/

    audio_filename = audio_filepath.split("/")[-1].split(".")[0]  # Extract filename without extension
    streams.append(f"path/to/separated/{audio_filename}_source0.wav")  
    streams.append(f"path/to/separated/{audio_filename}_source1.wav")
    streams.append(f"path/to/separated/{audio_filename}_source2.wav")
    streams.append(f"path/to/separated/{audio_filename}_source3.wav")

    return streams


def select_highest_snr(audio_list):
    """Find and return audio with highest Signal-to-Noise Ratio"""
    highest_snr = float('-inf')
    best_audio = None
    for audio in audio_list:
        # use calculate_approx_snr from get_bird_stream_SNR.py to calculate SNR
        snr, _ = calculate_approx_snr(audio)
        # keep track of highest SNR and corresponding audio
        if snr > highest_snr:
            highest_snr = snr
            best_audio = audio  
    return best_audio

def split_into_segments(audio, segment_length):
    """Split audio into segments of specified length (in seconds)"""
    # Example implementation, adjust as needed
    sr = 22050  # Sample rate
    segment_samples = segment_length * sr
    segments = [audio[i:i + segment_samples] for i in range(0, len(audio), segment_samples)]
    return segments

def convert_to_mel_spectrogram(audio):
    """Convert audio segment to Mel spectrogram for model input"""
    # Example implementation using librosa
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db  

def call_bird_identification_model(log_mel_spec):
    """Call the bird identification model and return predictions"""
    # Example implementation, replace with actual model inference code
    prediction = "predicted_bird_species"  # Placeholder
    return prediction

def vote_predictions(predictions):
    """Aggregate predictions from multiple segments (e.g., majority vote)"""
    # Example implementation, replace with actual voting logic
    from collections import Counter
    most_common_prediction = Counter(predictions).most_common(1)[0][0]
    return most_common_prediction

def run_prediction(audio):
    # split into 3 second segments and run prediction on each segment, then aggregate results
    predictions = []
    # call the bird identification model on each segment and store predictions
    # aggregate predictions (e.g., majority vote, average confidence, etc.)
    audio_segments = split_into_segments(audio, segment_length=3)  # Example function to split audio
    for segment in audio_segments:
        log_mel_spec = convert_to_mel_spectrogram(segment)  # Example function to convert to model input
        prediction = call_bird_identification_model(log_mel_spec)  
        predictions.append(prediction)  
    # aggregate predictions
    final_prediction = vote_predictions(predictions)  
    return final_prediction


if __name__ == "__main__":
    audio_filepath = "path/to/audio/file.wav"
    model_path = "path/to/bird_identification_model"

    # load model
    model = tf.keras.models.load_model(model_path)

    results = run_pipeline(model, audio_filepath)
    print(results)