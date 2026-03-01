# Bird Identification in High-Noise Environments

This repository implements a multi-stage bioacoustic framework designed to identify avian species in the dense, high-noise environments of Sarawak, Borneo. By integrating Mixture Invariant Training (MixIT) for unsupervised sound separation and a fine-tuned BirdNET-Analyzer, this pipeline "unmasks" bird vocalizations from dominant tropical biophony (insects, rain, etc.) to improve classification accuracy for endemic and endangered species.

# Setting up

This project relies on two core external repositories. Clone them into the root of this project:

### 1. MixIT (Sound Separation)

Used for isolating bird vocalizations from background noise.

```bash
git clone https://github.com/google-research/sound-separation.git
cd sound-separation
pip install -r requirements.txt
```

### 2. BirdNET-Analyzer (Classification)

Used as the core classification engine. We use v2.4.0 for consistency with the global species list.

```bash
git clone https://github.com/birdnet-team/BirdNET-Analyzer.git
cd BirdNET-Analyzer
git checkout tags/v2.4.0
pip install -r requirements.txt
```

# Data preparation

Species List: Create a species_list.txt containing the priority endemic and endangered species of Sarawak.

MixIT Pre-processing: Use MixIT to generate "clean" versions of your training data.

```bash
# Example: Separate an audio file
python sound-separation/models/tools/process_wav.py --input path/to/raw_audio.wav --output path/to/separated/
```

```bash
# Relabel 1 audio stream based on SNR calculation

```


Formatting: Organize the separated streams into folders named by species for BirdNET fine-tuning:
data/train/Pityriasis_gymnocephala/source_1.wav

# Training
Fine-tune BirdNet using the BirdNET analyzer GUI

```bash
python BirdNET-Analyzer/train.py \
    --data_dir path/to/overall_data/ \
    --model_variant "BirdNET_GLOBAL_6K_V2.4" \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001
```
*Note: This training phase exposes the model to MixIT "artifacts," allowing it to learn features that are robust to the separation process.*

# Inference
The inference pipeline runs in two stages:

- Stage 1: MixIT separates the incoming audio into 4 streams.
    Filter and select based on SNR
- Stage 2: The fine-tuned BirdNET model analyzes each stream and the original mixture.

```bash
# Run the combined pipeline script
python run_pipeline.py --input test_audio.wav --model custom_model.tflite
```

# Visualization

To validate results, we compare the original and separated signals. This is critical for expert verification.

- Spectrogram Comparison: Use the provided tools/visualize_results.py to generate side-by-side spectrograms.

```bash
python tools/visualize_results.py
```
