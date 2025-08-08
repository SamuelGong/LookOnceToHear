# Minimal Examples for Target Speech Hearing

This directory contains simplified examples to test the target speech hearing system with a single speaker, avoiding the complexity of batch processing.

## Quick Start

### 1. Speaker Embedding Demo (No Model Required)

This demo shows how speaker embeddings work without requiring the trained model:

```bash
python simple_embedding_demo.py
```

**What it does:**
- Loads audio from two different speakers
- Extracts speaker embeddings using Resemblyzer
- Creates a simple audio mixture
- Visualizes the embeddings and calculates similarity
- Saves audio files for listening

**Output:**
- `embedding_demo_results/` directory with:
  - `speaker1_clean.wav` - Clean speech from first speaker
  - `speaker2_clean.wav` - Clean speech from second speaker  
  - `mixture.wav` - Mixed audio (both speakers)
  - `target_clean.wav` - Clean target speech
  - `embedding_visualization.png` - Plot showing embeddings

### 2. Minimal Model Test (Requires Trained Model)

This test uses the actual trained model to separate speech:

```bash
python minimal_test.py
```

**What it does:**
- Loads the trained TSH model from `runs/tsh/best.ckpt`
- Creates a mixture of two speakers
- Uses speaker embedding to identify target speaker
- Runs the model to separate the target speaker's voice
- Calculates quality metrics (SI-SNR improvement)

**Output:**
- `minimal_test_results/` directory with:
  - `mixture.wav` - Original noisy mixture
  - `target_clean.wav` - Clean target speech
  - `separated_output.wav` - Model output (separated speech)
- Example standard output:
    ```
    (ts-hear) samuel@Paprikas-MacBook-Pro-2 LookOnceToHear % python minimal_test.py
    Using device: cpu
    /Users/samuel/anaconda3/envs/ts-hear/lib/python3.9/site-packages/espnet2/enh/decoder/stft_decoder.py:58: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      @torch.cuda.amp.autocast(enabled=False)
    /Users/samuel/anaconda3/envs/ts-hear/lib/python3.9/site-packages/espnet2/enh/encoder/stft_encoder.py:79: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      @torch.cuda.amp.autocast(enabled=False)
    Loading runs/tsh/best.ckpt
    Loaded the voice encoder model on cpu in 0.01 seconds.
    Loading audio files...
    Using speaker 1089 with:
      - Mixture audio: data/MixLibriSpeech/librispeech_scaper_fmt/test-clean/1089/1089-134686-0000.flac
        - Embedding audio: data/MixLibriSpeech/librispeech_scaper_fmt/test-clean/1089/1089-134686-0005.flac
        - Noise audio: data/MixLibriSpeech/librispeech_scaper_fmt/test-clean/121/121-121726-0000.flac
    Extracting speaker embedding from different utterance...
    Mixture shape: torch.Size([1, 2, 135360])
    Embedding shape: torch.Size([1, 1, 256])
    Target shape: torch.Size([1, 135360, 2])
    Running model...
    Output shape: torch.Size([1, 2, 135360])
    
    Results:
    Input SI-SNR: 0.00 dB
    Output SI-SNR: -7.90 dB
    SI-SNR Improvement: -7.90 dB
    
    Audio files saved to minimal_test_results/
    - mixture.wav: Original noisy mixture
      - target_clean.wav: Clean target speech
      - separated_output.wav: Model output (separated speech)
      - embedding_audio.wav: Audio used for speaker embedding extraction
    
    Running postprocessing (ASR + summarization + KV store)...
    4ae52c22-e0c8-4897-9f74-3a0866ed6258
    Loading Whisper model: base
    Whisper model loaded on device: cpu
    Transcribing: minimal_test_results/separated_output.wav
    /Users/samuel/anaconda3/envs/ts-hear/lib/python3.9/site-packages/whisper/transcribe.py:132: UserWarning: FP16 is not supported on CPU; using FP32 instead
      warnings.warn("FP16 is not supported on CPU; using FP32 instead")
    Transcript:  He hoped there would be stew for dinner, turnips and carrots and bruise potatoes and fat mutton pieces to be ladled out. Thick pepper.
    Stored KV record with key (embedding fingerprint): 144c30931caede34c4b1188404e62e8534575261b92542afe459d3c54afa351a
    Transcript (truncated): He hoped there would be stew for dinner, turnips and carrots and bruise potatoes and fat mutton pieces to be ladled out. Thick pepper.
    Summary:
     He anticipates a hearty dinner of stew. The stew is expected to contain turnips, carrots, slightly bruised potatoes, and chunks of fatty mutton. He also looks forward to the stew having a thick, peppery flavor, imagining the ladling - out of this comforting dish.
    
    Retrieved record by embedding fingerprint.
    Retrieved metadata: {'embedding_audio': 'data/MixLibriSpeech/librispeech_scaper_fmt/test-clean/1089/1089-134686-0005.flac', 'mixture_audio': 'data/MixLibriSpeech/librispeech_scaper_fmt/test-clean/1089/1089-134686-0000.flac', 'model_run_dir': 'runs/tsh', 'noise_audio': 'data/MixLibriSpeech/librispeech_scaper_fmt/test-clean/121/121-121726-0000.flac', 'sample_rate': 16000, 'source': 'minimal_test', 'speaker_id': '1089', 'test_type': 'different_utterances_same_speaker'}
    ```

## Understanding the Process

### Speaker Embeddings
- **What**: 256-dimensional vectors that uniquely identify a speaker
- **How**: Generated using Resemblyzer's VoiceEncoder
- **Purpose**: Acts as a "fingerprint" to identify the target speaker

### Audio Separation Process
1. **Input**: Audio mixture + Target speaker's embedding
2. **Model**: Neural network trained to separate speech
3. **Output**: Clean speech of just the target speaker

### Key Metrics
- **SI-SNR**: Scale-Invariant Signal-to-Noise Ratio (higher = better)
- **SI-SNR Improvement**: How much the model improves audio quality
- **Cosine Similarity**: How well embeddings match (0-1, higher = more similar)

## Customizing the Examples

### Change Speakers
Edit the file paths in the scripts:
```python
target_audio_path = "data/MixLibriSpeech/librispeech_scaper_fmt/test-clean/1089/1089-134686-0000.flac"
noise_audio_path = "data/MixLibriSpeech/librispeech_scaper_fmt/test-clean/121/121-123142-0000.flac"
```

### Adjust Mixture Ratio
Change the `target_ratio` parameter:
```python
mixture, target_clean = create_simple_mixture(audio1, audio2, target_ratio=0.6)
```

## Troubleshooting

### Missing Model
If you get "No model found" error:
- Ensure you have trained models in `runs/tsh/best.ckpt`
- Or download pre-trained models from the project's Google Drive

### Missing Audio Files
If audio files are not found:
- Check that the LibriSpeech data is downloaded to `data/MixLibriSpeech/`
- Update file paths to match your actual data structure

### Dependencies
Make sure you have all required packages:
```bash
pip install -r requirements.txt
```

## Expected Results

### Embedding Demo
- Different speakers should have low cosine similarity (~0.3-0.5)
- Same speaker should have high similarity (~0.9+)
- Audio files should be clearly distinguishable

### Model Test
- SI-SNR improvement should be positive (model improves quality)
- Separated output should sound clearer than mixture
- Target speech should be more prominent in output

## Next Steps

Once you understand the minimal examples:
1. Try with different speakers
2. Experiment with different mixture ratios
3. Run the full test script: `python -m src.ts_hear_test`
4. Explore the dataset structure in `data/MixLibriSpeech/` 