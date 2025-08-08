import os
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from torchmetrics.functional import scale_invariant_signal_noise_ratio as si_snr
from resemblyzer import VoiceEncoder, preprocess_wav

import src.utils as utils

def load_model(config, run_dir, device):
    """Load the trained model"""
    checkpoint = os.path.join(run_dir, 'best.ckpt')
    
    model = utils.import_attr(config.pl_module)(**config.pl_module_args)
    if os.path.exists(checkpoint):
        print(f"Loading {checkpoint}")
        checkpoint = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print(f'Warning: no checkpoint found in {run_dir}')
        return None
    
    model.eval()
    model.to(device)
    return model

def create_simple_mixture(target_audio, noise_audio, target_ratio=0.7):
    """Create a simple mixture of target speaker + noise"""
    # Ensure same length
    min_len = min(len(target_audio), len(noise_audio))
    target_audio = target_audio[:min_len]
    noise_audio = noise_audio[:min_len]
    
    # Normalize
    target_audio = target_audio / np.max(np.abs(target_audio))
    noise_audio = noise_audio / np.max(np.abs(noise_audio))
    
    # Mix with specified ratio
    mixture = target_ratio * target_audio + (1 - target_ratio) * noise_audio
    
    # Convert to stereo (duplicate mono to stereo)
    if len(mixture.shape) == 1:
        mixture = np.stack([mixture, mixture], axis=1)
    if len(target_audio.shape) == 1:
        target_audio = np.stack([target_audio, target_audio], axis=1)
    
    return mixture, target_audio

def get_speaker_embedding(audio_path, encoder, sr=16000):
    """Extract speaker embedding from audio file"""
    # Load audio
    audio, _ = sf.read(audio_path)
    
    # Preprocess for Resemblyzer
    audio = preprocess_wav(audio, sr)
    
    # Get embedding
    embedding = encoder.embed_utterance(audio)
    return torch.from_numpy(embedding).float()

def minimal_test():
    """Minimal test with single speaker"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    config = utils.Params('configs/tsh.json')
    
    # Load model
    model = load_model(config, 'runs/tsh', device)
    if model is None:
        print("No model found. Please ensure you have trained models in runs/tsh/")
        return
    
    # Load VoiceEncoder for embeddings
    encoder = VoiceEncoder()
    
    # Paths for minimal test
    # You'll need to replace these with actual paths to your data
    target_audio_path = "data/MixLibriSpeech/librispeech_scaper_fmt/test-clean/19/19-198-0000.flac"
    noise_audio_path = "data/MixLibriSpeech/librispeech_scaper_fmt/test-clean/1272/1272-128104-0000.flac"
    
    # Check if files exist
    if not os.path.exists(target_audio_path):
        print(f"Target audio not found: {target_audio_path}")
        print("Please update the path to an existing audio file")
        return
    
    if not os.path.exists(noise_audio_path):
        print(f"Noise audio not found: {noise_audio_path}")
        print("Please update the path to an existing audio file")
        return
    
    print("Loading audio files...")
    
    # Load audio files
    target_audio, sr = sf.read(target_audio_path)
    noise_audio, _ = sf.read(noise_audio_path)
    
    # Create mixture
    mixture, target_clean = create_simple_mixture(target_audio, noise_audio)
    
    # Get speaker embedding from target audio
    print("Extracting speaker embedding...")
    embedding = get_speaker_embedding(target_audio_path, encoder, sr)
    
    # Convert to tensors
    mixture_tensor = torch.from_numpy(mixture).float().unsqueeze(0).to(device)  # [1, T, 2]
    embedding_tensor = embedding.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 256]
    target_tensor = torch.from_numpy(target_clean).float().unsqueeze(0).to(device)  # [1, T, 2]
    
    print(f"Mixture shape: {mixture_tensor.shape}")
    print(f"Embedding shape: {embedding_tensor.shape}")
    print(f"Target shape: {target_tensor.shape}")
    
    # Run model
    print("Running model...")
    with torch.no_grad():
        output = model(mixture_tensor, embedding_tensor)
        output = output.cpu()
    
    # Calculate metrics
    output_sisnr = si_snr(output, target_tensor).item()
    input_sisnr = si_snr(mixture_tensor, target_tensor).item()
    si_snr_improvement = output_sisnr - input_sisnr
    
    print(f"\nResults:")
    print(f"Input SI-SNR: {input_sisnr:.2f} dB")
    print(f"Output SI-SNR: {output_sisnr:.2f} dB")
    print(f"SI-SNR Improvement: {si_snr_improvement:.2f} dB")
    
    # Save results
    output_dir = "minimal_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save audio files
    sf.write(f"{output_dir}/mixture.wav", mixture.squeeze(), sr)
    sf.write(f"{output_dir}/target_clean.wav", target_clean.squeeze(), sr)
    sf.write(f"{output_dir}/separated_output.wav", output.squeeze().numpy(), sr)
    
    print(f"\nAudio files saved to {output_dir}/")
    print("- mixture.wav: Original noisy mixture")
    print("- target_clean.wav: Clean target speech")
    print("- separated_output.wav: Model output (separated speech)")

if __name__ == '__main__':
    minimal_test() 