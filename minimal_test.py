import os
import torch
import torch.nn.functional as F
import numpy as np
from dotenv import load_dotenv
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
    load_dotenv(dotenv_path='.env', override=True)

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
    
    # Paths for minimal test - using different audio files from the same speaker
    # Speaker 1089 has many audio files, we'll use different ones for mixture vs embedding
    speaker_id = "1089"
    target_audio_path = f"data/MixLibriSpeech/librispeech_scaper_fmt/test-clean/{speaker_id}/{speaker_id}-134686-0000.flac"  # For mixture
    target_embedding_path = f"data/MixLibriSpeech/librispeech_scaper_fmt/test-clean/{speaker_id}/{speaker_id}-134686-0005.flac"  # For embedding
    noise_audio_path = "data/MixLibriSpeech/librispeech_scaper_fmt/test-clean/121/121-121726-0000.flac"
    
    # Check if files exist
    if not os.path.exists(target_audio_path):
        print(f"Target audio not found: {target_audio_path}")
        print("Please update the path to an existing audio file")
        return
    
    if not os.path.exists(target_embedding_path):
        print(f"Target embedding audio not found: {target_embedding_path}")
        print("Please update the path to an existing audio file")
        return
    
    if not os.path.exists(noise_audio_path):
        print(f"Noise audio not found: {noise_audio_path}")
        print("Please update the path to an existing audio file")
        return
    
    print("Loading audio files...")
    print(f"Using speaker {speaker_id} with:")
    print(f"  - Mixture audio: {target_audio_path}")
    print(f"  - Embedding audio: {target_embedding_path}")
    print(f"  - Noise audio: {noise_audio_path}")
    
    # Load audio files
    target_audio, sr = sf.read(target_audio_path)
    noise_audio, _ = sf.read(noise_audio_path)
    
    # Create mixture
    mixture, target_clean = create_simple_mixture(target_audio, noise_audio)
    
    # Get speaker embedding from a DIFFERENT audio file from the same speaker
    print("Extracting speaker embedding from different utterance...")
    embedding = get_speaker_embedding(target_embedding_path, encoder, sr)
    
    # Convert to tensors with proper format
    # Model expects [B, N, M] where N=samples, M=channels
    mixture_tensor = torch.from_numpy(mixture).float().unsqueeze(0).to(device)  # [1, T, 2]
    # Transpose to match model's expected format [B, M, N] where M=channels, N=samples
    mixture_tensor = mixture_tensor.transpose(1, 2)  # [1, 2, T] -> [B, M, N]
    embedding_tensor = embedding.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 256]
    target_tensor = torch.from_numpy(target_clean).float().unsqueeze(0).to(device)  # [1, T, 2]
    
    print(f"Mixture shape: {mixture_tensor.shape}")
    print(f"Embedding shape: {embedding_tensor.shape}")
    print(f"Target shape: {target_tensor.shape}")
    
    # Run model - let it handle chunking internally
    print("Running model...")
    with torch.no_grad():
        output = model(mixture_tensor, embedding_tensor)
        output = output.cpu()
    
    print(f"Output shape: {output.shape}")
    
    # Transpose output to match target format [B, N, M]
    output = output.transpose(1, 2)  # [B, M, N] -> [B, N, M]
    
    # Calculate metrics
    output_sisnr = si_snr(output, target_tensor).mean().item()
    input_sisnr = si_snr(mixture_tensor.transpose(1, 2), target_tensor).mean().item()
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
    separated_path = f"{output_dir}/separated_output.wav"
    sf.write(separated_path, output.squeeze().numpy(), sr)
    
    # Save the embedding audio file for reference
    embedding_audio, _ = sf.read(target_embedding_path)
    sf.write(f"{output_dir}/embedding_audio.wav", embedding_audio, sr)
    
    print(f"\nAudio files saved to {output_dir}/")
    print("- mixture.wav: Original noisy mixture")
    print("- target_clean.wav: Clean target speech")
    print("- separated_output.wav: Model output (separated speech)")
    print("- embedding_audio.wav: Audio used for speaker embedding extraction")
    
    # Postprocess: ASR -> Summarize -> KV store keyed by embedding fingerprint
    try:
        from src.postprocess import PostProcessor
    except ImportError as e:
        print("Postprocessing unavailable: missing dependency.")
        print("Install dependencies: pip install requests openai-whisper")
        return

    try:
        print("\nRunning postprocessing (ASR + summarization + KV store)...")
        pp = PostProcessor(
            db_path=os.getenv('VOICE_KV_DB', 'voice_kv.sqlite'),
            api_base=os.getenv('OPENAI_BASE_URL', None),
            api_key=os.getenv('OPENAI_API_KEY', None),
            whisper_model=os.getenv('WHISPER_MODEL', 'base'),
            chat_model=os.getenv('CHAT_MODEL', None),
            device=device,
        )
        record = pp.process_and_store(
            audio_path=separated_path,
            speaker_embedding=embedding,  # use the 1D embedding for fingerprint
            metadata={
                'source': 'minimal_test',
                'sample_rate': sr,
                'model_run_dir': 'runs/tsh',
                'speaker_id': speaker_id,
                'mixture_audio': target_audio_path,
                'embedding_audio': target_embedding_path,
                'noise_audio': noise_audio_path,
                'test_type': 'different_utterances_same_speaker',
            },
        )
        print("Stored KV record with key (embedding fingerprint):", record['key'])
        print("Transcript (truncated):", (record['transcript'][:160] + '...') if len(record['transcript']) > 160 else record['transcript'])
        print("Summary:\n", record['summary'])

        # Demonstrate retrieval by embedding
        retrieved = pp.get_by_embedding(embedding)
        if retrieved is not None:
            print("\nRetrieved record by embedding fingerprint.")
            print("Retrieved metadata:", retrieved.get('metadata', {}))
        else:
            print("\nNo record found on retrieval; storage may have failed.")
            
        # # Test with a third audio file from the same speaker to show consistency
        # print("\nTesting with a third audio file from the same speaker...")
        # test_embedding_path = f"data/MixLibriSpeech/librispeech_scaper_fmt/test-clean/{speaker_id}/{speaker_id}-134686-0002.flac"
        # if os.path.exists(test_embedding_path):
        #     test_embedding = get_speaker_embedding(test_embedding_path, encoder, sr)
        #     test_retrieved = pp.get_by_embedding(test_embedding)
        #     if test_retrieved is not None:
        #         print("✓ Successfully retrieved record using embedding from third audio file!")
        #         print("This demonstrates the speaker embedding is consistent across different utterances.")
        #     else:
        #         print("✗ Could not retrieve record using third audio file embedding.")
        #         print("This might indicate the embedding fingerprint is too sensitive to utterance differences.")
        # else:
        #     print(f"Test file not found: {test_embedding_path}")
    except Exception as e:
        print("Postprocessing failed:", str(e))
        print("Ensure OPENAI_API_KEY is set for summarization (optional) and openai-whisper is installed.")


if __name__ == '__main__':
    minimal_test() 