import os
import torch
import numpy as np
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
import matplotlib.pyplot as plt

def get_speaker_embedding(audio_path, encoder, sr=16000):
    """Extract speaker embedding from audio file"""
    # Load audio
    audio, _ = sf.read(audio_path)
    
    # Preprocess for Resemblyzer
    audio = preprocess_wav(audio, sr)
    
    # Get embedding
    embedding = encoder.embed_utterance(audio)
    return torch.from_numpy(embedding).float()

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
    
    return mixture, target_audio

def demo_speaker_embeddings():
    """Demo of speaker embeddings without the trained model"""
    print("Speaker Embedding Demo")
    print("=" * 50)
    
    # Load VoiceEncoder
    encoder = VoiceEncoder()
    
    # Find some audio files to test with
    data_dir = "data/MixLibriSpeech/librispeech_scaper_fmt/test-clean"
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please ensure you have the LibriSpeech data downloaded")
        return
    
    # Find some speaker directories
    speaker_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if len(speaker_dirs) < 2:
        print("Need at least 2 speakers for demo")
        return
    
    # Get audio files from first two speakers
    speaker1_id = speaker_dirs[0]
    speaker2_id = speaker_dirs[1]
    
    speaker1_dir = os.path.join(data_dir, speaker1_id)
    speaker2_dir = os.path.join(data_dir, speaker2_id)
    
    speaker1_files = [f for f in os.listdir(speaker1_dir) if f.endswith('.flac')]
    speaker2_files = [f for f in os.listdir(speaker2_dir) if f.endswith('.flac')]
    
    if not speaker1_files or not speaker2_files:
        print("No audio files found")
        return
    
    # Use first file from each speaker
    speaker1_audio = os.path.join(speaker1_dir, speaker1_files[0])
    speaker2_audio = os.path.join(speaker2_dir, speaker2_files[0])
    
    print(f"Using audio files:")
    print(f"  Speaker 1: {speaker1_audio}")
    print(f"  Speaker 2: {speaker2_audio}")
    
    print(f"Speaker 1 ({speaker1_id}): {speaker1_files[0]}")
    print(f"Speaker 2 ({speaker2_id}): {speaker2_files[0]}")
    
    # Extract embeddings
    print("\nExtracting speaker embeddings...")
    embedding1 = get_speaker_embedding(speaker1_audio, encoder)
    embedding2 = get_speaker_embedding(speaker2_audio, encoder)
    
    print(f"Embedding 1 shape: {embedding1.shape}")
    print(f"Embedding 2 shape: {embedding2.shape}")
    
    # Calculate similarity
    similarity = torch.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()
    print(f"\nCosine similarity between speakers: {similarity:.3f}")
    
    # Create a simple mixture
    print("\nCreating audio mixture...")
    audio1, sr = sf.read(speaker1_audio)
    audio2, _ = sf.read(speaker2_audio)
    
    mixture, target_clean = create_simple_mixture(audio1, audio2, target_ratio=0.6)
    
    # Save audio files for listening
    output_dir = "embedding_demo_results"
    os.makedirs(output_dir, exist_ok=True)
    
    sf.write(f"{output_dir}/speaker1_clean.wav", audio1, sr)
    sf.write(f"{output_dir}/speaker2_clean.wav", audio2, sr)
    sf.write(f"{output_dir}/mixture.wav", mixture, sr)
    sf.write(f"{output_dir}/target_clean.wav", target_clean, sr)
    
    print(f"\nAudio files saved to {output_dir}/")
    print("- speaker1_clean.wav: Clean speech from speaker 1")
    print("- speaker2_clean.wav: Clean speech from speaker 2") 
    print("- mixture.wav: Mixed audio (speaker 1 + speaker 2)")
    print("- target_clean.wav: Clean target speech (speaker 1)")
    
    # Visualize embeddings
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(embedding1.numpy())
    plt.title(f'Speaker 1 ({speaker1_id}) Embedding')
    plt.ylabel('Embedding Value')
    
    plt.subplot(1, 3, 2)
    plt.plot(embedding2.numpy())
    plt.title(f'Speaker 2 ({speaker2_id}) Embedding')
    plt.ylabel('Embedding Value')
    
    plt.subplot(1, 3, 3)
    plt.bar(['Same Speaker', 'Different Speakers'], [1.0, similarity])
    plt.title('Embedding Similarity')
    plt.ylabel('Cosine Similarity')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/embedding_visualization.png", dpi=150, bbox_inches='tight')
    print(f"\nEmbedding visualization saved to {output_dir}/embedding_visualization.png")
    
    print(f"\nDemo Summary:")
    print(f"- Speaker embeddings are 256-dimensional vectors")
    print(f"- Similarity between different speakers: {similarity:.3f}")
    print(f"- In a real system, the model would use speaker 1's embedding")
    print(f"  to extract speaker 1's voice from the mixture")

if __name__ == '__main__':
    demo_speaker_embeddings() 