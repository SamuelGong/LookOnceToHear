#!/usr/bin/env python3

import os
import scaper
import tempfile

def test_scaper_jams():
    """Test scaper with a single JAMS file to debug the issue"""
    
    # Test with the first JAMS file
    jams_file = "data/MixLibriSpeech/jams/test-clean/00000000/mixture.jams"
    fg_path = "data/MixLibriSpeech/librispeech_scaper_fmt/test-clean"
    bg_path = "data/MixLibriSpeech/wham_noise"
    
    print(f"Testing JAMS file: {jams_file}")
    print(f"Foreground path: {fg_path}")
    print(f"Background path: {bg_path}")
    
    try:
        # Try to generate from JAMS
        audio, jams, ann_list, event_audio_list = scaper.generate_from_jams(
            jams_file, fg_path=fg_path, bg_path=bg_path)
        print("Successfully generated audio from JAMS")
        print(f"Audio shape: {audio.shape}")
        print(f"Number of events: {len(event_audio_list)}")
        
    except Exception as e:
        print(f"Error generating from JAMS: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_scaper_jams() 