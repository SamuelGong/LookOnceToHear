#!/usr/bin/env python3

import torch
import src.utils as utils

def test_multi_worker():
    """Test the dataset with multiple workers to see if the multiprocessing issue is resolved"""
    
    # Load config
    config = utils.Params('configs/tsh.json')
    
    # Load dataset with multiple workers
    ds = utils.import_attr(config.test_dataset)(**config.test_data_args)
    dl_iter = iter(torch.utils.data.DataLoader(
        ds, batch_size=2, shuffle=False, num_workers=2))  # Multiple workers
    
    print("Testing dataset with multiple workers...")
    
    try:
        for i, (inputs, targets) in enumerate(dl_iter):
            print(f"Successfully loaded batch {i}")
            print(f"Mixture shape: {inputs['mixture'].shape}")
            print(f"Target shape: {targets['target'].shape}")
            if i >= 2:  # Just test first few batches
                break
                
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multi_worker() 