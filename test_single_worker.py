#!/usr/bin/env python3

import torch
import src.utils as utils

def test_single_worker():
    """Test the dataset with a single worker to debug multiprocessing issues"""
    
    # Load config
    config = utils.Params('configs/tsh.json')
    
    # Load dataset with single worker
    ds = utils.import_attr(config.test_dataset)(**config.test_data_args)
    dl_iter = iter(torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=0))  # Single worker
    
    print("Testing dataset with single worker...")
    
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
    test_single_worker() 