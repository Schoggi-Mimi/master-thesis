"""
Run Inference with SIIM Model

Script to run inference on ISIC 2018 dataset and save predictions.
"""

import os
import sys
import argparse
import torch
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.isic_dataset import create_dataloader
from src.models.siim_inference import SIIMModelWrapper
from src.utils.config import load_all_configs, ensure_directories


def main():
    parser = argparse.ArgumentParser(description='Run inference with SIIM model')
    parser.add_argument('--config-dir', type=str, default='configs',
                       help='Path to configs directory')
    parser.add_argument('--data-split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Data split to run inference on')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for inference')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Output CSV file for predictions')
    args = parser.parse_args()
    
    # Load configurations
    print("Loading configurations...")
    configs = load_all_configs(args.config_dir)
    paths_config = configs['paths']
    model_config = configs['model']
    
    # Ensure output directories exist
    ensure_directories(paths_config)
    
    # Load model
    print("Loading SIIM model...")
    model_path = paths_config['external']['siim']['model_path']
    model_wrapper = SIIMModelWrapper(
        model_path=model_path,
        device=model_config['inference']['device']
    )
    
    # Load data
    print(f"Loading {args.data_split} data...")
    csv_file = paths_config['data']['isic2018'][f'{args.data_split}_split']
    img_dir = paths_config['data']['isic2018']['images']
    
    dataloader = create_dataloader(
        csv_file=csv_file,
        img_dir=img_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=model_config['inference']['num_workers'],
        input_size=model_config['siim_model']['input_size'],
        augment=False,
        class_names=paths_config['classes']
    )
    
    # Run inference
    print(f"Running inference on {args.data_split} split...")
    results = {
        'image_name': [],
        'predicted_class': [],
        'predicted_class_name': [],
        'confidence': []
    }
    
    # Add probability columns for each class
    for class_name in paths_config['classes']:
        results[f'prob_{class_name}'] = []
    
    model_wrapper.model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image']
            image_names = batch['image_name']
            
            # Get predictions
            preds, probs = model_wrapper.predict(images)
            
            # Store results
            for i in range(len(images)):
                pred_class = preds[i].item()
                pred_class_name = paths_config['classes'][pred_class]
                confidence = probs[i, pred_class].item()
                
                results['image_name'].append(image_names[i])
                results['predicted_class'].append(pred_class)
                results['predicted_class_name'].append(pred_class_name)
                results['confidence'].append(confidence)
                
                # Store probabilities for all classes
                for j, class_name in enumerate(paths_config['classes']):
                    results[f'prob_{class_name}'].append(probs[i, j].item())
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*80)
    print(f"INFERENCE RESULTS ({args.data_split} split)")
    print("="*80)
    print(f"Total images: {len(results_df)}")
    print("\nPrediction distribution:")
    print(results_df['predicted_class_name'].value_counts())
    print(f"\nAverage confidence: {results_df['confidence'].mean():.4f}")
    
    # Save results
    if args.output_file:
        output_path = args.output_file
    else:
        output_dir = paths_config['outputs']['root']
        output_path = os.path.join(output_dir, f"predictions_{args.data_split}.csv")
    
    results_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()
