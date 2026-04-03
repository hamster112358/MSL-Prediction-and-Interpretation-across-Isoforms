import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import csv
from einops import rearrange, reduce
import torch.nn as nn
import torch.nn.functional as F

# Import all model classes from original file

import modules 

from modules import (
    Encode, 
    AttentionPool, 
    stem_block, 
    resnet_module, 
    Resnet_block,
    MHA_block,
    feed_forward,
    resnet_module_1,
    Resnet_block_1
)

import sys
sys.modules['__main__'].Encode = modules.Encode

# New module for prediction only

class PredictionDataset(Dataset):
    """Dataset for making predictions on unlabeled sequences"""
    
    def seq2one_hot(self, seq):
        mapping = {'a': 0, 't': 1, 'u':1, 'c': 2, 'g': 3, 'o':4}
        onehot_matrix = np.vstack((np.eye(4), np.zeros(4)))
        seq_lower = [mapping.get(s.lower(), 4) for s in seq]
        return onehot_matrix[seq_lower]
    
    def __init__(self, sequences, max_len=64*2**7):
        super(PredictionDataset, self).__init__()
        self.sequences = sequences
        self.max_len = max_len

    def __getitem__(self,index):
        seq_id, seq = self.sequences[index]
        seq_result = self.seq2one_hot(seq)
        if len(seq) < self.max_len:
            len1 = (self.max_len - len(seq)) // 2
            len2 = (self.max_len - len(seq)) - len1
            seq_result = np.pad(seq_result, ((len1, len2), (0, 0)), 'constant', constant_values=(0, 0))
        elif len(seq) > self.max_len:
            half = self.max_len // 2
            seq = seq[:half] + seq[-half:]
            seq_result = self.seq2one_hot(seq)            
        return seq_id, torch.from_numpy(seq_result).type(torch.float)
    
    def __len__(self):
        return len(self.sequences)

# Prepocessing of raw fasta file data

def parse_fasta(fasta_file):
    """Parse FASTA file and extract sequences"""
    
    sequences = []
    current_seq = ""
    seq_id = ""
    
    print(f"Reading {fasta_file}...")
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append((seq_id, current_seq))
                seq_id = line[1:]  # Remove '>'
                current_seq = ""
            else:
                current_seq += line
        
        # Include the last sequence
        if current_seq:
            sequences.append((seq_id, current_seq))
    
    return sequences

def predict_localizations(model, sequences, device, batch_size=8, threshold=0.5):
    """Make predictions on sequences"""
    
    # Label names in order
    label_names = [
        "chromatin--(nucleus)",
        "cytoplasm",
        "cytosol--(cytoplasm)",
        "endoplasmic reticulum--(cytoplasm)",
        "extracellular region--(nucleus)",
        "membrane",
        "mitochondrion--(cytoplasm)",
        "nucleolus--(nucleus)",
        "nucleoplasm--(nucleus)",
        "nucleus",
        "ribosome--(cytoplasm)"
    ]
    
    # Create dataset and loader
    pred_dataset = PredictionDataset(sequences, max_len=64*2**7)
    pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    model.eval()
    
    print(f"\nProcessing {len(sequences)} sequences...")


# Get predictions
    with torch.no_grad():
        for batch_idx, (seq_ids, seq_batch) in enumerate(pred_loader):
            seq_batch = seq_batch.to(device)
    
            # 1. Create Raw Mask (Batch, 8192)
            raw_mask = (seq_batch.sum(dim=-1) != 0).float() 
            
            # Run blindly through ResNet
            x = model.stem_(seq_batch)
            x = model.Resnet_(x)
            x = rearrange(x, 'b h s -> b s h')
            
            current_len = x.shape[1] # Should be 256
            
            # Interpolate requires (Batch, Channel, Length) -> (Batch, 1, 8192)
            mask = F.interpolate(
                raw_mask.unsqueeze(1), 
                size=current_len, 
                mode='nearest'
            ) # Output is (Batch, 1, 256)
    
            # 2. Pass mask into MHA
            for mha, ff in zip(model.MHA_, model.feedward_):
                x = mha(x, mask, "layer_norm")
                x = ff(x, "layer_norm")
            
            x = rearrange(x, 'b s h -> b h s')
            x = model.crop(x, model.len_)
            x = model.Conv1d_1(x)
            x = rearrange(x, 'b s h -> b (s h)')
            
            # 3. Manual Linear Layers
            for i in range(len(model.linear_1) - 1):
                x = model.linear_1[i](x)
                
            raw_logits = x    
            # 4. Apply Temperature scaling
            # From the prediction on the website (gold sdandard) and current corresponding logits without temperature and               # bias, considering multiple outstanding classes and top genes, we can calculate the adjusting parameters
            T = 0.5
            B = 6
            probs = torch.sigmoid((raw_logits - B) / T).cpu().numpy()
            

            # Get predictions
            # outputs = model(seq_batch, norm="layer_norm", mask=None)
            # probs = outputs.cpu().numpy()
            
            # Store results
            for seq_id, prob in zip(seq_ids, probs):
                predictions.append({
                    'sequence_id': seq_id,
                    'probabilities': prob,
                    'predictions': prob > threshold
                })
            
            # Progress update
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size}/{len(sequences)} sequences...")
    
    print(f"  Done! Processed all {len(sequences)} sequences.\n")
    return predictions, label_names

def save_predictions(predictions, label_names, output_file='predictions.csv'):
    """Save predictions to CSV file with probabilities and predicted locations"""
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header: sequence_id + all probabilities + combined prediction
        header = ['sequence_id'] + [f'{name}_prob' for name in label_names] + ['predicted_locations']
        writer.writerow(header)
        
        # Data rows
        for pred in predictions:
            # Get predicted location names (above threshold)
            predicted_locs = ';'.join([
                label_names[i] for i, is_pred in enumerate(pred['predictions']) if is_pred
            ])
            if not predicted_locs:
                predicted_locs = "None"
            
            row = [pred['sequence_id']] + list(pred['probabilities']) + [predicted_locs]
            writer.writerow(row)
    
    print(f"✓ Predictions saved to {output_file}")

def print_predictions(predictions, label_names, n=5):
    """Print first n predictions in a readable format"""
    
    print(f"\n{'='*80}")
    print(f"Sample Predictions (first {min(n, len(predictions))} sequences):")
    print(f"{'='*80}\n")
    
    for i, pred in enumerate(predictions[:n], 1):
        print(f"{i}. Sequence ID: {pred['sequence_id']}")
        print("   Predicted subcellular localizations:")
        
        has_prediction = False
        for name, prob, is_pred in zip(label_names, pred['probabilities'], pred['predictions']):
            if is_pred:
                print(f"      ✓ {name:45s} (confidence: {prob:.3f})")
                has_prediction = True
        
        if not has_prediction:
            print("      (No localization above threshold)")
            # Show top 3 probabilities even if below threshold
            top_3 = sorted(enumerate(pred['probabilities']), key=lambda x: x[1], reverse=True)[:3]
            print("      Top 3 (below threshold):")
            for idx, prob in top_3:
                print(f"        - {label_names[idx]:43s} ({prob:.3f})")
        print()

def print_summary_statistics(predictions, label_names):
    """Print summary statistics of predictions"""
    
    print(f"\n{'='*80}")
    print("Prediction Summary Statistics:")
    print(f"{'='*80}\n")
    
    total_seqs = len(predictions)
    print(f"Total sequences processed: {total_seqs}")
    
    # Count predictions per location
    location_counts = {name: 0 for name in label_names}
    sequences_with_predictions = 0
    
    for pred in predictions:
        has_pred = False
        for i, is_pred in enumerate(pred['predictions']):
            if is_pred:
                location_counts[label_names[i]] += 1
                has_pred = True
        if has_pred:
            sequences_with_predictions += 1
    
    print(f"Sequences with at least one prediction: {sequences_with_predictions} ({sequences_with_predictions/total_seqs*100:.1f}%)")
    print(f"\nPredictions by subcellular location:")
    
    for name, count in sorted(location_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:45s}: {count:4d} ({count/total_seqs*100:5.1f}%)")

# Allows the GPU to optimize the convolution algorithms
torch.backends.cudnn.benchmark = True

# Main Function
def main():
    # ========== CONFIGURATION - CHANGE THESE ==========
    FASTA_FILE = "FASTA.txt"      
    MODEL_PATH = "model.pth"                
    OUTPUT_FILE = "RNALovaeV3_predictions.csv"          
    BATCH_SIZE = 32                           
    THRESHOLD = 0.5                          # Probability threshold for positive prediction
    
    # Model hyperparameters (must match training!)
    MODEL_CONFIG = {
        'resnet_dim': 64,
        'MHA_num': 6,
        'input_dim': 2048,
        'qk_dim': 64,
        'v_dim': 256,
        'q_head': 6,
        'n_kv_head': 6,
        'len_': 16,
        'max_len': 64*2**7,
        'dropout_ratio': 0
    }
    # ==================================================
    
    print("\n" + "="*80)
    print("RNA Subcellular Localization Predictor")
    print("="*80 + "\n")
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    # Parse FASTA file
    sequences = parse_fasta(FASTA_FILE)
    print(f"✓ Found {len(sequences)} sequences\n")
    
    if len(sequences) == 0:
        print("ERROR: No sequences found in FASTA file!")
        return
    
    # Load model
    print("Loading model...")
    model = Encode(
        resnet_dim=MODEL_CONFIG['resnet_dim'],
        MHA_num=MODEL_CONFIG['MHA_num'],
        input_dim=MODEL_CONFIG['input_dim'],
        qk_dim=MODEL_CONFIG['qk_dim'],
        v_dim=MODEL_CONFIG['v_dim'],
        q_head=MODEL_CONFIG['q_head'],
        n_kv_head=MODEL_CONFIG['n_kv_head'],
        len_=MODEL_CONFIG['len_'],
        max_len=MODEL_CONFIG['max_len'],
        device=device,
        dropout_ratio=MODEL_CONFIG['dropout_ratio']
    ).to(device)
    
    try:
        print("Loading full model object...")
        model = torch.load(MODEL_PATH, map_location=device)
        model.to(device)
        model.eval()
        print("✓ Model loaded and normalization reset")

        # Detect the true hyperparameters during the trainging step
        print("--- Loaded Model Hyperparameters ---")
        print(f"ResNet Dimension (resnet_dim): {model.resnet_dim}")
        print(f"Number of MHA blocks (MHA_num): {model.MHA_num}")
        print(f"Input Dimension (input_dim):    {model.input_dim}")
        print(f"QK Dim (qk_dim):         {model.qk_dim}")
        print(f"V Dim (v_dim):          {model.v_dim}")
        print(f"Q Heads (q_head):        {model.q_head}")
        print(f"KV Heads (n_kv_head):       {model.n_kv_head}")
        print(f"Sequence Length (max_len):      {model.max_len}")
        print(f"Crop Length (len_):             {model.len_}")
        print(f"Dropout Ratio (dropout_ratio):               {model.dropout_ratio}")
        print("-" * 40)
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return

    
    # Make predictions
    predictions, label_names = predict_localizations(
        model, sequences, device, 
        batch_size=BATCH_SIZE, 
        threshold=THRESHOLD
    )
    
    # Display sample predictions
    # print_predictions(predictions, label_names, n=5)
    
    # Show summary statistics
    # print_summary_statistics(predictions, label_names)

    # make the outputs look clean
    clean_labels = [name.split('--')[0] for name in label_names]
    # Save results
    print()
    save_predictions(predictions, clean_labels, OUTPUT_FILE)
    
    print(f"\n{'='*80}")
    print("✓ Prediction complete")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()