import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
if not hasattr(torch, 'concat'):
    torch.concat = torch.cat
import matplotlib.pyplot as plt
import csv
import sys
import original 
from original import (
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

sys.modules['__main__'].Encode = Encode
sys.modules['__main__'].AttentionPool = AttentionPool
sys.modules['__main__'].stem_block = stem_block
sys.modules['__main__'].resnet_module = resnet_module
sys.modules['__main__'].Resnet_block = Resnet_block
sys.modules['__main__'].MHA_block = MHA_block
sys.modules['__main__'].feed_forward = feed_forward
sys.modules['__main__'].resnet_module_1 = resnet_module_1
sys.modules['__main__'].Resnet_block_1 = Resnet_block_1

class PredictionDataset(Dataset):
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

def parse_fasta(fasta_file):
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
                seq_id = line[1:] 
                current_seq = ""
            else:
                current_seq += line
        if current_seq:
            sequences.append((seq_id, current_seq))
    return sequences

def extract_raw_logits(model, sequences, device, batch_size=32):
    pred_dataset = PredictionDataset(sequences, max_len=64*2**7)
    pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_logits = []
    
    print(f"Running primary forward pass on {len(sequences)} sequences to extract logits...")
    with torch.no_grad():
        for batch_idx, (seq_ids, seq_batch) in enumerate(pred_loader):
            seq_batch = seq_batch.to(device)
            raw_logits = model(seq_batch, norm="layer_norm", mask=None)
            all_logits.append(raw_logits.cpu())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Extracted {(batch_idx + 1) * batch_size}/{len(sequences)}...")
                
    return torch.cat(all_logits, dim=0)

def plot_logit_diagnostics(logits_tensor, label_names=None):
    print("\nGenerating Logit Diagnostics...")
    logits = logits_tensor.numpy()
    n_classes = logits.shape[1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(logits.flatten(), bins=100, alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', label='logit=0 (sigmoid=0.5)')
    ax.set_xlabel("Raw logit value")
    ax.set_ylabel("Count")
    ax.set_title(f"Pooled logit distribution (N={logits.shape[0]} seq x {n_classes} classes)")
    ax.legend()
    plt.tight_layout()
    fig.savefig("RNALocateV3.0/Data/Validation/Temprature/logit_distribution_pooled.png", dpi=300)
    plt.close(fig)
    
    fig, axes = plt.subplots(n_classes, 1, figsize=(8, 2*n_classes), sharex=True)
    for i, ax in enumerate(axes):
        ax.hist(logits[:, i], bins=50, alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        name = label_names[i] if label_names else f"class {i}"
        ax.set_title(name, fontsize=9)
    axes[-1].set_xlabel("Raw logit")
    plt.tight_layout()
    fig.savefig("RNALocateV3.0/Data/Validation/Temprature/logit_distribution_per_class.png", dpi=300)
    plt.close(fig)
    
    print("\n" + "="*60)
    print("Logit summary statistics per class:")
    print("="*60)
    for i in range(n_classes):
        col = logits[:, i]
        name = label_names[i] if label_names else f"class {i}"
        print(f"  {name:35s}  mean={col.mean():+.2f}  std={col.std():.2f}  min={col.min():+.2f}  max={col.max():+.2f}")
    
    print("\n Saved 'logit_distribution_pooled.png'")
    print(" Saved 'logit_distribution_per_class.png'")

def plot_logit_correlations(logits_tensor, sequences, label_names):
    print("\nExecuting Bulk Feature Correlation Check per class...")
    logits = logits_tensor.numpy()
    
    lengths = np.array([len(seq) for seq_id, seq in sequences])
    gc_contents = np.array([(seq.lower().count('g') + seq.lower().count('c')) / max(1, len(seq)) for seq_id, seq in sequences])
    
    print("\n" + "="*60)
    print("Logit Correlation Analysis (Pearson r):")
    print("="*60)
    print(f"  {'Class Name':35s} | {'Length r':>10s} | {'GC r':>10s}")
    print("-" * 62)
    
    for i in range(logits.shape[1]):
        col = logits[:, i]
        r_len = np.corrcoef(col, lengths)[0, 1]
        r_gc = np.corrcoef(col, gc_contents)[0, 1]
        name = label_names[i]
        
        print(f"  {name:35s} | {r_len:10.4f} | {r_gc:10.4f}")
        
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.scatter(lengths, col, alpha=0.3, s=5, color='darkblue')
        ax1.set_xlabel("Sequence Length (Nucleotides)")
        ax1.set_ylabel(f"Raw Logit ({name})")
        ax1.set_title(f"{name}\nLogit vs Sequence Length (r = {r_len:.3f})")
        ax1.grid(True, alpha=0.3)
        
        ax2.scatter(gc_contents, col, alpha=0.3, s=5, color='darkred')
        ax2.set_xlabel("GC Content (Ratio)")
        ax2.set_ylabel(f"Raw Logit ({name})")
        ax2.set_title(f"{name}\nLogit vs GC Content (r = {r_gc:.3f})")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(f"RNALocateV3.0/Data/Validation/Temprature/logit_corr_{name}.png", dpi=300)
        plt.close(fig)
        
    print("\n Saved 11 class-specific correlation figures.")

def plot_temperature_sweep(logits_tensor, t_values, threshold=0.75):
    print(f"\nExecuting mathematical sweep across {len(t_values)} temperature parameters...")
    
    prob_distributions = []
    positive_counts = []
    
    for T in t_values:
        scaled_logits = logits_tensor / T
        probs = torch.sigmoid(scaled_logits).numpy()
        
        prob_distributions.append(probs.flatten())
        total_positives = np.sum(probs > threshold)
        positive_counts.append(total_positives)

    print("Generating sweep statistical figures...")
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.boxplot(prob_distributions, positions=range(1, len(t_values) + 1), flierprops={'marker': '.', 'alpha': 0.1})
    ax1.set_xticks(range(1, len(t_values) + 1))
    ax1.set_xticklabels([f"{t:.1f}" for t in t_values])
    ax1.set_title("Probability Distribution Variance per Temperature Parameter")
    ax1.set_xlabel("Temperature (T)")
    ax1.set_ylabel("Sigmoid Probability Density")
    ax1.axhline(threshold, color='red', linestyle='--', alpha=0.5, label=f'Classification Threshold ({threshold})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.tight_layout()
    fig1.savefig("RNALocateV3.0/Data/Validation/Temprature/sweep_distribution_variance.png", dpi=300)
    plt.close(fig1)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(t_values, positive_counts, marker='o', color='darkblue', linewidth=2)
    ax2.set_title("Total Positive Predictions vs Temperature")
    ax2.set_xlabel("Temperature (T)")
    ax2.set_ylabel(f"Total Labels Exceeding {threshold} Confidence")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig("RNALocateV3.0/Data/Validation/Temprature/sweep_prediction_decay.png", dpi=300)
    plt.close(fig2)
    
    print(" Saved 'sweep_distribution_variance.png'")
    print(" Saved 'sweep_prediction_decay.png'")

def save_scaled_predictions(logits_tensor, sequences, label_names, T, threshold, output_file):
    print(f"\nWriting final Probabilities file for T={T}...")
    scaled_logits = logits_tensor / T
    probs = torch.sigmoid(scaled_logits).numpy()
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['sequence_id'] + [f'{name}_prob' for name in label_names] + ['predicted_locations']
        writer.writerow(header)
        
        for i in range(len(sequences)):
            seq_id = sequences[i][0]
            p = probs[i]
            predicted_locs = ';'.join([label_names[j] for j in range(len(label_names)) if p[j] > threshold])
            if not predicted_locs:
                predicted_locs = "None"
            row = [seq_id] + list(p) + [predicted_locs]
            writer.writerow(row)
            
    print(f" Output saved to {output_file}")

def main():
    FASTA_FILE = "RNALocateV3.0/Data/Raw/FASTA.txt"      
    MODEL_PATH = "RNALocateV3.0/Codes/Prediction/Implementation/model.pth"                
    BATCH_SIZE = 32                           
    TARGET_T = 25
    THRESHOLD = 0.75
    OUTPUT_CSV = f"RNALocateV3.0/Data/Validation/Temprature/Probabilities_T{TARGET_T}.csv"
    
    TEMPERATURE_SWEEP = [1, 5, 10, 20, 30, 50, 75]
    
    LABEL_NAMES = [
        "chromatin", "cytoplasm", "cytosol",
        "endoplasmic reticulum", "extracellular region",
        "membrane", "mitochondrion", "nucleolus",
        "nucleoplasm", "nucleus", "ribosome"
    ]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Hardware context: {device}\n")
    
    print("Loading full model object...")
    try:
        model = torch.load(MODEL_PATH, map_location=device)
        model.to(device)
        
        if isinstance(model.linear_1[-1], torch.nn.Sigmoid):
            model.linear_1 = model.linear_1[:-1]
            print(" Terminal nn.Sigmoid successfully verified and stripped.")
        else:
            raise ValueError(f"Architectural mismatch: Expected nn.Sigmoid, found {type(model.linear_1[-1])}")
            
        model.eval()
        for module in model.modules():
            if hasattr(module, 'device'):
                module.device = device
                
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return

    sequences = parse_fasta(FASTA_FILE)
    if len(sequences) == 0:
        print("ERROR: No sequences found.")
        return
        
    logits_tensor = extract_raw_logits(model, sequences, device, batch_size=BATCH_SIZE)
    
    plot_logit_diagnostics(logits_tensor, label_names=LABEL_NAMES)
    plot_logit_correlations(logits_tensor, sequences, label_names=LABEL_NAMES)
    plot_temperature_sweep(logits_tensor, TEMPERATURE_SWEEP, threshold=THRESHOLD)
    save_scaled_predictions(logits_tensor, sequences, LABEL_NAMES, T=TARGET_T, threshold=THRESHOLD, output_file=OUTPUT_CSV)

if __name__ == "__main__":
    main()