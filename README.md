# Deciphering Subcellular Localization via Alternative Splicing in Neural Crest Lineages


## Overview
This repository contains the computational pipeline used to investigate the role of alternative splicing in modulating mRNA subcellular localization (SCL) during the developmental trajectories of mouse neural crest cells and Schwann cell precursors (E8.5–E13.5). The pipeline integrates deep learning-based SCL prediction (RNALocate v3.0) with custom interpretation, motif discovery, and single-cell RNA sequencing data to isolate the gene regulatory networks underlying cellular differenation.


## Environment
The pipeline requires Python 3.8.20 and the MEME Suite. Environment management via Anaconda is strictly recommended.


## Repository Structure
```text
├── Prediction/
│   └── Implementation/
│       ├── original.py          # Network modules and architecture definitions
│       ├── model.pth            # Pre-trained RNALocate v3.0 weights
│       └── Implementation.py    # Main execution script (encoding, hyperparameters, inference)
├── Interpretation/
│   ├── Full_Trans/              # Full-length transcript analysis
│   │   ├── MHA.ipynb            # Multi-Head Attention extraction; Candidate filtering
│   │   ├── ISM.ipynb            # In-silico Mutagenesis; Saliency map generation
│   │   ├── MEME.ipynb           # De novo motif elicitation and database alignment
│   │   └── FIMO.ipynb           # Motifs mapping and idenfitication
│   └── 3_UTR/                   # 3' Untranslated Region focused analysis
│       ├── MEME_3UTR.ipynb      # UTR-specific motif elicitation
│       ├── ME_FIMO_3UTR.ipynb   # Motif-to-transcript mapping
│       ├── ATTRACT_3UTR.ipynb   # Databasz integration (ATtRACT + CISBP-RNA) and clustering
│       └── FIMO_3UTR.ipynb      # RBP-to-transcript mapping
└── README.md
```


## Design of  Experiments
### Primary: Full Transcript Interpretative Pipeline
1. **Attention Extraction:** Saturated probabilities were found to be partly addressed via masking padding regions in Multi-Head Attention (MHA) layers and applying temperature scaling prior to activation. Attention matrices were extracted at a 32-base resolution. Transcripts exhibiting high sequence-wide entropy, statistically significant attention peaks, and biological alignment (both key and query located within the actual sequence) were retained.
2. **In-Silico Mutagenesis (ISM):** A robust mutagenesis framework was executed by shuffling 20-base windows. To capture edge-motif signals, each position underwent 3 iterations (including 5-base forward and backward shifts). 
3. **Statistical Divergence and Saliency:** The Jensen-Shannon Divergence (JSD) and individual probability shifts were calculated against the baseline predictions. A saliency score was formalized as `0.7(max) + 0.3(mean)`, utilizing the maximum over the standard deviation to establish prediction confidence. 
4. **Motif Extraction:** Transcript regions passing dual absolute and relative saliency thresholds were cropped (50-base span) and binned. Motifs identified as decreasing prediction likelihood were classified as promoters, while those increasing likelihood were classified as suppressors (achieving >94.9% and >99.7% bucket accuracy, respectively).
5. **Database Mapping and Filtering:** MEME motif elicitation was partitioned by length (4-10 bp and 10-30 bp) to bypass algorithmic greedy bias. Buckets were mapped via TOMTOM against RBP, RNA and miRNA databases, strictly filtering out reverse-direction matches and enforcing thresholds (p-value < 5e-4, q-value < 0.1, E-value < 0.05).
6. **Transcript-Motif Alignment:** FIMO scanning yielded >1.6 million hits, which were aggressively filtered (q-value < 0.5) and merged by motif to generate a localized consensus. Transcripts exhibiting identical spatial distributions across all isoforms were discarded to strictly isolate splicing-driven SCL modulation, resulting in a finalized matrix of 433 differentially localized isoform-resolution genes.

### Improvement: 3' Untranslated Region (UTR) Architecture
1. **Class Binning:** The top 150 high-confidence sequences (probability > 0.5) per subcellular compartment were extracted. MEME was executed using a 0-order Markov background model to isolate compartment-specific motifs.
2. **Custom Database Construction:** To overcome the limitations of the CISBP-RNA database, ATtRACT and CISBP-RNA datasets were merged. Highly redundant Position Weight Matrices (PWMs) were grouped (Pearson correlation distance, threshold 0.25). Shorter matrices were slid against the cluster's consensus matrix to calculate minimum distance offsets prior to locking and averaging.
3. **Dual-Gate Structural Filtering:** Query motifs were required to clear a MEME E-value < 0.05 and appear in >20% of input sequences. TOMTOM mapping enforced a q-value < 0.05. FIMO scanning across the complete 3' UTR repository (p < 1e-4, q < 0.01) was aggregated using Fisher's combined probability test, keeping only highly significant transcript-level events (combined p < 1e-10). 
4. **Macro-Compartment Mapping:** Related sub-compartments were collapsed into spatial macro-compartments (decoupling experiments dhad demonstrated high dependence; chromatin/nucleolus/nucleoplasm/nucleus into Nucleus; cytosol/cytoplasm into Cytoplasm). A final spatial filter discarded genes lacking divergent macro-compartment localization.
5. **SCL and RBP Divergence Filtering:** Genes were required to have at least two different RBP profiles. Those who had all the isoforms containing exactly the same RBP profiles were discarded. Genes were also required to have a high differential SCL assignments across isoforms. Cosine similarity was calculated and the minimum was recorded for isoforms within each gene. Genes with minimum value higher than 0.6 were discarded.
6. **Single-Cell Transcriptomics Integration:** To eliminate low-expression artifacts, scRNA-seq matrices were pseudo-bulked by cell type to calculate Counts Per Million (CPM). Transcripts were required to pass an absolute floor of 1 CPM. Transcritps highly expressed in most cell types were discarded (14 out of 16).
7. **RBP-SCL Network Assignment:** Transcripts were clustered by gene via Agglomerative Clustering (cosine metric, K=2) on their RBP profiles, in a simplified assumption that there are mostly two distinctive patterns within a gene. Two patterns were compared against each other in terms of their RBP profile and SCL prediction, across each differential expressed cell type. To check if two patterns are differentially expressed, the maximum expression in specific cell type had to be higher than 50 CPM, while the average expression level 3 times higher than the medeian level in other cell types.

(Thresholds are all based on visaulizations)

### Technical Analysis of the Model Performance and Architecture
1. **Weight sharing in ResNet bottleneck stages:**
   Each ResNet stage constructs its repeated bottleneck blocks with nn.ModuleList([self.block2 for _ in range(num-1)]), which inserts the same module reference rather than independent copies. All repeated blocks within a stage therefore share a single parameter set, so each stage contains only two unique transformations: the initial block1 and one reused block2. The encoder's effective capacity is therefore well below what the canonical ResNet (3, 4, 6, 3) layout suggests, and the features learned per stage are correspondingly limited. The shared block2 further includes BatchNorm1d, so its affine parameters (γ, β) and the buffers (running_mean, running_var) are also shared across every iteration. Running statistics accumulate over inputs originating at different network depths, and inference normalizes all iterations against the same averaged statistics, producing a train/inference mismatch beyond what standard BatchNorm incurs. The minimal correction is nn.ModuleList([copy.deepcopy(self.block2) for _ in range(num-1)]), which would require retraining.
2. **Fixed cropping with unmasked attention:**
   After the transformer block, the model applies a fixed-length cropping of length len_ (a hyperparameter) on each side of the post-pooling feature map. Because len_ is independent of the original input length, short sequences retain padding-contaminated edges while near-full-length sequences lose real signal. The flattened linear layer that follows is sized for this fixed scheme, coupling the classifier to a single padding configuration and preventing native support for variable-length inputs. This is compounded by an unset attention mask: the MHA block accepts a mask argument but receives None by default, so attention is computed over all positions including zero-padded ones. Real positions integrate information from padding during the global context stage, and the crop only removes those edges post-hoc. A principled fix would pass a length-determined mask through MHA and replace the crop-and-flatten head with masked global pooling followed by an MLP, decoupling the classifier from the padding scheme
3. **Length dependence:**
   Predictions correlate substantially with sequence length on real data (r = 0.27–0.60 across classes). Running 200 uniform-random pseudo-sequences through the same model produced length-correlation r values of 0.27–0.50 for nine of eleven classes, demonstrating that much of the length signal is architectural rather than biological. Comparing real and pseudo correlations per class isolates the biological component (Δr = real_r − pseudo_r): ER shows the largest biology-driven length signal (Δr = 0.33), followed by chromatin (0.24) and nucleoplasm (0.17). For cytosol, nucleus, and extracellular region, Δr is below 0.10, indicating that real-data length correlations are largely architectural artifact rather than biological signal. This stratification is class-specific and undermines uniform interpretation of the model's outputs across compartments.

4. **Class conflation:**
   Cosine similarity between predicted-probability vectors reveals high pairwise similarity for several anatomically distant compartments. Cytosol and nucleus show similarity of 0.93 despite occupying mutually exclusive cellular regions; ribosome and mitochondrion (0.90) and cytoplasm and ribosome (0.89) show similarly high values. These pairs cannot reflect underlying biological co-occurrence at the level the model implies, because most transcripts cannot be simultaneously enriched in mutually exclusive compartments. ER and Extracellular Region are exceptions, with cosine similarity below 0.22 to all other classes. Combined with ER's high biology-driven length signal (Δr = 0.33), this isolation is most consistent with the model having learned a class-specific signature for ER, rather than treating it as noise.
5. **Hierarchical consistency:**
   Of transcripts predicted to localize to the nucleus (probability > 0.5), 43.5% lack support from any sub-nuclear compartment (chromatin, nucleolus, or nucleoplasm). The model does not enforce the anatomical containment relationship between nucleus and its sub-compartments, suggesting that either the loss function does not penalize hierarchical violations or the training labels themselves are inconsistent at this level.
