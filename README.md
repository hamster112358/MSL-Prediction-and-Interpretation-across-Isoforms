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

### Technical Analysis of the Model Performance and Architecture
1. **Calibration and Testing issues:**
   The model's sigmoid outputs exhibit a saturated bimodal distribution, with most predictions concentrated near 0 or 1. This limits the usefulness of probability thresholds for downstream filtering (a transcript with predicted probability 0.95 is not meaningfully more confident than one at 0.5). Post-hoc temperature scaling would address this by introducing a single learned scalar to soften the logit distribution before the sigmoid. We attempted this calibration but could not apply it reliably. Temperature scaling requires a held-out partition on which to fit the temperature parameter without contaminating the evaluation of generalization, and this partition is not recoverable from the available data.
   **The model is reported as having been trained and validated via 5-fold cross-validation on 559,651 entries, with no independent held-out test set described in either the publication or the supplementary materials, and no annotation of fold membership in the released database. The absence raises a broader concern. The model's reported performance reflects cross-validation metrics, which were used during model selection and are therefore optimistically biased, rather than independent test performance. The predictor's reliability cannot be independently verified, and downstream interpretations of its output inherit this uncertainty.**
   
3. **Fixed cropping with unmasked attention:**
   Each ResNet stage constructs its repeated bottleneck blocks with nn.ModuleList([self.block2 for _ in range(num-1)]), which inserts the same module reference rather than independent copies. All repeated blocks within a stage therefore share a single parameter set, so each stage contains only two unique transformations: the initial block1 and one reused block2. The encoder's effective capacity is therefore well below what the canonical ResNet (3, 4, 6, 3) layout suggests, and the features learned per stage are correspondingly limited. The shared block2 further includes BatchNorm1d, so its affine parameters (γ, β) and the buffers (running_mean, running_var) are also shared across every iteration. Running statistics accumulate over inputs originating at different network depths, and inference normalizes all iterations against the same averaged statistics, producing a train/inference mismatch beyond what standard BatchNorm incurs. The minimal correction is nn.ModuleList([copy.deepcopy(self.block2) for _ in range(num-1)]), which would require retraining.
   
5. **Fixed cropping with unmasked attention:**
   After the transformer block, the model applies a fixed-length cropping of length len_ (a hyperparameter) on each side of the post-pooling feature map. Because len_ is independent of the original input length, short sequences retain padding-contaminated edges while near-full-length sequences lose real signal. The flattened linear layer that follows is sized for this fixed scheme, coupling the classifier to a single padding configuration and preventing native support for variable-length inputs. This is compounded by an unset attention mask: the MHA block accepts a mask argument but receives None by default, so attention is computed over all positions including zero-padded ones. Real positions integrate information from padding during the global context stage, and the crop only removes those edges post-hoc. A principled fix would pass a length-determined mask through MHA and replace the crop-and-flatten head with masked global pooling followed by an MLP, decoupling the classifier from the padding scheme.
   
6. **Length dependence:**
   Predictions correlate substantially with sequence length on real data (r = 0.27–0.60 across classes). Running 200 uniform-random pseudo-sequences through the same model produced length-correlation r values of 0.27 - 0.50 for nine of eleven classes. Comparing real and pseudo correlations per class isolates the biological component (Δr = real_r − pseudo_r). Δr being strictly larger than 0 suggests length carries information in a biological sense. ER shows the largest biology-driven length signal (Δr = 0.33), followed by chromatin (0.24) and nucleoplasm (0.17). For cytosol, nucleus, and extracellular region, Δr is below 0.10, indicating that real-data length correlations are largely architectural artifacts rather than biological signals.
   The cosine similarity heatmap of predicted-probability vectors provides converging evidence about ER specifically. ER shows low pairwise similarity to all other compartments (≤ 0.22), making it the most isolated class in the model's output space. Combined with its low abundance in the training database (cite original paper), as well as its high biology-driven length signal, this isolation suggests the model has primarily learned length as the discriminative feature for ER. Whether this reflects genuine biology (whether long mRNAs really are enriched at ER) is yet to be explored. ER's apparent distinctiveness should therefore be interpreted with caution.

7. **Hierarchical consistency:**
   Some biological inductive bias impose constraints on what predictions are technically plausible across compartments. Three relationships were tested and indicate that the model has not learned complete anatomical rules. No correction for these inconsistencies was applied, therefore candidate transcripts whose predictions rely on the predictions inherit the underlying violations.
The nucleus contains chromatin, nucleolus, and nucleoplasm. This suggests that only when transcripts are in the nucleus can they be localized to other clusters, and that nucleus prediction should be at least supported by one of the sub-locations simultaneously. The former feature was captured as shown in scatter plots. However, of 3,805 nucleus-positive transcripts (probability > 0.5), 1,657 (43.5%) lack any sub-nuclear support. While it can be partly explained by the lack of nuclear-envelope specification, it indicates the model does not completely enforce the parent-child relationship between nucleus and its sub-compartments.
   The cytoplasm also anatomically contains cytosol, ribosomes, and mitochondria. Pairwise scatter plots reveal a substantial population of transcripts with high predicted probability for one of these sub-compartments and near-zero for cytoplasm itself. The relationship between cytoplasm and cytosol is entirely inversed, which violates anatomical containment more directly. A transcript cannot be cytosol-localized without being cytoplasmic, because cytosol is by definition the cytoplasmic fluid. The pattern indicates the model treats cytoplasm and its sub-compartments as competing labels rather than nested ones.
   Membrane and extracellular regions are not strictly hierarchical but share regulatory zipcodes (need citations), as some of mRNA sent to distal cellular regions might be carried by the same molecular machinery, while extracellular-region-located ones may require more transferring steps. The pairwise scatter for these two compartments shows a containment pattern, where the membrane is a sub-localization.

