# Implmentation of RNALocate v3.0 predictor; Downstream interpretation combined with sc-transcriptomics

## Tools
    Anaconda
    python 3.8.20
    MEME Suite

## Introduction
Neural crest cells (NCCs) are pluripotent cells in early development often described as the “fourth germ layer” due to the plethora of cell types they give rise
to. The developmental trajectories of neural crest lineages have been studied with single-cell RNA sequencing and genetic lineage tracing; however, the role of
mRNA isoforms remains unknown despite pervasive alternative splicing during mammalian development. As is well-known that alternatively spliced mRNA may change
post-transcriptional regulation, examples include subcellular localization (SCL) of mRNA, which is essential during development as it enables spatiotemporally
restricted protein synthesis and establishes cellular polarity. There are several mechanisms of SCL, notably the non-coding sequence can act as a zip code
specifying target location for the transcript by binding to transport proteins. Hence we ask whether alterative splicing regulates subcellular localization during
differentiation of NCCs. Here, we utilize RNALocate v3.0, as it was trained on comprehensive databases, to predict the subcellular localization of mRNA from mouse
neural crest and SCP lineages across developmental stages (E8.5–E13.5).

## Folders: Codes
### Prediction
#### Implementation
    original.py: specify the modules (non-editable)
    model.pth: stores the weights (non-editable)
    Implementation.py: encode the input; set the hyperparameter; fine-tune the architecture; run the model

### Interpretation
#### Full_Trans
    MHA.ipynb: extract the attention and choose the candidates
    ISM.ipynb: run the mutagenesis on each transcrip and put them into buckets
    MEME.ipynb: elicit the motifs by MEME; align the motifs by TOMTOM
    FIMO.ipynb: compare the raw sequences aginst the identified proteins/RNA
#### 3_UTR
    MEME_3UTR.ipynb: elicit the motifs by MEME; align the motifs by TOMTOM
    ME_FIMO_3UTR.ipynb: compare the raw sequences aginst the motifs directly from MEME
    ATTRACT_3UTR.ipynb: consturct a combined database from ATTRACT and RISBPR; run TOMTOM aginst the new database
    FIMO_3UTR.ipynb: 


## DOE
### Full Transcripts
1. The saturated probabilities were a bit of concern. Casting masks on padding regions in MHA layers and temprature scaling before activation were both tested
to improve the accuracy. However we did not keep these fine-tuning strategies for interpretability.
2. Due to a large data size, it was expected to be extremely slow to run the In-silico Mutagenesis. We extracted the attention matrix of each transcripts in
the resolution of 32 bases. Those with a high entropy across the whole length and a high peak of attention score (statistical significant), as well as having
both key and query of the peak inside the real sequence (biological significant) were retained. 
3. Different strategies were tested when running ISM, including varied sizes of window to be mutated, and different methods (mutation; shuffling on different
length). At last, we chose to shuffle on a single base in length of 20. For robustness, each position was repeated 3 times, with a 5 bases shift forward in 2nd
rep and a 5 bases shift backward in 3th rep. This would in theory capture the signal of motifs on edges. Compared with the original prediction, the Jensen-Shannon
Divergence (JSD) of the distributions and the changes in individual probabilities were calculated, the max, mean and std were recorded to create the saliency map.
The saliency score was calculated as 0.7 max + 0.3 mean, and the confidance was described as the max over std.
4. Transcripts were binned into different clusters and buckets, based the original prediction. The same transcript could have been found in all the clusters
and either in positive or negative buckets (if without selecting). During binning, saliency map for each cluster was extracted for each transcript, the peaks that
pass both the absolute and relative thresholds were cropped with a span of 50 bases and stayed in the buckets. Chosen regions were masked so that we could
iteratively find the next peak.
5. The motifs found to decrease the likelihood of a certain cluster to be predicted were marked as promoters, increase as suppressors. At least 94.9%/99.7% of
promotors/suppressors were found in the positive/negative buckets, which was biologically intuitive.
6. Motif elicitation of MEME was done within each cluster and bucket. Though the same motif could appear multiple times on the same transcript, we assume each sequence
cropped would only have at most one of each motif. Due to the inherently greedy bias of the algorithm, we ran the program twice, where we specified shorter motifs
as in a range of 4-10, and longer as in 10-30. 
7. Buckets were compared agianst three databases in TOMTOM in search of RBP, miR and other types of RNA. We
discarded the motifs that are too common, or that have too low abundance. We also filtered out the matches that run in the reverse direction, or that could not
pass at least two of three thresholds, where p-value was 5e-4, q-value was 0.1 and E-value was 0.05.
8. All the transcripts were compared aginst the motifs using FIMO, and we ended up with more than 1.6 million hits. To deal with this huge data, we only kept the ones this q-value smaller than 0.5. Additionally, since both transcripts and motifs have their own clusters, we checked if they match, otherwise would be discarded.
9. With that coarse transcript-motif data, we fisrt compressed the table by motif. Genes and transcripts were put aside. If a motif was found related to the same subcellular localization, we merged the rows (there was not single motif indicating two or more localizations). Fisher scores were calculated to see if a motif was strong enough. Later me made a little adjustment —— we merged the motifs on the same transcripts, to make the statistics more rational. The confidance was described by Phred scores. Finally we got a list of 89 motifs that might be related to SCL.
10. We then temporarily ignored the motifs, only focusing on the genes and transcripts. If each transcript's localization pattern perfectly match, we thought that was not good and cut it out, as we were interested in splicing-related SCL modulation. Also we had merged the rows where same transcripts were in the same localizations, as well as having discarded _negative clusters for a clearer comparison. At last we ended up with a csv file where 433 genes have a differentially distribution of localization on an isoform resolution.


### 3' Untranslational Region
1. We cluster all sequences into different classes based on the prediction. Because 3' UTR is longer than previous 100 bases, we only extracted the top 150 high-confidence sequences (probability > 0.5) for each subcellular compartment. MEME was run using a 0-order Markov background model to elicit up to 10 motifs per compartment, with motif widths constrained between 4 and 10 bases.
2. We first utilized traditional databases in TOMTOM to annotate the motifs, followed by FIMO. However, the results are not satsifactory, partly because the size of CISBP databse is limited, and many Position Weight Matrices (PWMs) are mathematically inferred. We tried running FIMO direcely aginst MEME, as well as constructing a more comprehensive database for TOMTOM. 
3. We first merged the CISBP-RNA and ATtRACT databases (both only contain RBPs). Because ATtRACT database inherently contains highly redundant PWMs, we pooled them and performed hierarchical clustering using Pearson correlation distance with a threshold of 0.25. To manage matrices of varying lengths during averaging, we slid the shorter matrices against the longest matrix in the cluster to identify the minimum distance offset before locking them into a consensus matrix.
4. A strict dual-gate filter was applied to the motif discovery results. We only retained query motifs that possessed a MEME E-value < 0.05 and appeared in at least 20% of the input sequences. For the downstream TOMTOM mapping, we enforced a q-value threshold of < 0.05. FIMO was then used to scan the complete repository of 3' UTR sequences, applying a scanning threshold of p < 1e-4. We aggregated the sequence-level p-values using Fisher's combined probability test, keeping only highly significant transcript-level events with a combined p-value < 1e-6. Transcripts were grouped by their parent gene. If all isoforms within a gene possessed the exact same binary RBP binding profile, the entire gene was discarded as lacking structurally driven variance.
5. Single-cell RNA sequencing data was introduced to eliminate low-expression artifacts. We reversed the log1p transformation, pseudo-bulked the matrices by cell type assignment, and calculated CPM. Transcripts were required to pass an absolute floor of 15 CPM in at least one cell type. Furthermore, we calculated the cosine similarity of the relative isoform usage fractions across cell lineages. Genes exhibiting a minimum transcript cosine similarity > 0.8 were completely discarded, guaranteeing that structural variance was accompanied by genuine tissue-specific expression divergence.
6. The surviving transcripts were mapped back to their predicted localizations. A threshold of > 0.75 was used to lock localization assignments. To mitigate over-granularity in the model's labels, related sub-compartments were collapsed into macro-compartments (chromatin, nucleolus, nucleoplasm, and nucleus were merged into Nucleus; cytosol and cytoplasm into Cytoplasm, by taking the maximum). We applied a final spatial filter, strictly discarding genes whose isoforms ultimately localized to the same macro-compartment.
7. The functionally validated candidates from the ATtRACT pipeline and the computationally derived candidates from the MEME pipeline were largely overlapping, demonstrating that newly constructed database didn't cause loss of information. Two lists of candidate were concatenated, resulting in a finalized, comprehensive dataset of divergently localized isoforms.
8. To decipher the specific structural rules driving these localizations, we grouped transcripts by gene and performed Agglomerative Clustering (cosine metric, K=2) on their RBP profiles. We selected the transcript with the highest average internal similarity to represent each of the two structural patterns. By mathematically subtracting their unique motifs and unique macro-SCLs, we isolated the driving RBPs and utilized a global voting scoreboard to assign each RBP to its top three subcellular destinations.
9. A four-gate assignment logic was constructed to link these RBP-SCL networks to specific cell types. We required the dominant transcript pattern to clear an absolute threshold of 15 CPM and account for a relative dominance of at least 20% of the gene's total expression. To suppress ubiquitous targets, we introduced a housekeeping silencer: if a transcript exceeded 15 CPM in more than 8 of the 16 cell types, the pattern was silenced. Lastly, a specificity fold-change gate demanded that the pattern's expression be at least 1.2 times higher than the median of all other cell types. For final network clarity, we manually dropped AGO2_C1 due to redundant sequence affinities with AGO1_C1.