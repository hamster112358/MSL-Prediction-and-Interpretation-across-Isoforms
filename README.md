# Implmentation of RNALocate v3.0 predictor to analyze mRNA subcellular localization across splice isoforms

## Tools
    Anaconda
    python 3.8.20
    MEME Suite

## Introduction
    Neural crest cells (NCCs) are pluripotent cells in early development often described as the “fourth germ layer” due to the plethora of cell types they give rise to. The developmental trajectories of neural crest lineages have been studied with single-cell RNA sequencing and genetic lineage tracing; however, the role of mRNA isoforms remains unknown despite pervasive alternative splicing during mammalian development. As is well-known that alternatively spliced mRNA may change post-transcriptional regulation, examples include subcellular localization (SCL) of mRNA, which is essential during development as it enables spatiotemporally restricted protein synthesis and establishes cellular polarity. There are several mechanisms of SCL, notably the non-coding sequence can act as a zip code specifying target location for the transcript by binding to transport proteins. Hence we ask whether alterative splicing regulates subcellular localization during differentiation of NCCs. Here, we utilize RNALocate v3.0, as it was trained on comprehensive databases, to predict the subcellular localization of mRNA from mouse neural crest and SCP lineages across developmental stages (E8.5–E13.5).

## Folders
### Model/{Prediction, Interpretation}
    Prediction: main python programs and the model
    Interpretation: downstream analysis
### Data/{ISM, MEME, MHA, MAIN, RAW, TOMTOM}
    RAW: FASTS file of isoforms, aquired from Ensembl BioMart
    MAIN: outputs of the model
    MHA: isoform candidates filtered
    ISM: saliency maps and clusters
    MEME: potential motifs in each buckets
    TOMTOM: good alignments filtered

## DOE
    1. The saturated probabilities were a bit of concern. Casting masks on padding regions in MHA layers and temprature scaling before activation were both tested to improve the accuracy. However we did not keep these fine-tuning  for strategiesinterpretability.
    2. Due to a large data size, it was expected to be extremely slow to run the In-silico Mutagenesis. We extracted the attention matrix of each transcripts in the resolution of 32 bases. Those with a high entropy across the whole length and a high peak of attention score (statistical significant), as well as having both key and query of the peak inside the real sequence (biological significant) were retained. 
    3. Different strategies were tested when running ISM, including varied sizes of window to be mutated, and different methods (mutation; shuffling on different length). At last, we chose to shuffle on a single base in length of 20. For robustness, each position was repeated 3 times, with a 5 bases shift forward in 2nd rep and a 5 bases shift backward in 3th rep. This would in theory capture the signal of motifs on edges. Compared with the original prediction, the Jensen-Shannon Divergence (JSD) of the distributions and the changes in individual probabilities were calculated, the max, mean and std were recorded to create the saliency map. The saliency score was calculated as 0.7 max + 0.3 mean, and the confidance was described as the max over std.
    4. Transcripts were binned into different clusters and buckets, based the original prediction. The same transcript could have been found in all the clusters and either in positive or negative buckets (if without selecting). During binning, saliency map for each cluster was extracted for each transcript, the peaks that pass both the absolute and relative thresholds were cropped with a span of 50 bases and stayed in the buckets. Chosen regions were masked so that we could iteratively find the next peak.
    5. The motifs found to decrease the likelihood of a certain cluster to be predicted were marked as promoters, increase as suppressors. At least 94.9%/99.7% of promotors/suppressors were found in the positive/negative buckets, which was biologically intuitive.
    6. Self-alignment was done within each cluster and bucket. Though the same motif could appear multiple times on the same transcript, we assume each sequence cropped would only have at most one of each motif. Due to the inherently greedy bias of the algorithm, we ran the program twice, where we specified shorter motifs as in a range of 4-10, and longer as in 10-30. 
    7. Buckets were compared agianst three databases in search of RBP, miR and other types of RNA. All species were included given evolutionary conservation. We discarded the motifs that are too common, or that have too low abundance. We also filtered out the matches that run in the reverse direction, or that could not pass at least two of three thresholds, where p-value was 5e-4, q-value was 0.1 and E-value was 0.05.
     