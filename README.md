
**Experiment Overview:** Representation Learning in Genomic CNNsThis repository implements the experiments described in "Representation learning of genomic sequence motifs with convolutional neural networks". The study investigates how CNN architecture—specifically max-pooling and filter size—influences whether the network learns "localist" (whole motif) or "distributed" (partial motif) representations of regulatory DNA sequences.

**The Core Hypothesis:** Spatial Information Bottlenecks
The researchers hypothesized that the first layer's representation of motifs is determined by the ability of deeper layers to assemble those features hierarchically.
a. Distributed Representations: When max-pooling is small, deeper layers can resolve the exact spatial ordering of partial features. Consequently, first-layer filters tend to capture only parts of a motif.
b. Localist Representations: When max-pooling is large relative to filter size, it creates a "spatial information bottleneck". Because deeper layers can no longer resolve the spatial arrangement of features, the first layer is "forced" to learn whole motifs for the network to remain accurate.


**Experimental Design** - Synthetic Dataset Study
To establish a ground truth, the authors generated a controlled dataset:
Sequences: 25,000 random 200 nt DNA sequences.
Labels: Multi-label binary classification for 12 unique Transcription Factors (TFs).
Motif Embedding: 1 to 5 PWM-like motifs from the JASPAR database were randomly embedded into each sequence.


