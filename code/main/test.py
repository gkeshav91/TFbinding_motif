from data import get_data
from torch.utils.data import DataLoader
from model import GenomicDataset, GenomicCNN, train_model, test_model, get_filter_pwms, get_filter_pwms2, get_filter_pwms3
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import numpy as np
from visualise import plot_filter_logos
from helper import meme_generate, clip_filters, run_tomtom_to_tsv, match_hits_to_ground_truth
import os
from memelite import tomtom
from memelite.io import read_meme
import pandas as pd



seq_length = 200
num_seq = 25000
Batch_size = 100
lr_rate = 0.0003
epochs = 100
patience = 20
window_size = 19
threshold = 0.7
num_filters = 30
save_path_model = '/workspace/projects/motif/results/best_model.pth'
save_path_conv_filters = '/workspace/projects/motif/results/conv_filters.pdf'
save_path_meme = '/workspace/projects/motif/results/meme.txt'
Train_model = True
print_filers = True




#--------------------------------------------------------


arid3 = ['MA0151.1', 'MA0601.1', 'PB0001.1']
cebpb = ['MA0466.1', 'MA0466.2']
fosl1 = ['MA0477.1']
gabpa = ['MA0062.1', 'MA0062.2']
mafk = ['MA0496.1', 'MA0496.2']
max1 = ['MA0058.1', 'MA0058.2', 'MA0058.3']
mef2a = ['MA0052.1', 'MA0052.2', 'MA0052.3']
nfyb = ['MA0502.1', 'MA0060.1', 'MA0060.2']
sp1 = ['MA0079.1', 'MA0079.2', 'MA0079.3']
srf = ['MA0083.1', 'MA0083.2', 'MA0083.3']
stat1 = ['MA0137.1', 'MA0137.2', 'MA0137.3', 'MA0660.1', 'MA0773.1']
yy1 = ['MA0095.1', 'MA0095.2']

motifs = [[''],arid3, cebpb, fosl1, gabpa, mafk, max1, mef2a, nfyb, sp1, srf, stat1, yy1]
tsv_file = '/workspace/projects/motif/results/tomtom/tomtom_results1.tsv'
ground_truth_motifs = motifs
num_filters = 30






df = pd.read_csv(tsv_file, delimiter='\t')
best_evalues = np.ones(num_filters) 
best_match_idx = np.full(num_filters, -1) # -1 means no match found

# 2. Iterate through unique filters found in the results
for name in df['Query_ID'].unique():
    if 'filter' in name:
        # Extract index (handles 'filter_1' or 'filter1')
        filter_index = int(''.join(filter(str.isdigit, name)))

        
        
        if filter_index >= num_filters: continue

        # Get all hits for this specific filter
        subdf = df[df['Query_ID'] == name]
        targets = subdf['Target_ID'].values
        evalues = subdf['E-value'].values

        if filter_index == 5:
            print(f"targets: {targets}")


        # 3. Check against each Ground Truth motif
        for k, gt_motif_variants in enumerate(ground_truth_motifs):
            if filter_index == 5:
                print(f"gt_motif_variants: {gt_motif_variants}")
            for variant_id in gt_motif_variants:
                # Is this GT variant in our Tomtom hits?
                match_mask = [variant_id in str(t) for t in targets]
                if filter_index == 5:
                    print(f"variant_id: {variant_id}")

                if any(match_mask):
                    if filter_index == 5:
                        print(f"variant_id: {variant_id}")
                    # Get the E-value for this specific match
                    idx = np.where(match_mask)[0][0]
                    current_e = evalues[idx]

                    # Update if this is the best (lowest) E-value for this filter
                    if current_e < best_evalues[filter_index]:
                        best_evalues[filter_index] = current_e
                        best_match_idx[filter_index] = k

# 4. Calculate summary stats
# match_fraction: how many filters matched at least one GT motif
matched_filters = np.where(best_match_idx != -1)[0]
match_fraction = len(matched_filters) / float(num_filters)

# min_evalue: strongest hit found for each of the ground truth motifs
num_gt = len(ground_truth_motifs)
min_gt_evalues = np.ones(num_gt)
for i in range(num_gt):
    hits = best_evalues[best_match_idx == i]
    if len(hits) > 0:
        min_gt_evalues[i] = np.min(hits)


print(f"match_fraction: {match_fraction}")

#--------------------------------------------------------
