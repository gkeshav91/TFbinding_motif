from data import get_data
from torch.utils.data import DataLoader
from model import GenomicDataset, GenomicCNN, train_model, test_model, get_filter_pwms3
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import numpy as np
from visualise import plot_filter_logos
from helper import meme_generate, clip_filters, run_tomtom_to_tsv, match_hits_to_ground_truth
import os

seq_length = 200
num_seq = 25000
Batch_size = 100
lr_rate = 0.0003
epochs = 100
patience = 20
threshold = 0.7

window_size = 19
num_filters = 30

save_path_model = '/workspace/projects/motif/results/best_model.pth'
save_path_conv_filters = '/workspace/projects/motif/results/conv_filters.pdf'
save_path_meme = '/workspace/projects/motif/results/meme.txt'
Train_model = True
print_filers = True


X_train, X_test, Y_train, Y_test, M_train, M_test = get_data(seq_length, num_seq)
train_dataset = GenomicDataset(X_train, Y_train)
test_dataset = GenomicDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=Batch_size, shuffle=False)

print(f"Train size: {X_train.shape} | Test size: {X_test.shape} | Y_train shape: {Y_train.shape} | Y_test shape: {Y_test.shape} | M_train shape: {M_train.shape} | M_test shape: {M_test.shape} ")

#--------------------------------------------------------

num_labels = Y_train.shape[1]

model = GenomicCNN(num_labels, seq_length)
if Train_model:
    train_model(model, train_loader, test_loader, num_epochs=epochs, patience=patience, learning_rate=lr_rate, save_path=save_path_model)
model.load_state_dict(torch.load(save_path_model))

predictions, true_labels = test_model(model, test_loader)

roc_auc = roc_auc_score(true_labels, predictions, average='macro')
pr_auc = average_precision_score(true_labels, predictions, average='macro')

print(f"Test ROC-AUC: {roc_auc:.4f}")
print(f"Test PR-AUC: {pr_auc:.4f}")

example_prediction_index = 2

print(f"Example prediction: \n {predictions[example_prediction_index]} \n and true label: {true_labels[example_prediction_index]} \n" )

#--------------------------------------------------------


pwms = get_filter_pwms3(model, test_loader, threshold=threshold, window=window_size)
if print_filers:
    plot_filter_logos(pwms, save_path=save_path_conv_filters)


meme_generate(pwms, base_save_path=save_path_meme, prefix='filter', factor=None, is_clipped=False)
clipped_pwms = clip_filters(pwms, threshold=threshold, pad=3)
meme_generate(clipped_pwms, base_save_path=save_path_meme, prefix='filter', factor=None, is_clipped=True)

#--------------------------------------------------------

db_path = '/workspace/projects/motif/data/JASPAR_CORE_2016_vertebrates.meme'
input_meme1 = save_path_meme
input_meme2 = save_path_meme.replace('.txt', '_clipped.txt')
out_dir = '/workspace/projects/motif/results/tomtom'

run_tomtom_to_tsv( query_meme=input_meme1, target_meme=db_path, output_file=os.path.join(out_dir, 'tomtom_results1.tsv'), thresh=0.1 )
run_tomtom_to_tsv( query_meme=input_meme2, target_meme=db_path, output_file=os.path.join(out_dir, 'tomtom_results2.tsv'), thresh=0.1 )

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
motifnames = [ '','arid3', 'cebpb', 'fosl1', 'gabpa', 'mafk', 'max', 'mef2a', 'nfyb', 'sp1', 'srf', 'stat1', 'yy1']

tomtom_results1 = os.path.join(out_dir, 'tomtom_results1.tsv')
tomtom_results2 = os.path.join(out_dir, 'tomtom_results2.tsv')
evals, matches, min_evals, fraction = match_hits_to_ground_truth(tomtom_results1, motifs, num_filters)
evals2, matches2, min_evals2, fraction2 = match_hits_to_ground_truth(tomtom_results2, motifs, num_filters)

print(f"fraction1: {fraction}")
print(f"fraction2: {fraction2}")


#--------------------------------------------------------
