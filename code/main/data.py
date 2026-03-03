import os, sys
import numpy as np
from helper import get_jaspar_motifs, get_label, generate_model, simulate_sequence
from visualise import get_images_of_motifs
import torch
from sklearn.model_selection import train_test_split


def get_data(seq_length, num_seq  ):

    # parse JASPAR motifs
    savepath_data = '/workspace/projects/motif/data/'
    savepath_results = '/workspace/projects/motif/results/'
    file_path = os.path.join(savepath_data, 'pfm_vertebrates.txt')
    motif_set, motif_names = get_jaspar_motifs(file_path)
    core_names = ['Arid3a', 'CEBPB', 'FOSL1', 'Gabpa', 'MAFK', 'MAX', 'MEF2A', 'NFYB', 'SP1', 'SRF', 'STAT1', 'YY1']
    print_images = True

    max_labels = len(core_names)
    split_ratio = 0.2

    core_motifs = []
    sizes_motifs = []
    for name in core_names:
        index = motif_names.index(name)
        pwm = motif_set[index]
        core_motifs.append(pwm)
        reverse = pwm[:,::-1]
        core_motifs.append(reverse[::-1,:]) 
        sizes_motifs.append(pwm.shape[1])
        print(f"Processed {name} motif : {index} with pwn \n {pwm} \n of shape {pwm.shape} with reverse \n {reverse[::-1,:]} \n of shape {reverse[::-1,:].shape} \n\n" )

    print(f"Quantiles of sizes of motifs: {np.quantile(sizes_motifs, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])}")

    if print_images:
        get_images_of_motifs(core_motifs, core_names, savepath_results)


    # generate sythetic sequences as a one-hot representation
    seq_pwm = []; seq_model = []; targets = []
    for j in range(num_seq):
        signal_pwm, labels = generate_model(core_motifs, seq_length)
        seq_pwm.append(simulate_sequence(signal_pwm))
        targets.append(get_label(labels, max_labels))
        seq_model.append(signal_pwm)

    X_all = torch.from_numpy(np.array(seq_pwm)).float()
    Y_all = torch.from_numpy(np.vstack(targets)).float()
    M_all = torch.from_numpy(np.array(seq_model)).float()


    indices = np.arange(num_seq)
    train_idx, test_idx = train_test_split( indices, test_size=split_ratio, random_state=42, shuffle=True )

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    Y_train, Y_test = Y_all[train_idx], Y_all[test_idx]
    M_train, M_test = M_all[train_idx], M_all[test_idx]

    return X_train, X_test, Y_train, Y_test, M_train, M_test

####------------------------------------------------------------------------------------------------

