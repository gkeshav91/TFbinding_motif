import numpy as np
from visualise import normalize_pwm
import os
import csv
from memelite import tomtom
from memelite.io import read_meme
import pandas as pd



def get_jaspar_motifs(file_path):
    def get_motif(f):
        line = f.readline()
        name = line.strip().split()[1]
        pfm = []
        for i in range(4):
            line = f.readline()
            if len(line.split()[1]) > 1:
                pfm.append(np.asarray(np.hstack([line.split()[1][1:], line.split()[2:-1]]), dtype=float))
            else:
                pfm.append(np.asarray(line.split()[2:-1], dtype=float))
        pfm = np.vstack(pfm)
        sum_pfm = np.sum(pfm, axis=0)
        pwm = pfm/np.outer(np.ones(4), sum_pfm)
        line = f.readline()
        return name, pwm

    num_lines = sum(1 for line in open(file_path))
    num_motifs = int(num_lines/6)

    f = open(file_path)
    tf_names = []
    tf_motifs = []
    for i in range(num_motifs):
        name, pwm = get_motif(f)
        tf_names.append(name)
        tf_motifs.append(pwm)

    return tf_motifs, tf_names


def generate_model(core_motifs, seq_length):
    
    num_motif = len(core_motifs)
    cum_dist = np.cumsum([0, 0.5, 0.25, 0.17, 0.05, 0.3])
    
    # sample core motifs for each grammar
    valid_sim = False
    while not valid_sim:

        # determine number of core motifs in a given grammar model
        num_interactions = np.where(np.random.rand() > cum_dist)[0][-1]+1 #np.random.randint(min_interactions, max_interactions)

        # randomly sample motifs
        sim_motifs = np.random.randint(num_motif, size=num_interactions)
        num_sim_motifs = len(sim_motifs)
        #sim_motifs = sim_motifs[np.random.permutation(num_sim_motifs)]
        
        # verify that distances aresmaller than sequence length
        distance = 0
        for i in range(num_sim_motifs):
            distance += core_motifs[sim_motifs[i]].shape[1]
        if seq_length > distance > 0:
            valid_sim = True    

    # simulate distances between motifs + start 
    valid_dist = False
    while not valid_dist:
        remainder = seq_length - distance
        sep = np.random.uniform(0, 1, size=num_sim_motifs+1)
        sep = np.round(sep/sum(sep)*remainder).astype(int)
        if np.sum(sep) == remainder:
            valid_dist = True

    # build a PWM for each regulatory grammar
    pwm = np.ones((4,sep[0]))/4
    for i in range(num_sim_motifs):
        pwm = np.hstack([pwm, core_motifs[sim_motifs[i]], np.ones((4,sep[i+1]))/4])

    return pwm, sim_motifs


def simulate_sequence(sequence_pwm):
    cum_prob = sequence_pwm.cumsum(axis=0)
    Z = np.random.uniform(0, 1, sequence_pwm.shape[1])
    # Find the first index where Z < cum_prob
    indices = (Z < cum_prob).argmax(axis=0)
    
    one_hot = np.zeros(sequence_pwm.shape)
    one_hot[indices, np.arange(sequence_pwm.shape[1])] = 1
    return one_hot


def get_label(labels, max_labels):
    unique_groups = np.unique(np.floor(labels / 2).astype(int))    
    targets = np.zeros((1, max_labels))    
    targets[0, unique_groups] = 1
    return targets



def meme_generate(W, base_save_path, prefix='filter', factor=None, is_clipped=False):
    """
    Generic MEME generator that handles both fixed-width arrays and 
    variable-width clipped lists.
    
    W: numpy array [N, L, 4] OR list of [4, L] arrays (if clipped)
    base_save_path: directory or filename
    is_clipped: boolean to trigger path naming and list logic
    """
    
    # 1. Update save_path based on format
    if is_clipped:
        # If user passed 'model.meme', it becomes 'model_clipped.meme'
        name, ext = os.path.splitext(base_save_path)
        save_path = f"{name}_clipped{ext}"
    else:
        save_path = base_save_path

    # background frequency
    nt_freqs = [0.25 for i in range(4)]

    # open file for writing
    with open(save_path, 'w') as f:
        f.write('MEME version 4\n\n')
        f.write('ALPHABET= ACGT\n\n')
        f.write('Background letter frequencies:\n')
        f.write('A %.4f C %.4f G %.4f T %.4f \n\n' % tuple(nt_freqs))

        for j in range(len(W)):
            # 2. Extract and format the specific PWM
            pwm = W[j] # If W is list, this is the element. If W is array, this is a slice.
            
            # 3. Handle shape normalization
            # Our PyTorch code gives [Window, 4]. 
            # Clipping logic usually produces [4, Window].
            # We force it to [Window, 4] for the print loop.
            if pwm.shape[0] == 4 and pwm.shape[1] != 4:
                pwm = pwm.T
            
            # Optional normalization factor
            if factor:
                pwm = normalize_pwm(pwm, factor=factor)

            f.write('MOTIF %s%d \n' % (prefix, j))
            
            # MEME columns: pwm.shape[0] is width, pwm.shape[1] is 4
            width = pwm.shape[0]
            f.write('letter-probability matrix: alength= 4 w= %d nsites= %d \n' % (width, width))
            
            # Print the matrix (A C G T)
            for row in pwm:
                f.write('%.4f %.4f %.4f %.4f \n' % tuple(row))
            f.write('\n')

    print(f"MEME file saved to: {save_path}")



def clip_filters(W, threshold=0.5, pad=3):
    """
    Trims uninformative flanking regions from filters based on Information Content.
    
    W: numpy array or list of arrays. 
       If array, shape is assumed [num_filters, length, 4] (PyTorch style)
    """
    W_clipped = []
    
    for w in W:
        # 1. Ensure w is in [4, length] format for the entropy calculation
        # If input is [length, 4], transpose it
        is_transposed = False
        if w.shape[0] != 4 and w.shape[1] == 4:
            w = w.T
            is_transposed = True
            
        # 2. Calculate Information Content (IC)
        # IC = log2(AlphabetSize) - Entropy
        # Entropy = -sum(p * log2(p))
        filter_length = w.shape[1]
        entropy = np.log2(4) + np.sum(w * np.log2(w + 1e-7), axis=0)
        
        # 3. Find indices where Information Content > threshold
        index = np.where(entropy > threshold)[0]
        
        if index.any():
            start = max(np.min(index) - pad, 0)
            end = min(np.max(index) + pad + 1, filter_length)
            w_crop = w[:, start:end]
        else:
            w_crop = w

        # 4. Return to original orientation if it was transposed
        if is_transposed:
            W_clipped.append(w_crop.T)
        else:
            W_clipped.append(w_crop)

    return W_clipped    





def run_tomtom_to_tsv(query_meme, target_meme, output_file, thresh=0.1):
    # 1. Load the motifs
    print(f"Loading query: {query_meme}")
    query_motifs = read_meme(query_meme)
    
    print(f"Loading database: {target_meme}")
    target_motifs = read_meme(target_meme)
    
    q_names = list(query_motifs.keys())
    t_names = list(target_motifs.keys())
    
    q_pwms = list(query_motifs.values())
    t_pwms = list(target_motifs.values())

    # 2. Run the comparison
    print("Running motif comparison...")
    # Returns matrices of shape (len(q), len(t))
    p_vals, scores, offsets, overlaps, strands = tomtom(q_pwms, t_pwms)

    # 3. Write results to TSV
    num_targets = len(t_names)
    matches_found = 0

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # Standard Tomtom Headers
        writer.writerow(["Query_ID", "Target_ID", "Optimal_Offset", "p-value", "E-value", "Overlap", "Strand"])

        for i, q_name in enumerate(q_names):
            for j, t_name in enumerate(t_names):
                p_val = p_vals[i, j]
                
                if p_val <= thresh:
                    # E-value is roughly p-value * number of targets
                    e_value = p_val * num_targets 
                    
                    writer.writerow([
                        q_name, 
                        t_name, 
                        offsets[i, j], 
                        f"{p_val:.4e}", 
                        f"{e_value:.4e}", 
                        overlaps[i, j],
                        "+" if strands[i, j] == 0 else "-"
                    ])
                    matches_found += 1

    print(f"Done! Saved {matches_found} significant matches to {output_file}")




def match_hits_to_ground_truth(tsv_file, ground_truth_motifs, num_filters=64):
    """
    Compares Tomtom TSV results against ground truth motifs.
    
    tsv_file: Path to the TSV generated by run_tomtom_to_tsv
    ground_truth_motifs: List of lists (each sublist contains IDs for one GT motif)
    num_filters: Number of filters in your CNN (e.g., 64)
    """
    # 1. Load data (handling empty files if no matches were found)
    try:
        df = pd.read_csv(tsv_file, delimiter='\t')
    except pd.errors.EmptyDataError:
        print(f"No matches found in {tsv_file}")
        return np.ones(num_filters), np.zeros(num_filters), np.zeros(len(ground_truth_motifs)), 0.0

    # Initialize tracking arrays
    # We use E-value here because your run_tomtom_to_tsv generates E-values

    
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

            # 3. Check against each Ground Truth motif
            for k, gt_motif_variants in enumerate(ground_truth_motifs):
                for variant_id in gt_motif_variants:
                    # Is this GT variant in our Tomtom hits?
                    match_mask = [variant_id in str(t) for t in targets]
                    if any(match_mask):
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

    return best_evalues, best_match_idx, min_gt_evalues, match_fraction