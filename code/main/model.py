from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch.nn.init as init

class GenomicCNN(nn.Module):
    def __init__(self, num_labels, seq_length):
        super(GenomicCNN, self).__init__()
        
        # Layer 1
        # kernel=15, padding=7 results in 'same' padding
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=30, kernel_size=19, padding=9)
        self.bn1 = nn.BatchNorm1d(30)
        self.pool1 = nn.MaxPool1d(kernel_size=50)
        self.dropout1 = nn.Dropout(0.1)
        
        # Layer 2
        # kernel=7, padding=3 results in 'same' padding
        self.conv2 = nn.Conv1d(in_channels=30, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.1)

        # Calculate flatten size (seq_length reduced by two 4x pooling layers)
        self.flatten_size = 128 * (seq_length // 100)
        
        # Fully Connected
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_labels)

        self._initialize_weights()

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                    # He initialization for layers followed by ReLU
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten and Dense
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = self.fc2(x) 
        return x


class GenomicDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]





def train_model(model, train_loader, test_loader, num_epochs=100, patience=20, learning_rate=0.001, save_path='/workspace/projects/motif/results/best_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Loss and Optimizer
    # BCEWithLogitsLoss is best for multi-label (applies Sigmoid internally)
    criterion = nn.BCEWithLogitsLoss() # applies Sigmoid internally; output of CNN is logits, so we need to convert them to probabilities
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    
    best_test_loss = float('inf')
    counter = 0
    
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- Test Phase ---
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
        
        # --- Early Stopping Logic ---
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), save_path)
            counter = 0 # Reset patience
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    if not os.path.exists(save_path):
        torch.save(model.state_dict(), save_path)


def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            logits = model(sequences)
            probs = torch.sigmoid(logits) # convert logits to probabilities
            all_preds.append(probs.cpu().numpy())
            all_true.append(targets.cpu().numpy())
    
    predictions = np.vstack(all_preds)
    true_labels = np.vstack(all_true)
    return predictions, true_labels


def get_filter_pwms3(model, loader, threshold=0.7, window=19):
    model.eval()
    device = next(model.parameters()).device
    
    all_fmaps = []
    all_x = []

    # --- Pass 1: Save all activations and sequences ---
    print("Collecting activations from test set...")
    with torch.no_grad():
        for sequences, _ in loader:
            sequences = sequences.to(device)
            # PyTorch shape: [Batch, Filters, Length]
            fmap = torch.relu(model.conv1(sequences))
            
            # Move to CPU and transform to your expected shape: [Batch, Length, 1, Filters]
            fmap_np = fmap.permute(0, 2, 1).unsqueeze(2).cpu().numpy()
            # Transform X to your expected shape: [Batch, Length, 1, 4]
            x_np = sequences.permute(0, 2, 1).unsqueeze(2).cpu().numpy()
            
            all_fmaps.append(fmap_np)
            all_x.append(x_np)

    # Concatenate all batches
    fmap = np.concatenate(all_fmaps, axis=0)
    X = np.concatenate(all_x, axis=0)
    
    # --- Step 2: Run your exact activation_pwm logic ---
    print("Generating PWMs using your logic...")
    return activation_pwm(fmap, X, threshold=threshold, window=window)

def activation_pwm(fmap, X, threshold=0.7, window=19):
    # extract sequences with aligned activation
    window_left = int(window/2)
    window_right = window - window_left

    N, seq_length, _, num_dims = X.shape
    num_filters = fmap.shape[-1]

    W = []
    for filter_index in range(num_filters):
        # find regions above threshold (relative to global max of THIS filter)
        # fmap shape: [N, seq_length, 1, num_filters]
        filter_fmap = fmap[:, :, 0, filter_index]
        max_val = np.max(filter_fmap)
        
        if max_val == 0:
            W.append(np.full((window, num_dims), 0.25))
            continue

        x, y = np.where(filter_fmap > max_val * threshold)

        # sort score descending
        index = np.argsort(filter_fmap[x, y])[::-1]
        data_index = x[index].astype(int)
        pos_index = y[index].astype(int)

        seq_align = []
        count_matrix = []
        for i in range(len(pos_index)):
            # handle boundary conditions at start
            start_window = pos_index[i] - window_left
            if start_window < 0:
                start_buffer = np.zeros((-start_window, num_dims))
                start = 0
            else:
                start = start_window

            # handle boundary conditions at end 
            end_window = pos_index[i] + window_right
            end_remainder = end_window - seq_length
            if end_remainder > 0:
                end = seq_length
                end_buffer = np.zeros((end_remainder, num_dims))
            else:
                end = end_window

            # Get sequence [Length, 4]
            seq = X[data_index[i], start:end, 0, :]

            if start_window < 0:
                seq = np.vstack([start_buffer, seq])
            if end_remainder > 0:
                seq = np.vstack([seq, end_buffer])

            weight = filter_fmap[data_index[i], pos_index[i]]
            seq_align.append(seq * weight)
            # Count presence of nucleotides per position
            count_matrix.append(np.sum(seq, axis=1, keepdims=True) * weight)

        if len(seq_align) > 0:
            seq_align = np.array(seq_align)
            count_matrix = np.array(count_matrix)

            # normalize counts
            # result is [window, 4]
            sum_counts = np.sum(count_matrix, axis=0)
            sum_counts[sum_counts == 0] = 1 # avoid div by zero
            
            norm_seq = np.sum(seq_align, axis=0) / sum_counts
            norm_seq[np.isnan(norm_seq)] = 0
            W.append(norm_seq)
        else:
            W.append(np.full((window, num_dims), 0.25))

    # W becomes [window, 1, 4, num_filters]
    #W = np.expand_dims(np.transpose(np.array(W), [1, 2, 0]), axis=1)
    W = np.array(W)
    return W    