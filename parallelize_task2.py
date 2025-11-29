import json

# Read the notebook
with open('Barnett_Cottereau_Zhang_Assignment2.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find the cell with Task 2 training
new_code = """# Task 2 Training with Cross-Validation - PARALLELIZED
# Uses cross-validation on training data only, then evaluates on separate test set
# Parallelized: All folds run in parallel with multiprocessing
# Optimized with: Mixed precision, larger batch size, pin_memory, non_blocking transfers

import multiprocessing as mp
import time

# Prepare datasets first (as requested - same data generation process)
task2_train_dataset = Task2TrainingDataset()
task2_test_dataset = Task2TestDataset()

# First, split into training and test sets (test set is never used during training)
all_indices = list(range(len(task2_train_dataset)))
train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)

# Store test indices for later evaluation (never used during training)
task2_test_indices = test_indices

# GPU setup
if torch.cuda.is_available():
    device_type = 'cuda'
    print(f"Using device: cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    use_amp_global = True
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device_type = 'mps'
    print(f"Using device: mps (Apple Silicon GPU)")
    use_amp_global = False
else:
    device_type = 'cpu'
    print(f"Using device: cpu")
    use_amp_global = False

print(f"\\nTotal samples: {len(all_indices)}")
print(f"Training samples (for CV): {len(train_indices)}")
print(f"Test samples (held out): {len(test_indices)}")
print(f"Mixed precision training: {use_amp_global}")

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
batch_size = 128

# Prepare all fold splits first
fold_splits = []
for fold, (cv_train_idx, cv_val_idx) in enumerate(kf.split(train_indices)):
    cv_train_indices = [train_indices[i] for i in cv_train_idx]
    cv_val_indices = [train_indices[i] for i in cv_val_idx]
    fold_splits.append((fold, cv_train_indices, cv_val_indices))

print(f"\\nPrepared {len(fold_splits)} fold splits for parallel training")

# Function to train a single fold
def train_fold(fold_data):
    fold_idx, cv_train_indices, cv_val_indices = fold_data
    
    # Each process needs its own device
    if device_type == 'cuda':
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        scaler = torch.amp.GradScaler('cuda')
        use_amp = True
    elif device_type == 'mps':
        device = torch.device('mps')
        scaler = None
        use_amp = False
    else:
        device = torch.device('cpu')
        scaler = None
        use_amp = False
    
    print(f"\\n[Fold {fold_idx+1}] Starting training...")
    
    # Create datasets for this fold
    fold_train_subset = Subset(task2_train_dataset, cv_train_indices)
    fold_val_subset = Subset(task2_test_dataset, cv_val_indices)
    
    # DataLoader settings - reduced workers per process to avoid overload
    fold_train_loader = DataLoader(fold_train_subset, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True, prefetch_factor=2)
    fold_val_loader = DataLoader(fold_val_subset, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True, prefetch_factor=2)
    
    # Create model for this fold
    model_task2 = create_resnet_model(num_classes=10)
    model_task2 = model_task2.to(device)
    
    # Warmup
    _ = next(iter(fold_train_loader))
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model_task2.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # Training loop
    num_epochs = 50
    fold_best_val_acc = 0.0
    val_frequency = 2
    
    for epoch in range(num_epochs):
        model_task2.train()
        train_loss = 0.0
        for inputs, labels in tqdm(fold_train_loader, desc=f'Fold {fold_idx+1} Epoch {epoch+1}/{num_epochs}', leave=False):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model_task2(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model_task2(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        if (epoch + 1) % val_frequency == 0 or epoch == 0 or epoch == num_epochs - 1:
            model_task2.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in tqdm(fold_val_loader, desc=f'Fold {fold_idx+1} Validation', leave=False):
                    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            outputs = model_task2(inputs)
                    else:
                        outputs = model_task2(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = val_correct / val_total
            
            if val_acc > fold_best_val_acc:
                fold_best_val_acc = val_acc
        else:
            val_acc = None
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            if val_acc is not None:
                print(f'[Fold {fold_idx+1}] Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(fold_train_loader):.4f}, CV Val Acc: {val_acc:.4f}')
            else:
                print(f'[Fold {fold_idx+1}] Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(fold_train_loader):.4f}')
    
    print(f'[Fold {fold_idx+1}] Complete. Best CV validation accuracy: {fold_best_val_acc:.4f}')
    
    # Save model state for this fold
    fold_model_path = f'@generated-data/task2_fold_{fold_idx+1}_model.pth'
    torch.save(model_task2.state_dict(), fold_model_path)
    
    return (fold_idx, fold_best_val_acc, fold_model_path)

# Set multiprocessing start method (required for Windows)
if __name__ == '__main__' or True:  # Always run for notebook
    mp.set_start_method('spawn', force=True)
    
    print(f"\\n{'='*60}")
    print("Starting parallel fold training...")
    print(f"{'='*60}\\n")
    
    start_time = time.time()
    
    # Run all folds in parallel
    with mp.Pool(processes=5) as pool:
        results = pool.map(train_fold, fold_splits)
    
    elapsed_time = time.time() - start_time
    
    print(f"\\n{'='*60}")
    print("Parallel Cross-Validation Complete!")
    print(f"{'='*60}")
    
    # Process results
    results.sort(key=lambda x: x[0])  # Sort by fold index
    cv_results = [r[1] for r in results]
    best_fold_idx = max(results, key=lambda x: x[1])[0]
    best_cv_acc = max(cv_results)
    
    # Load and save best model
    best_model_path = results[best_fold_idx][2]
    best_model_state = torch.load(best_model_path)
    torch.save(best_model_state, '@generated-data/task2_best_model.pth')
    
    print(f"CV Results: {cv_results}")
    print(f"Mean CV Accuracy: {np.mean(cv_results):.4f}")
    print(f"Std CV Accuracy: {np.std(cv_results):.4f}")
    print(f"Best Fold: {best_fold_idx+1} with accuracy: {best_cv_acc:.4f}")
    print(f"Total training time: {elapsed_time/60:.2f} minutes")
    print(f"{'='*60}")

# Save test_indices for later evaluation
np.save('@generated-data/task2_test_indices.npy', test_indices)
print(f"\\nTest indices saved to @generated-data/task2_test_indices.npy")
"""

# Convert to notebook format (array of strings with newlines)
new_source = [line + '\n' for line in new_code.split('\n')]
# Remove the last newline from the last line
if new_source and new_source[-1] == '\n':
    new_source[-1] = new_source[-1].rstrip('\n')

# Find and replace the cell
for i, cell in enumerate(notebook['cells']):
    if cell.get('cell_type') == 'code' and cell.get('source'):
        source_text = ''.join(cell['source'])
        if '# Task 2 Training with Cross-Validation' in source_text and 'PARALLELIZED' not in source_text:
            print(f"Found Task 2 training cell at index {i}")
            cell['source'] = new_source
            break

# Write back
with open('Barnett_Cottereau_Zhang_Assignment2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully!")

