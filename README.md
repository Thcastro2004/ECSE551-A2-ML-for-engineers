# Task 2 — Image Classification Workflow

This repository contains a structured workflow for training and evaluating image classification models for the Kaggle competition.

## 📋 Workflow Overview

The workflow follows a three-stage process:

1. **Quick Experiments** → 2. **Full Training** → 3. **Submission Generation** → 4. **Final Submission**

```
experiment_notebook.ipynb → Train_notebook.ipynb → generate_submission.ipynb → COLAB_notebook.ipynb
```

## 🔬 Stage 1: Quick Experiments (`experiment_notebook.ipynb`)

**Purpose:** Fast testing of new ideas (< 5-10 minutes)

**When to use:**
- Testing new hyperparameters (learning rate, label smoothing, etc.)
- Trying different architectures or image sizes
- Experimenting with data augmentation
- Quick validation of ideas before full training

**Features:**
- Uses subset of data (default: 1000 samples)
- Few epochs (default: 5)
- No cross-validation
- Fast feedback: "Does this idea help or not?"

**Workflow:**
1. Modify experiment config (hyperparameters, experiment name)
2. Run all cells
3. Check final validation accuracy
4. **If promising → move to Stage 2**
5. **If not → discard and try next idea**

---

## 🚀 Stage 2: Full Training (`Train_notebook.ipynb`)

**Purpose:** Complete model training with full dataset (10-40 minutes)

**When to use:**
- After validating an idea in `experiment_notebook.ipynb`
- When you have a promising configuration to train fully
- For hyperparameter tuning and cross-validation

**Features:**
- Full dataset (10,000 training images)
- Configurable epochs (default: 50)
- Optional 5-fold cross-validation
- Learning curve visualization
- Model checkpointing
- Saves to `generated_data/current_results/`

**Workflow:**
1. Configure hyperparameters in the config cell
2. Choose: cross-validation or single train/val split
3. Run training cells
4. Review learning curves and validation metrics
5. Final model saved to `generated_data/current_results/task2_model.pth`
6. **Move to Stage 3 to generate submission**

**Output Files:**
- `generated_data/current_results/task2_model.pth` - Trained model
- `generated_data/current_results/training_info.json` - Training metadata
- `generated_data/current_results/training_curves.png` - Learning curves

**Promoting Results:**
After training, if your current model performs better than the best:
- Run the `promote_current_to_best()` function cell
- Moves current results to best, archives old best models/submissions

---

## 📤 Stage 3: Generate Submission (`generate_submission.ipynb`)

**Purpose:** Generate Kaggle submission CSV from trained model (< 1 minute)

**When to use:**
- After training a model in `Train_notebook.ipynb`
- To test model performance on Kaggle test set
- Before final submission

**Features:**
- Loads model from `current_results` or best model
- Generates predictions on 2,000 test images
- Creates submission CSV in correct format
- Saves to `generated_data/current_results/task2_submission.csv`

**Workflow:**
1. Ensure model exists in `generated_data/current_results/` or `generated_data/`
2. Run all cells
3. Submission saved to `generated_data/current_results/task2_submission.csv`
4. Upload to Kaggle to test performance
5. **If results are good → move to Stage 4**

---

## 📝 Stage 4: Final Submission (`COLAB_notebook.ipynb`)

**Purpose:** Complete notebook for final submission

**When to use:**
- After validating model performance on Kaggle
- For final submission to assignment/competition
- Contains complete, reproducible training and submission code

**Workflow:**
1. Copy your best training configuration from `Train_notebook.ipynb`
2. Copy submission generation code from `generate_submission.ipynb`
3. Combine into a single, clean notebook
4. Ensure all code is self-contained and reproducible
5. Submit this notebook

---

## 📁 Folder Structure

```
generated_data/
├── task2_final_model.pth              # Best model (promoted from current_results)
├── task2_kaggle_submission.csv        # Best submission (promoted from current_results)
├── current_results/                    # Current training run results
│   ├── task2_model.pth                 # Current trained model
│   ├── task2_submission.csv            # Current submission
│   ├── training_info.json              # Training metadata
│   └── training_curves.png             # Learning curves
├── models_experimented/                # Archived past models
│   └── task2_model_YYYYMMDD_HHMMSS.pth
└── past_submissions/                   # Archived past submissions
    └── task2_submission_YYYYMMDD_HHMMSS.csv
```

**File Organization:**
- **Best results** → `generated_data/` (root)
- **Current experiments** → `generated_data/current_results/`
- **Past experiments** → `generated_data/models_experimented/` and `generated_data/past_submissions/`

---

## 🔄 Complete Workflow Example

### Example: Testing a new learning rate

1. **Experiment Stage:**
   ```python
   # In experiment_notebook.ipynb
   experiment_name = "test_lr_5e-4"
   learning_rate = 5e-4
   # Run notebook → Get quick validation accuracy
   ```

2. **Training Stage:**
   ```python
   # In Train_notebook.ipynb
   learning_rate = 5e-4  # Use the promising value
   # Run full training → Get model in current_results/
   ```

3. **Submission Stage:**
   ```python
   # In generate_submission.ipynb
   # Automatically loads from current_results/
   # Run → Get submission CSV
   ```

4. **Test on Kaggle:**
   - Upload `generated_data/current_results/task2_submission.csv`
   - Check leaderboard score

5. **If Better:**
   ```python
   # In Train_notebook.ipynb
   promote_current_to_best()  # Promote to best
   ```

6. **Final Submission:**
   - Copy best configuration to `COLAB_notebook.ipynb`
   - Submit complete notebook

---

## ⚙️ Configuration Tips

### Experiment Notebook
- Keep `num_epochs` small (3-8)
- Use `n_samples_to_use = 1000` for speed
- Test one idea at a time

### Training Notebook
- Start with `use_cross_validation = False` for faster iteration
- Use `use_cross_validation = True` for final validation
- Adjust `val_frequency` to control validation frequency
- Experiment with different optimizers, schedulers, and hyperparameters

### Submission Notebook
- Automatically finds best available model
- Priority: `current_results/` > `generated_data/` root

---

## 📊 Best Practices

1. **Always test in experiment notebook first** - Saves hours of training time
2. **Use descriptive experiment names** - Track what you tested
3. **Save training info** - JSON files help track what worked
4. **Compare before promoting** - Test on Kaggle before promoting to best
5. **Keep COLAB notebook clean** - Final submission should be reproducible

---

## 🛠️ Dependencies

See `requirements.txt` for full list. Main dependencies:
- PyTorch
- torchvision
- scikit-learn
- pandas
- matplotlib
- tqdm

---

## 📝 Notes

- All models use EfficientNet-B0 architecture (can be extended)
- Image size: 224x224
- 10 classes: truck, deer, bird, frog, ship, horse, cat, dog, automobile, airplane
- Training data: 10,000 labeled images
- Test data: 2,000 unlabeled images

---

## 🎯 Quick Start

1. **Quick test:** Run `experiment_notebook.ipynb`
2. **Full train:** Run `Train_notebook.ipynb`
3. **Generate submission:** Run `generate_submission.ipynb`
4. **Final submission:** Prepare `COLAB_notebook.ipynb`

Happy training! 🚀

