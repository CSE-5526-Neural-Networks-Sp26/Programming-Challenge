# 🎤 Speech Emotion Recognition Challenge
### CSE 5526: Introduction to Neural Networks — Autumn 2025

> **Due: Wednesday, May 04, 2026 at 11:59 a.m.**  
> **Team size: 2-3 students**  
> **Submit to:** your team's private GitHub repository under the `challenge/` folder

---

## Overview

In this challenge, your team will build a Speech Emotion Recognition (SER) system using PyTorch. You are given a baseline model and a labelled audio dataset. Your goal is to improve upon the baseline by exploring better architectures, features, and training strategies. Your final model will be evaluated on a **hidden test set** that you do not have access to during development.

The dataset contains audio recordings from multiple speakers, each expressing one of **6 emotions**:

| Label | Emotion |
|-------|---------|
| 0 | Anger |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Neutral |
| 5 | Sad |

The dataset is split by **speaker identity** — no speaker appears in more than one split. This makes the task harder than a random split and requires your model to generalize to unseen speakers.

| Split | Clips | Labels provided? |
|-------|-------|-----------------|
| Train | ~5,950 | ✅ Yes |
| Test | ~950 | ✅ Yes |
| Hidden test | ~1,050 | ❌ No — used for final grading |

---

## Dataset

> 📦 **Download link:** `[TO BE ANNOUNCED ON CANVAS]`

After downloading, unzip and place the dataset so the directory structure looks like this:

```
Dataset/
  train/
    audio/
    train_labels.csv
  public_test/
    audio/
    test_labels.csv
```

Each labels CSV has two columns: `clip_id` and `emotion`.

---

## Repository Structure

```
challenge/
  dataloader.py      ← loads audio, computes Mel-spectrograms, returns DataLoaders
  baseline.py        ← baseline LSTM model definition
  train.py           ← training loop with early stopping, wandb logging, checkpointing
  test.py            ← evaluates a saved model on the test set
  results/
    baseline/
      best_model.pt  ← saved baseline model after training
      norm_stats.pt  ← normalisation statistics (required by test.py)
      loss_curve.png ← generated after training
```

You are free to modify any of the provided files or add new ones. The only requirement is that your final submission contains a working `test.py` (see [Submission](#submission) below).

---

## Baseline Model

The provided baseline is a 2-layer LSTM trained on 64-bin Mel-spectrograms. It is intentionally simple — there is clear room to improve it. The baseline configuration:

- **Input:** 64-bin Mel-spectrogram (25 ms window, 10 ms hop)
- **Architecture:** 2-layer LSTM, 128 hidden units, linear output layer
- **Optimizer:** Adam, lr = 1e-2
- **Scheduler:** ReduceLROnPlateau (factor 0.5, patience 5)
- **Batch size:** 64 | **Max epochs:** 100 | **Early stopping patience:** 10
- **Initialisation:** Xavier uniform

---

## Getting Started

### 1. Install dependencies

```bash
pip install torch torchaudio scikit-learn pandas matplotlib wandb tqdm
```

### 2. Download and prepare the dataset

Download the dataset from the link on Canvas and unzip it so the `Dataset/` folder is in the same directory as your code.

### 3. Train the baseline

```bash
python train.py \
    --data_dir  dataset \
    --output_dir results/baseline \
    --run_name  baseline_train
```

This will:
- Split the training set into train (85%) and validation (15%)
- Train for up to 20 epochs with early stopping
- Save `best_model.pt` and `norm_stats.pt` to `results/baseline/`
- Save `loss_curve.png` to `results/baseline/`
- Log training metrics to [Weights & Biases](https://wandb.ai)

### 4. Evaluate on the test set

```bash
python test.py \
    --test_dir   dataset/public_test \
    --model_path results/baseline/best_model.pt \
    --stats_path results/baseline/norm_stats.pt \
    --run_name   baseline_test
```

This prints weighted F1 and per-class F1 to stdout and logs a confusion matrix to wandb.

### 5. Resume an interrupted training run

If training is interrupted, simply re-run the same `train.py` command — it will automatically detect the checkpoint and resume from where it left off, continuing the same wandb run.

---

## Your Task

Implement and evaluate your model that improves upon the baseline. You are free to explore any direction, including but not limited to:

**Architecture**
- Bidirectional LSTM
- CNN front-end for local feature extraction
- Attention mechanism over LSTM outputs
- Transformer encoder
- Hybrid CNN + LSTM / Transformer

**Features**
- MFCCs, delta and delta-delta features
- Log-Mel filterbanks
- Combinations of multiple feature types

**Training strategy**
- Dropout, weight decay, label smoothing
- Learning rate warmup or cosine annealing
- Data augmentation: noise injection, pitch shifting, time stretching

For each approach, your notebook must include:
1. A description of what was changed and why
2. Training and validation learning curves
3. Weighted F1 and per-class F1 on the test set
4. A comparison with the baseline and a discussion of what helped

---

## Submission

Commit the following to the `challenge/` folder in your team's private GitHub repository before the deadline:

| File | Description |
|------|-------------|
| `dataloader.py` | Your data loading pipeline |
| `baseline.py` | Model definition(s) |
| `train.py` | Training pipeline |
| `test.py` | Evaluation script |
| `inference.py` | **Self-contained inference script** (see spec below) |
| `best_model.pt` | Your best trained model weights |
| `run.ipynb` | Executed notebook with all outputs visible |

> ⚠️ **Unexecuted notebooks receive zero.** Run all cells before committing.

### Inference script specification

Your `inference.py` must follow this exact interface:

```bash
python inference.py \
    --audio_dir /path/to/audio \
    --labels_csv /path/to/labels.csv
```

It must load `best_model.pt` from the same directory as the script, run inference on all `.wav` files in `--audio_dir`, and print the following to stdout:

```
Weighted F1:  0.XXXX
Per-class F1:
  Anger:    0.XXXX
  Disgust:  0.XXXX
  Fear:     0.XXXX
  Happy:    0.XXXX
  Neutral:  0.XXXX
  Sad:      0.XXXX
```

> ⚠️ **A script that crashes or produces incorrectly formatted output receives zero for the hidden test component.** Test it on the public test set before submitting.

---

## Grading

| Component | Points |
|-----------|--------|
| Experimental approaches (≥2, each with curves + analysis) | 40 |
| Hidden test performance (weighted F1 vs. baseline) | 30 |
| Code quality (readability, comments, efficiency) | 15 |
| Notebook completeness (all outputs visible, well-organized) | 15 |
| **Total** | **100** |

### Hidden test performance rubric

| Weighted F1 vs. baseline | Points |
|--------------------------|--------|
| Below baseline | 0 |
| Within 2% of baseline | 10 |
| Beats baseline by 2–5% | 18 |
| Beats baseline by 5–10% | 25 |
| Beats baseline by > 10% | 30 |

---

## Academic Integrity

- All code must be written by your team
- You may consult papers, documentation, and blogs for learning — but **do not copy code**
- Sharing code, models, or predictions between teams is not permitted
- Use of pre-trained models or external transfer learning is not allowed
- All team members must be able to explain every part of the submitted code

Violations will result in a zero for the assignment and may be referred to the university academic conduct process.

---

## Questions

Post general questions in the **`#challenge`** channel on MS Teams. Do not include any part of your solution in public posts. For help with your specific implementation, message the instructor or TA privately.
