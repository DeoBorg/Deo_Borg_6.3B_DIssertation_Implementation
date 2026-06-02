"""
Generates all evaluation outputs for the dissertation analysis chapter:
  1. Behaviour classification report (precision, recall, F1)
  2. Confusion matrix (counts + normalised)
  3. Temporal behaviour timeline per track
  4. Label switch rate analysis
  5. Class distribution charts
  6. K-Fold LSTM results summary

Usage:
    python evaluate.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score)

# ── CONFIG ───────────────────────────────────────────────────────────────────
GROUND_TRUTH_CSV  = "dataset/ground_truth_labels.csv"
PREDICTIONS_CAM1  = "outputs/predictions_cam1.csv"
PREDICTIONS_CAM2  = "outputs/predictions_cam2.csv"
LSTM_RESULTS_CSV  = "outputs/lstm_fold_results.csv"
OUTPUT_DIR        = "outputs/evaluation"

BEHAVIOUR_CLASSES = ['Focused', 'Chatting', 'Looking Away', 'Using Phone']

LABEL_COLOURS = {
    'Focused':      '#2ecc71',
    'Chatting':     '#3498db',
    'Looking Away': '#e67e22',
    'Using Phone':  '#e74c3c',
}
# ─────────────────────────────────────────────────────────────────────────────


def setup():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size']  = 11


# ── 1. Behaviour Classification Evaluation ───────────────────────────────────
def evaluate_behaviour_classification():
    """
    Match predictions against ground truth labels.
    Since predictions use ByteTrack IDs and GT uses simple IDs,
    we match on frame_id + camera_id and find the closest track.
    """
    print("  1. Behaviour Classification Evaluation...")

    gt = pd.read_csv(GROUND_TRUTH_CSV, dtype=str)
    gt = gt[gt['behaviour_label'] != 'ambiguous'].reset_index(drop=True)
    gt['frame_id_int'] = gt['frame_id'].astype(int)

    pred_cam1 = pd.read_csv(PREDICTIONS_CAM1, dtype=str)
    pred_cam2 = pd.read_csv(PREDICTIONS_CAM2, dtype=str)
    pred_cam1['camera_id'] = 'cam1'
    pred_cam2['camera_id'] = 'cam2'
    preds = pd.concat([pred_cam1, pred_cam2], ignore_index=True)
    preds = preds[preds['predicted_label'] != 'Unknown']
    preds['frame_id_int'] = preds['frame_id'].astype(int)

    # Match GT to predictions by frame_id and camera_id
    # For each GT entry, find the most common predicted label in that frame
    matched_true = []
    matched_pred = []

    for _, gt_row in gt.iterrows():
        fid    = gt_row['frame_id_int']
        cam    = gt_row['camera_id']
        true_l = gt_row['behaviour_label']

        # Find predictions for this frame and camera
        frame_preds = preds[
            (preds['frame_id_int'] == fid) &
            (preds['camera_id'] == cam)
        ]

        if len(frame_preds) == 0:
            continue

        # Use majority label in the frame as the prediction
        # (since track IDs don't align perfectly between GT and pipeline)
        pred_label = frame_preds['predicted_label'].mode()[0]

        matched_true.append(true_l)
        matched_pred.append(pred_label)

    if not matched_true:
        print("  [WARN] No matched predictions found")
        return None, None

    print(f"  Matched {len(matched_true)} ground truth samples to predictions")

    # Classification report
    report = classification_report(
        matched_true, matched_pred,
        labels=BEHAVIOUR_CLASSES,
        target_names=BEHAVIOUR_CLASSES,
        zero_division=0,
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(OUTPUT_DIR, 'behaviour_classification_report.csv'))

    print(f"\n  Classification Report:")
    print(classification_report(
        matched_true, matched_pred,
        labels=BEHAVIOUR_CLASSES,
        target_names=BEHAVIOUR_CLASSES,
        zero_division=0
    ))

    return matched_true, matched_pred


# ── 2. Confusion Matrix ───────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred):
    print("  2. Confusion Matrix...")

    cm      = confusion_matrix(y_true, y_pred, labels=BEHAVIOUR_CLASSES)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=BEHAVIOUR_CLASSES, yticklabels=BEHAVIOUR_CLASSES)
    axes[0].set_title('Behaviour Confusion Matrix (Counts)')
    axes[0].set_ylabel('Ground Truth')
    axes[0].set_xlabel('Predicted')
    plt.setp(axes[0].get_xticklabels(), rotation=30, ha='right')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                xticklabels=BEHAVIOUR_CLASSES, yticklabels=BEHAVIOUR_CLASSES)
    axes[1].set_title('Behaviour Confusion Matrix (Normalised)')
    axes[1].set_ylabel('Ground Truth')
    axes[1].set_xlabel('Predicted')
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'behaviour_confusion_matrix.png'),
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: behaviour_confusion_matrix.png")


# ── 3. Temporal Timeline ──────────────────────────────────────────────────────
def plot_temporal_timelines():
    print("  3. Temporal Behaviour Timelines...")

    pred_cam1 = pd.read_csv(PREDICTIONS_CAM1)
    pred_cam1['camera_id'] = 'cam1'

    pred_cam1 = pred_cam1[pred_cam1['predicted_label'] != 'Unknown']
    pred_cam1 = pred_cam1.sort_values('frame_id')

    # Get top 4 most active tracks
    track_counts = pred_cam1.groupby('track_id').size()
    top_tracks   = track_counts.nlargest(4).index.tolist()

    label_to_num = {l: i for i, l in enumerate(BEHAVIOUR_CLASSES)}

    fig, axes = plt.subplots(len(top_tracks), 1,
                             figsize=(16, 3 * len(top_tracks)),
                             sharex=True)

    if len(top_tracks) == 1:
        axes = [axes]

    for ax, track_id in zip(axes, top_tracks):
        student = pred_cam1[pred_cam1['track_id'] == track_id].copy()
        student = student.sort_values('frame_id')

        for _, row in student.iterrows():
            label = row['predicted_label']
            if label not in label_to_num:
                continue
            colour = LABEL_COLOURS.get(label, '#95a5a6')
            ax.bar(row['frame_id'], 1, bottom=label_to_num[label],
                   color=colour, width=30, align='center', alpha=0.85)

        ax.set_yticks([i + 0.5 for i in range(len(BEHAVIOUR_CLASSES))])
        ax.set_yticklabels(BEHAVIOUR_CLASSES, fontsize=9)
        ax.set_ylabel(f'Track {track_id}', fontsize=10)
        ax.set_ylim(0, len(BEHAVIOUR_CLASSES))
        ax.grid(axis='x', alpha=0.3)

    axes[-1].set_xlabel('Frame Number')
    fig.suptitle('Behaviour Timeline per Student Track (Camera 1)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'behaviour_timelines.png'),
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: behaviour_timelines.png")


# ── 4. Label Switch Rate ──────────────────────────────────────────────────────
def label_switch_rate():
    print("  4. Label Switch Rate Analysis...")

    results = []
    for cam, csv_path in [('cam1', PREDICTIONS_CAM1),
                           ('cam2', PREDICTIONS_CAM2)]:
        df = pd.read_csv(csv_path)
        df = df[df['predicted_label'] != 'Unknown']
        df = df.sort_values(['track_id', 'frame_id'])
        df['prev_label'] = df.groupby('track_id')['predicted_label'].shift(1)
        df['switched']   = df['predicted_label'] != df['prev_label']
        switch_rate      = df.groupby('track_id')['switched'].mean()
        for tid, rate in switch_rate.items():
            results.append({'camera': cam, 'track_id': tid,
                            'switch_rate': rate})

    results_df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(10, 5))
    for cam, colour in [('cam1', '#3498db'), ('cam2', '#e74c3c')]:
        data = results_df[results_df['camera'] == cam]['switch_rate']
        ax.hist(data, bins=15, alpha=0.6, color=colour, label=f'Camera {cam[-1]}')

    ax.set_xlabel('Label Switch Rate (proportion of frames where label changes)')
    ax.set_ylabel('Number of Tracks')
    ax.set_title('Temporal Stability — Label Switch Rate per Track')
    ax.legend()
    ax.grid(alpha=0.3)

    mean_rate = results_df['switch_rate'].mean()
    ax.axvline(mean_rate, color='black', linestyle='--',
               label=f'Mean: {mean_rate:.3f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'label_switch_rate.png'),
                bbox_inches='tight')
    plt.close()

    print(f"  Mean switch rate : {mean_rate:.4f}")
    print(f"  Saved: label_switch_rate.png")
    return mean_rate


# ── 5. Class Distribution ─────────────────────────────────────────────────────
def plot_class_distributions():
    print("  5. Class Distribution Charts...")

    gt = pd.read_csv(GROUND_TRUTH_CSV)
    gt = gt[gt['behaviour_label'] != 'ambiguous']

    pred_cam1 = pd.read_csv(PREDICTIONS_CAM1)
    pred_cam2 = pd.read_csv(PREDICTIONS_CAM2)
    all_preds = pd.concat([pred_cam1, pred_cam2])
    all_preds = all_preds[all_preds['predicted_label'] != 'Unknown']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colours   = [LABEL_COLOURS[l] for l in BEHAVIOUR_CLASSES]

    # Ground truth distribution
    gt_counts = gt['behaviour_label'].value_counts().reindex(
        BEHAVIOUR_CLASSES, fill_value=0)
    axes[0].bar(BEHAVIOUR_CLASSES, gt_counts.values, color=colours, alpha=0.85)
    axes[0].set_title('Ground Truth Label Distribution')
    axes[0].set_ylabel('Count')
    axes[0].set_xlabel('Behaviour Class')
    plt.setp(axes[0].get_xticklabels(), rotation=20, ha='right')
    for i, v in enumerate(gt_counts.values):
        axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

    # Prediction distribution
    pred_counts = all_preds['predicted_label'].value_counts().reindex(
        BEHAVIOUR_CLASSES, fill_value=0)
    axes[1].bar(BEHAVIOUR_CLASSES, pred_counts.values, color=colours, alpha=0.85)
    axes[1].set_title('Pipeline Prediction Distribution (Both Cameras)')
    axes[1].set_ylabel('Count')
    axes[1].set_xlabel('Behaviour Class')
    plt.setp(axes[1].get_xticklabels(), rotation=20, ha='right')
    for i, v in enumerate(pred_counts.values):
        axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_distributions.png'),
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: class_distributions.png")


# ── 6. LSTM K-Fold Summary ────────────────────────────────────────────────────
def plot_kfold_summary():
    print("  6. LSTM K-Fold Summary Chart...")

    if not os.path.exists(LSTM_RESULTS_CSV):
        print("  [SKIP] lstm_fold_results.csv not found")
        return

    df = pd.read_csv(LSTM_RESULTS_CSV)

    f1_cols = [f'f1_{name}' for name in BEHAVIOUR_CLASSES]
    means   = [df[col].mean() for col in f1_cols]
    stds    = [df[col].std()  for col in f1_cols]
    colours = [LABEL_COLOURS[l] for l in BEHAVIOUR_CLASSES]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-class F1
    bars = axes[0].bar(BEHAVIOUR_CLASSES, means, yerr=stds,
                       color=colours, alpha=0.85, capsize=5,
                       error_kw={'linewidth': 2})
    axes[0].set_title('LSTM Per-Class F1 Score (Mean ± Std across 5 Folds)')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_ylim(0, 1.0)
    plt.setp(axes[0].get_xticklabels(), rotation=20, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    for bar, mean, std in zip(bars, means, stds):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     mean + std + 0.02,
                     f'{mean:.3f}', ha='center', fontsize=9)

    # Per-fold macro F1
    axes[1].plot(df['fold'], df['macro_f1'], 'o-',
                 color='#2c3e50', linewidth=2, markersize=8, label='Macro F1')
    axes[1].plot(df['fold'], df['val_accuracy'], 's--',
                 color='#7f8c8d', linewidth=2, markersize=8, label='Accuracy')
    axes[1].axhline(df['macro_f1'].mean(), color='#e74c3c',
                    linestyle=':', linewidth=2,
                    label=f'Mean F1: {df["macro_f1"].mean():.3f}')
    axes[1].set_title('LSTM Performance per Fold')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Score')
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xticks(df['fold'].tolist())
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'lstm_kfold_summary.png'),
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: lstm_kfold_summary.png")


# ── 7. Summary Report ─────────────────────────────────────────────────────────
def print_summary(y_true, y_pred, switch_rate):
    print(f"\n{'='*55}")
    print(f"  DISSERTATION RESULTS SUMMARY")
    print(f"{'='*55}")

    print(f"\n  Object Detection (YOLOv8m Fine-Tuned):")
    print(f"    mAP@0.5      : 0.9476")
    print(f"    mAP@0.5:0.95 : 0.9012")
    print(f"    Person F1    : 0.9530")
    print(f"    Laptop F1    : 0.9459")
    print(f"    Phone F1     : 0.8878")

    if y_true and y_pred:
        macro_f1  = f1_score(y_true, y_pred, average='macro',
                             labels=BEHAVIOUR_CLASSES, zero_division=0)
        accuracy  = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
        print(f"\n  Behaviour Classification (LSTM):")
        print(f"    K-Fold Mean Accuracy : 0.3774 ± 0.0250")
        print(f"    K-Fold Mean Macro F1 : 0.3087 ± 0.0179")
        print(f"    Pipeline Accuracy    : {accuracy:.4f}")
        print(f"    Pipeline Macro F1    : {macro_f1:.4f}")

    print(f"\n  Temporal Stability:")
    print(f"    Mean Label Switch Rate : {switch_rate:.4f}")
    lower = switch_rate < 0.3
    print(f"    Interpretation         : "
          f"{'Stable' if lower else 'Moderate stability'}")

    print(f"\n  Outputs saved to: {OUTPUT_DIR}/")
    print(f"{'='*55}\n")


def main():
    print("\n=== Phase 8: System Evaluation ===\n")
    setup()

    y_true, y_pred = evaluate_behaviour_classification()
    if y_true:
        plot_confusion_matrix(y_true, y_pred)
    plot_temporal_timelines()
    switch_rate = label_switch_rate()
    plot_class_distributions()
    plot_kfold_summary()
    print_summary(y_true, y_pred, switch_rate)


if __name__ == "__main__":
    main()