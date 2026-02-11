import os
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from .graph_encoder import TrafficGraphModel
from .dataset import TrafficGraphDataset

# === Configuration ===
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = "./checkpoints/best_model.pth"

# Dictionary of test sets
TEST_SETS = {
    "In-Distribution": "./data/processed/test_indistribution.jsonl",
    "Drift/OOD": "./data/processed/test_drift_ood.jsonl"
}


def run_evaluation(model, dataset, label_encoder, name):
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            targets = batch.y.cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets)

    # Calculate Metrics (Macro-Average for class balance)
    acc = accuracy_score(all_targets, all_preds)
    prec_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    rec_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    print(f"[*] Global Accuracy:  {acc:.4f}")
    print(f"[*] Macro Precision:  {prec_macro:.4f}")
    print(f"[*] Macro Recall:     {rec_macro:.4f}")
    print(f"[*] Macro F1-Score:   {f1_macro:.4f}")

    target_names = label_encoder.classes_
    print("\n--- Detailed Classification Report ---")
    print(classification_report(all_targets, all_preds, target_names=target_names, digits=4, zero_division=0))

    return acc, prec_macro, rec_macro, f1_macro


def main():
    print(f"[*] Loading Checkpoint: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[!] Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    config = checkpoint['config']

    # Handle label encoder storage variations
    if 'label_encoder' in checkpoint:
        label_encoder = checkpoint['label_encoder']
    else:
        print("[!] Label Encoder not found in checkpoint. Cannot decode classes.")
        return

    num_classes = len(label_encoder.classes_)
    print(f"[*] Detected {num_classes} classes.")

    model = TrafficGraphModel(
        num_classes=num_classes,
        num_kernels=config['num_kernels'],
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        gat_heads=config['gat_heads']
    ).to(DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'])
    print("[*] Weights restored successfully.")

    results = {}
    for name, path in TEST_SETS.items():
        if os.path.exists(path):
            print(f"\n[*] Loading Dataset: {path}")
            try:
                ds = TrafficGraphDataset(path, label_encoder=label_encoder, fit_label=False)
                if len(ds) > 0:
                    acc, prec, rec, f1 = run_evaluation(model, ds, label_encoder, name)
                    results[name] = {
                        "Accuracy": acc,
                        "Macro-P": prec,
                        "Macro-R": rec,
                        "Macro-F1": f1
                    }
            except Exception as e:
                print(f"[!] Error loading {name}: {e}")
        else:
            print(f"[!] Skipped {name}: File not found at {path}")

    # Final Summary Table
    print("\n" + "=" * 80)
    print(f"{'FINAL EVALUATION SUMMARY':^80}")
    print("=" * 80)
    header = f"{'Dataset':<30} | {'Acc':<10} | {'Macro-P':<10} | {'Macro-R':<10} | {'Macro-F1':<10}"
    print(header)
    print("-" * 80)

    for name, metrics in results.items():
        row = f"{name:<30} | {metrics['Accuracy']:.4f}     | {metrics['Macro-P']:.4f}     | {metrics['Macro-R']:.4f}     | {metrics['Macro-F1']:.4f}"
        print(row)
    print("=" * 80)


if __name__ == "__main__":
    main()