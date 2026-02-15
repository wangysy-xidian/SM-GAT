import os
import torch
import logging
import random
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict

# Assumes relative import structure
from .graph_encoder import TrafficGraphModel
# NOTE: User needs to provide dataset.py
from .dataset import TrafficGraphDataset

# === Configuration ===
CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'min_samples': 100,  # Threshold for class balancing
    'num_kernels': 64,
    'embed_dim': 128,
    'hidden_dim': 256,
    'gat_heads': 4,
    # Generic Paths
    'data_root': './data/processed',
    'checkpoint_dir': './checkpoints'
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def balance_graph_dataset(dataset_obj, min_samples=100):
    """
    Memory-level oversampling for the TrafficGraphDataset to handle class imbalance.
    """
    logger.info(f"Balancing Graph Dataset (Threshold: {min_samples})...")

    labels = [d.y.item() for d in dataset_obj.data_list]
    category_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        category_indices[label].append(idx)

    balanced_indices = []
    for label, indices in category_indices.items():
        count = len(indices)
        balanced_indices.extend(indices)

        if count < min_samples:
            gap = min_samples - count
            supplement = random.choices(indices, k=gap)
            balanced_indices.extend(supplement)

    original_data = dataset_obj.data_list
    balanced_data = [original_data[i] for i in balanced_indices]
    random.shuffle(balanced_data)

    dataset_obj.data_list = balanced_data
    logger.info(f"Balancing Complete: {len(original_data)} -> {len(balanced_data)}")
    return dataset_obj


def main():
    if not os.path.exists(CONFIG['checkpoint_dir']):
        os.makedirs(CONFIG['checkpoint_dir'])

    logger.info(f"[*] Device: {CONFIG['device']}")

    # --- A. Data Preparation ---
    logger.info(" Loading Datasets...")
    label_encoder = LabelEncoder()

    train_path = os.path.join(CONFIG['data_root'], "train.jsonl")
    val_path = os.path.join(CONFIG['data_root'], "valid.jsonl")

    # Load Training Data
    train_dataset = TrafficGraphDataset(train_path, label_encoder=label_encoder, fit_label=True)
    train_dataset = balance_graph_dataset(train_dataset, min_samples=CONFIG['min_samples'])

    # Load Validation Data
    valid_dataset = TrafficGraphDataset(val_path, label_encoder=label_encoder, fit_label=False)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    num_classes = len(label_encoder.classes_)
    logger.info(f"[*] Detected {num_classes} classes.")

    # --- B. Model Initialization ---
    model = TrafficGraphModel(
        num_classes=num_classes,
        num_kernels=CONFIG['num_kernels'],
        embed_dim=CONFIG['embed_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        gat_heads=CONFIG['gat_heads']
    ).to(CONFIG['device'])

    # --- C. Anchor Initialization ---
    logger.info("Initializing Numerical Anchors (K-Means)...")
    sample_packets = []
    for i in range(min(1000, len(train_dataset))):
        data = train_dataset[i]
        seqs = [s for s in data.x_seq.view(-1).tolist() if s != 0]
        sample_packets.extend(seqs)
    model.node_processor.init_anchors(sample_packets)

    # --- D. Optimizer & Loss ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_acc = 0.0

    # --- E. Training Loop ---
    logger.info("Starting Training...")
    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        total_loss = 0
        train_preds, train_labels = [], []

        for batch in train_loader:
            batch = batch.to(CONFIG['device'])
            optimizer.zero_grad()

            logits = model(batch)
            loss = criterion(logits, batch.y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_preds.extend(logits.argmax(dim=1).cpu().numpy())
            train_labels.extend(batch.y.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(CONFIG['device'])
                logits = model(batch)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(batch.y.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        scheduler.step(val_acc)
        logger.info(
            f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(CONFIG['checkpoint_dir'], "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_encoder': label_encoder,
                'config': CONFIG
            }, save_path)

    # Save Last Model
    last_path = os.path.join(CONFIG['checkpoint_dir'], "last_model.pth")
    torch.save({
        'epoch': CONFIG['epochs'],
        'model_state_dict': model.state_dict(),
        'config': CONFIG
    }, last_path)
    logger.info(f"Training Finished.")


if __name__ == "__main__":

    main()
