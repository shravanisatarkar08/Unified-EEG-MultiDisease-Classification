from pathlib import Path
import sys
import torch
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.eeg_classifier import EEGClassifier
from training.dataset import get_dataloaders, IDX_TO_CLASS

device = torch.device("cpu")
ckpt_path = Path("checkpoints/best_model.pt")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

model = EEGClassifier(
    n_channels=19,
    n_samples=1280,
    num_classes=5,
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    ff_dim=128,
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

loaders = get_dataloaders(
    index_csv="datasets/metadata/model_dataset_index.csv",
    project_root=".",
    batch_size=64,
    num_workers=0,
    shuffle_train=False,
    interpolate_missing=False,
)
val_loader = loaders["val"]

records = []
with torch.no_grad():
    for eeg, labels, metas in val_loader:
        logits = model(eeg)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
        for i in range(len(labels)):
            records.append({
                "dataset_name": metas[i]["dataset_name"],
                "subject_id": metas[i]["subject_id"],
                "true_label": int(labels[i]),
                "true_class": IDX_TO_CLASS[int(labels[i])],
                "pred_label": int(preds[i]),
                "pred_class": IDX_TO_CLASS[int(preds[i])],
                "correct": int(labels[i] == preds[i]),
                "p_healthy": probs[i, 0],
                "p_epilepsy": probs[i, 1],
                "p_alzheimers": probs[i, 2],
                "p_parkinsons": probs[i, 3],
                "p_depression": probs[i, 4],
            })

df = pd.DataFrame(records)
print("=== Softmax Probability Distributions for True Healthy Samples ===")
healthy = df[df['true_class'] == 'healthy']
for ds in sorted(healthy['dataset_name'].unique()):
    sub = healthy[healthy['dataset_name'] == ds]
    print(f"\nDataset: {ds} (n={len(sub)})")
    print(sub[['p_healthy', 'p_alzheimers', 'p_parkinsons', 'p_depression', 'pred_class', 'correct']].describe().round(3))
