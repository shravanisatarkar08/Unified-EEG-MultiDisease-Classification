import pandas as pd

df = pd.read_csv('results/experiments/exp3_hyperparam_tuning/control_lr5e-4_wd1e-2_drop0.25/val_predictions.csv')
print("=== Overall Breakdown ===")
print(f"Total validation samples: {len(df)}")
n_corr = df['correct'].sum()
print(f"Total correct: {n_corr} ({df['correct'].mean()*100:.2f}%)")
print(f"Total errors: {len(df) - n_corr} ({(1-df['correct'].mean())*100:.2f}%)")

print("\n=== Confusion Matrix (True vs Pred) ===")
ct = pd.crosstab(df['true_class'], df['pred_class'], margins=True)
print(ct)

print("\n=== Top Misclassification Pairs ===")
errors = df[df['correct'] == 0]
top_pairs = errors.groupby(['true_class', 'pred_class']).size().reset_index(name='count').sort_values('count', ascending=False)
print(top_pairs.to_string(index=False))

print("\n=== Misclassification Pairs by Dataset ===")
by_ds = errors.groupby(['dataset_name', 'true_class', 'pred_class']).size().reset_index(name='count').sort_values('count', ascending=False)
print(by_ds.to_string(index=False))

print("\n=== Healthy Misclassification Detail ===")
healthy = df[df['true_class'] == 'healthy']
print(pd.crosstab(healthy['dataset_name'], healthy['pred_class'], margins=True))

print("\n=== Disease Misclassification Detail (False Positives for Healthy) ===")
disease = df[df['true_class'] != 'healthy']
print(pd.crosstab(disease['dataset_name'], disease['pred_class'], margins=True))
