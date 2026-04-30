import ast
import csv
import random

path = "/data/home/zhanghx/code/infilling-score/results/s1-32B-0.8_s1K_split_optimized_half_clipped_results.csv"

with open(path, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if not rows:
    raise ValueError("CSV file contains no data rows")

scores = ast.literal_eval(rows[-1]["scores"])
labels = ast.literal_eval(rows[-1]["labels"])

# split into val and test sets, 100 and rest
random.seed(42)
indices = list(range(len(scores)))
random.shuffle(indices)
val_indices = indices[:100]
test_indices = indices[100:]
val_scores = [scores[i] for i in val_indices]
val_labels = [labels[i] for i in val_indices]
test_scores = [scores[i] for i in test_indices]
test_labels = [labels[i] for i in test_indices]


# import pdb; pdb.set_trace()
# get threshold from val set by maxmizing largest accuracy
best_acc = 0.0
best_threshold = 0.0
for threshold in set(val_scores):
    preds = [1 if score >= threshold else 0 for score in val_scores]
    correct = sum([1 for p, l in zip(preds, val_labels) if p == l])
    acc = correct / len(val_labels)
    if acc > best_acc:
        best_acc = acc
        best_threshold = threshold
print(f"Best threshold on val set: {best_threshold}, Best accuracy: {best_acc:.4f}")


# evaluate on test set
test_preds = [1 if score >= best_threshold else 0 for score in test_scores]
test_correct = sum([1 for p, l in zip(test_preds, test_labels) if p == l])
test_acc = test_correct / len(test_labels)
tpr = sum([1 for p, l in zip(test_preds, test_labels) if p == 1 and l ==1]) / sum([1 for l in test_labels if l ==1])
fpr = sum([1 for p, l in zip(test_preds, test_labels) if p == 1 and l ==0]) / sum([1 for l in test_labels if l ==0])
print(f"Test set TPR: {tpr:.4f}, FPR: {fpr:.4f}")
print(f"Test set accuracy: {test_acc:.4f}")