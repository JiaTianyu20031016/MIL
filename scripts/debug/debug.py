from MILdata.annotation.dataset import load_dataset

ds = load_dataset("data/math-shepherd-Qwen3-4B-softmin-document-balanced/eval_annotations.jsonl")

print(f"Loaded {len(ds)} samples.")
accuracies = [(sample.rating == 1.0) == (sample.positive_prob == 1.0) for sample in ds]
positive_mask = [sample.positive_prob == 1.0 for sample in ds]
print(f"Accuracy: {sum(accuracies) / len(accuracies):.4f}")
print(f"Positive accuracy: {sum([a and m for a, m in zip(accuracies, positive_mask)]) / sum(positive_mask):.4f}")