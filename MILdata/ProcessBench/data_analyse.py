from datasets import load_dataset

ds = load_dataset("Qwen/ProcessBench", split="math")

cnt = 0
for row in ds:
    if row['label'] == -1:
        cnt += 1
        assert row['final_answer_correct'] == True

print(cnt/len(ds))