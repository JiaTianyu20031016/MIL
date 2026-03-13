from MILdata.shepherd.dataset import load_dataset as load_shepherd_dataset

ds = load_shepherd_dataset("math")

cnt = 0
for row in ds:
    labels = [i.label for i in row.segments]
    cnt += labels[-1] == 1

print(cnt / len(ds))