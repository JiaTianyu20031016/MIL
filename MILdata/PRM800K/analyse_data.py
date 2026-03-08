from datasets import load_dataset, load_from_disk

path = 'MIL/MILdata/PRM800K/data/data_processed'
dataset = load_from_disk(path)['train']

length_correct = []
length_incorrect = []
first_error_relative_positions = []
for example in dataset:
    if example['labels'].count(True) == len(example['labels']):
        length_correct.append(len(example['completions']))
    else:
        length_incorrect.append(len(example['completions']))
        total_steps = len(example['labels'])
        for idx, label in enumerate(example['labels']):
            if not label:
                first_error_relative_positions.append((idx + 1) / total_steps)
                break

print(f"Average length of correct solutions: {sum(length_correct) / len(length_correct)}")
print(f"Average length of incorrect solutions: {sum(length_incorrect) / len(length_incorrect)}")

import matplotlib.pyplot as plt
plt.hist(length_correct, bins=20, alpha=0.5, label='Correct Solutions', density=True)
plt.hist(length_incorrect, bins=20, alpha=0.5, label='Incorrect Solutions', density=True)
plt.xlabel('Number of Steps')
plt.ylabel('Relative Frequency')
plt.title('Distribution of Solution Lengths')
plt.legend()
plt.savefig('solution_length_distribution.png')

if first_error_relative_positions:
    plt.figure()
    plt.hist(
        first_error_relative_positions,
        bins=20,
        alpha=0.7,
        color='tab:orange',
        density=True,
    )
    plt.xlabel('First Error Relative Position')
    plt.ylabel('Relative Frequency')
    plt.title('Relative Position of First Error in Incorrect Solutions')
    plt.savefig('first_error_relative_position_distribution.png')