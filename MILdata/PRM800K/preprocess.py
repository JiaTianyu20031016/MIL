# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from datasets import load_dataset
from huggingface_hub import ModelCard
from transformers import HfArgumentParser

FILTERED_FINISH_REASONS = {"give_up", "bad_problem"}


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the dataset to the Hugging Face Hub.
        save_to_disk (`str`, *optional*):
            Local directory to save the processed dataset to.
        repo_id (`str`, *optional*, defaults to `"trl-lib/prm800k"`):
            Hugging Face repository ID to push the dataset to.
        dataset_num_proc (`int`, *optional*):
            Number of workers to use for dataset processing.
    """

    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the dataset to the Hugging Face Hub."},
    )
    repo_id: str = field(
        default="trl-lib/prm800k",
        metadata={"help": "Hugging Face repository ID to push the dataset to."},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of workers to use for dataset processing."},
    )
    save_to_disk: str | None = field(
        default='MIL/MILdata/PRM800K_raw/data/data_processed',
        metadata={"help": "Local directory to save the processed dataset to."},
    )


def process_example(example):
    
    outputs = []
    prompt = example["question"]["problem"]
    pre_generated_steps = example["question"]["pre_generated_steps"]

    # Iterate through each step
    previous_completions = []
    previous_labels = []
    for step_idx, step in enumerate(example["label"]["steps"]):
        if step["completions"] is None and step["human_completion"] is None and step["chosen_completion"] is None:
            # happens sometimes
            break
        # Loop through completions
        for completion_idx, completion in enumerate(step["completions"]):
            if completion['rating'] is None:
                # happens sometimes
                continue
            if completion['rating'] > -1 and '# Answer' in completion['text']:
                # Reach a terminal state with a correct answer, we can stop here
                outputs.append({"prompt": prompt, "completions": previous_completions + [completion['text']], "labels": previous_labels + [True]})
            if completion['rating'] == -1 and completion['text'] == pre_generated_steps[step_idx]:
                # This is the pre-generated step with error, so we can mark the pre_generated solution as incorrect and stop here
                outputs.append({"prompt": prompt, "completions": pre_generated_steps, "labels": previous_labels + [False] + [None] * (len(pre_generated_steps) - len(previous_labels) - 1)})
                assert step_idx == len(example["label"]["steps"]) - 1, "If the pre-generated step is wrong, the labeler should not label the next steps"

        previous_completions.append(pre_generated_steps[step_idx])
        previous_labels.append(True)

    return outputs


def process_batch(examples):
    outputs = []
    batch_size = len(examples["label"])
    for idx in range(batch_size):
        example = {k: v[idx] for k, v in examples.items()}
        outputs.extend(process_example(example))
    # list of dict to dict of list
    outputs = {k: [v[k] for v in outputs] for k in outputs[0]}
    return outputs


def keep_example(example):
    return example["label"]["finish_reason"] not in FILTERED_FINISH_REASONS


model_card = ModelCard("""
---
tags: [trl]
---

# PRM800K Dataset

## Summary

The PRM800K dataset is a processed version of [OpenAI's PRM800K](https://github.com/openai/prm800k), designed to train models using the [TRL library](https://github.com/huggingface/trl) for stepwise supervision tasks. It contains 800,000 step-level correctness labels for model-generated solutions to problems from the MATH dataset. This dataset enables models to learn and verify each step of a solution, enhancing their reasoning capabilities.

## Data Structure

- **Format**: [Standard](https://huggingface.co/docs/trl/main/dataset_formats#standard)
- **Type**: [Stepwise supervision](https://huggingface.co/docs/trl/main/dataset_formats#stepwise-supervision)

Columns:
- `"prompt"`: The problem statement.
- `"completions"`: A list of reasoning steps generated to solve the problem.
- `"labels"`: A list of booleans or floats indicating the correctness of each corresponding reasoning step.

This structure allows models to learn the correctness of each step in a solution, facilitating improved reasoning and problem-solving abilities.

## Generation script

The script used to generate this dataset can be found [here](https://github.com/huggingface/trl/blob/main/examples/datasets/prm800k.py).
""")

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    data_files = {
        "train": "MIL/MILdata/PRM800K_raw/data/data_raw/prm800k/data/phase2_train.jsonl",
        "test": "MIL/MILdata/PRM800K_raw/data/data_raw/prm800k/data/phase2_test.jsonl",
    }
    dataset = load_dataset("json", data_files=data_files)

    dataset = dataset.filter(
        keep_example,
        num_proc=script_args.dataset_num_proc,
    )

    dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=10,
        remove_columns=[
            "labeler",
            "timestamp",
            "generation",
            "is_quality_control_question",
            "is_initial_screening_question",
            "question",
            "label",
        ],
        num_proc=script_args.dataset_num_proc,
    )

    if script_args.save_to_disk is not None:
        dataset.save_to_disk(script_args.save_to_disk)

    if script_args.push_to_hub:
        dataset.push_to_hub(script_args.repo_id)
        model_card.push_to_hub(script_args.repo_id, repo_type="dataset")