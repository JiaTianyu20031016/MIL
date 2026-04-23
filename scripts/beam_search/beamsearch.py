import argparse
import json
import os
import re
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from MILdata.dataset_common import (
	DocumentSample,
	Segment,
	TokenizedDocumentDataset,
	build_document_dataloader,
)
from MILdata.collator import MILDataCollator
from MILmodel.mil_model_for_prm import ARCHITECTURE_TO_MODEL_CLASS

try:
	from vllm import LLM, SamplingParams
except Exception as exc:  # pragma: no cover - runtime import guard
	raise RuntimeError(
		"vLLM is required for beam search generation. "
		"Please install vllm before running this script."
	) from exc


INSTRUCTION_TEMPLATE = (
	"Let's think step by step. Separate each step with "
	"{step_separator!r}, and output the final answer within \\boxed{{}}."
)



@dataclass
class PromptItem:
	idx: int
	prompt: str
	reference: str

@dataclass
class Beam:
	prompt: PromptItem = None  # the raw question
	steps: List[str] = []    # the generated reasoning steps so far, without separators
	score: float = 0.0        # the PRM score for the current set of steps (higher is better)
	completed: bool = False     # whether the LLM has indicated completion (e.g., by outputting a boxed answer)
	response: str = ""       # the full generated response from the LLM for the current set of steps, including separators and instructions
	vllm_input: str = ""     # the final input string that was fed into vLLM to generate the next step


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Beam search with PRM scoring")

	# dataset args
	parser.add_argument(
		"--dataset_name",
		type=str,
		default="HuggingFaceH4/MATH-500",
		help="HuggingFace dataset name",
	)
	parser.add_argument(
		"--dataset_config",
		type=str,
		default=None,
		help="Optional dataset config name",
	)
	parser.add_argument(
		"--split",
		type=str,
		default="test",
		help="Dataset split to load",
	)
	parser.add_argument(
		"--prompt_column",
		type=str,
		default="problem",
		help="Column containing the prompt/problem",
	)
	parser.add_argument(
		"--answer_column",
		type=str,
		default="answer",
		help="Column containing the reference answer",
	)

	# common model args
	parser.add_argument("--dtype", type=str, default="auto")
	parser.add_argument("--trust_remote_code", action="store_true")

	# vllm args
	parser.add_argument(
		"--step_separator",
		type=str,
		default="\n\n",
		help="Separator inserted between reasoning steps",
	)
	parser.add_argument(
		"--model",
		type=str,
		required=True,
		help="vLLM model name or local path",
	)
	parser.add_argument("--temperature", type=float, default=0.7)
	parser.add_argument("--top_p", type=float, default=0.95)
	parser.add_argument("--max_tokens", type=int, default=256)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--tensor_parallel_size", type=int, default=1)
	parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
	parser.add_argument("--max_model_len", type=int, default=2048)
	parser.add_argument(
		"--stop_strings",
		type=str,
		nargs="*",
		default=None,
		help="Optional stop strings for sampling",
	)
	parser.add_argument(
		"--use_chat_template",
		action="store_true",
		help="Apply tokenizer chat template before generation",
	)
	parser.add_argument(
		"--chat_template",
		type=str,
		default=None,
		help="Optional chat template override (Jinja2 string)",
	)

	# PRM args
	parser.add_argument(
		"--prm_model",
		type=str,
		required=True,
		help="PRM model name or local path",
	)
	parser.add_argument(
		"--prm_architecture",
		type=str,
		default="InstanceAveragePoolMILModelforPRM",
		help="PRM architecture name from ARCHITECTURE_TO_MODEL_CLASS",
	)
	parser.add_argument(
		"--prm_batch_size",
		type=int,
		default=8,
		help="Batch size for PRM scoring",
	)
	parser.add_argument(
		"--prm_max_length",
		type=int,
		default=None,
		help="Optional max length for PRM tokenization",
	)
	parser.add_argument(
		"--prm_separator",
		type=str,
		default="\n\n",
		help="Separator for PRM document construction",
	)
	parser.add_argument(
		"--prm_apply_chat_template",
		action="store_true",
		help="Apply chat template when constructing PRM documents",
	)

	# beam search args
	parser.add_argument(
		"--beam_size",
		type=int,
		default=4,
		help="Beam width for search",
	)
	parser.add_argument(
		"--expansion_per_beam",
		type=int,
		default=2,
		help="Number of candidate steps per beam expansion",
	)
	parser.add_argument(
		"--max_steps",
		type=int,
		default=8,
		help="Maximum reasoning steps for beam search",
	)

	# other args
	parser.add_argument(
		"--max_examples",
		type=int,
		default=None,
		help="Optional cap on number of examples",
	)
	parser.add_argument(
		"--output_path",
		type=str,
		required=True,
		help="Where to write the JSON output",
	)
	parser.add_argument(
		"--num_gpus",
		type=int,
		default=1,
		help="Number of GPUs / processes for data-parallel beam search",
	)
	parser.add_argument(
		"--memory_mode",
		type=str,
		default="resident",
		choices=["resident", "swap"],
		help="Keep vLLM+PRM on GPU (resident) or swap them per step (swap)",
	)
	parser.add_argument("--cache_dir", type=str, default=None)

	return parser.parse_args()


def build_instruction(step_separator: str) -> str:
	return INSTRUCTION_TEMPLATE.format(step_separator=step_separator)


def normalize_text(text: str) -> str:
	cleaned = text.strip()
	cleaned = cleaned.replace("$", "")
	cleaned = cleaned.replace("\\left", "").replace("\\right", "")
	cleaned = re.sub(r"\s+", "", cleaned)
	return cleaned


def extract_boxed(text: str) -> Optional[str]:
	if "\\boxed" not in text:
		return None
	start = text.find("\\boxed")
	brace_start = text.find("{", start)
	if brace_start == -1:
		return None
	depth = 0
	for idx in range(brace_start, len(text)):
		if text[idx] == "{":
			depth += 1
		elif text[idx] == "}":
			depth -= 1
			if depth == 0:
				return text[brace_start + 1 : idx]
	return None


def split_steps(response: str, step_separator: str) -> List[str]:
	if step_separator and step_separator in response:
		parts = [part.strip() for part in response.split(step_separator)]
		return [part for part in parts if part]
	return [response.strip()] if response.strip() else []


def coerce_to_str(value: Any) -> str:
	if isinstance(value, str):
		return value
	if isinstance(value, (list, tuple)) and value:
		return str(value[0])
	if isinstance(value, dict):
		for key in ("text", "value", "answer"):
			if key in value:
				return str(value[key])
		return json.dumps(value, ensure_ascii=False)
	return str(value)


def load_prompts(
	dataset_name: str,
	dataset_config: Optional[str],
	split: str,
	prompt_column: str,
	answer_column: str,
	max_examples: Optional[int],
	cache_dir: Optional[str],
) -> List[PromptItem]:
	dataset = load_dataset(
		dataset_name,
		dataset_config,
		split=split,
		cache_dir=cache_dir,
	)
	prompts: List[PromptItem] = []
	for idx, example in enumerate(dataset):
		prompt = coerce_to_str(example[prompt_column])
		reference = coerce_to_str(example[answer_column])
		prompts.append(PromptItem(idx=idx, prompt=prompt, reference=reference))
		if max_examples is not None and len(prompts) >= max_examples:
			break
	return prompts


def shard_list(items: List[Any], shard_id: int, num_shards: int) -> List[Any]:
	if num_shards <= 1:
		return items
	return items[shard_id::num_shards]


def build_candidate_prompts(
	prompt: str,
	steps: Sequence[str],
	instruction: str,
	step_separator: str,
	use_chat_template: bool,
	tokenizer: AutoTokenizer,
) -> str:
	steps_text = step_separator.join(steps)
	if steps_text:
		steps_text = f"{steps_text}{step_separator}"

	if use_chat_template:
		user_content = f"{prompt}\n\n{instruction}\n\n{steps_text}"
		messages = [{"role": "user", "content": user_content}]
		return tokenizer.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=True,
		)

	return f"{prompt}\n\n{instruction}\n\n{steps_text}"


def build_sampling_params(
	args: argparse.Namespace,
	step_separator: str,
) -> SamplingParams:
	stop_strings = list(args.stop_strings) if args.stop_strings else []
	if step_separator and step_separator not in stop_strings:
		stop_strings.append(step_separator)
	return SamplingParams(
		temperature=args.temperature,
		top_p=args.top_p,
		max_tokens=args.max_tokens,
		n=args.expansion_per_beam,
		seed=args.seed,
		stop=stop_strings if stop_strings else None,
	)


def generate_next_steps(
	llm: LLM,
	prompts: Iterable[str],
	sampling_params: SamplingParams,
) -> List[List[str]]:
	outputs = llm.generate(list(prompts), sampling_params)
	generations: List[List[str]] = []
	for output in outputs:
		generations.append([item.text for item in output.outputs])
	return generations


def build_document_samples(
	prompts: Sequence[str],
	step_lists: Sequence[List[str]],
) -> List[DocumentSample]:
	if len(prompts) != len(step_lists):
		raise ValueError("prompts and step_lists must have the same length.")
	samples: List[DocumentSample] = []
	for idx, (prompt, steps) in enumerate(zip(prompts, step_lists)):
		segments = [Segment(label=1, positive_prob=1.0, text=step) for step in steps]
		samples.append(
			DocumentSample(
				doc_id=str(idx),
				rating=1.0,
				positive_prob=1.0,
				prompt=prompt,
				segments=segments,
				granularity="step",
				source="beamsearch",
			)
		)
	return samples


def score_with_prm(
	model,
	tokenizer,
	prompts: Sequence[str],
	candidate_steps: Sequence[List[str]],
	*,
	batch_size: int,
	max_length: Optional[int],
	separator: Optional[str],
	apply_chat_template: bool,
	device: torch.device,
) -> List[float]:
	samples = build_document_samples(prompts, candidate_steps)
	collator = MILDataCollator(pad_token_id=tokenizer.pad_token_id)
	dataloader = build_document_dataloader(
		samples,
		tokenizer,
		batch_size=batch_size,
		shuffle=False,
		drop_last=False,
		max_length=max_length,
		collate_fn=collator,
	)
	scores: List[float] = []
	model.eval()
	with torch.no_grad():
		for batch in dataloader:
			batch = {
				key: value.to(device) if isinstance(value, torch.Tensor) else value
				for key, value in batch.items()
			}
			outputs = model(eval=True, **batch)
			doc_probs = outputs.document_probs
			if doc_probs is None or doc_probs.numel() == 0:
				batch_scores = torch.zeros((batch["input_ids"].size(0),), device=device)
			else:
				batch_scores = doc_probs[:, 1]
			scores.extend(batch_scores.detach().cpu().tolist())
	return scores


def beam_search_batch(
	llm_tokenizer: AutoTokenizer,
	prm_tokenizer: AutoTokenizer,
	items: Sequence[PromptItem],
	args: argparse.Namespace,
	device: torch.device,
	*,
	llm: Optional[LLM] = None,
	prm_model: Optional[torch.nn.Module] = None,
) -> List[Dict[str, Any]]:
	if not items:
		return []

	instruction = build_instruction(args.step_separator)
	sampling_params = build_sampling_params(args, args.step_separator)

	beams: List[Beam] = [
		Beam(prompt=item) for item in items
	]

	for _ in range(args.max_steps):
		# expand active beams
		active_beams: List[Beam] = []
		for beam in beams:
			if beam.completed:
				continue
			beam.vllm_input = build_candidate_prompts(
				prompt=beam.prompt.prompt,
				steps=beam.steps,
				instruction=instruction,
				step_separator=args.step_separator,
				use_chat_template=args.use_chat_template,
				tokenizer=llm_tokenizer,
			)
			active_beams.append(beam)

		if not active_beams:
			break

		if args.memory_mode == "swap":
			llm = build_llm(args)
		prompt_pool = [beam.vllm_input for beam in active_beams]
		generated = generate_next_steps(llm, prompt_pool, sampling_params)
		if args.memory_mode == "swap":
			del llm
			llm = None
			release_cuda_memory()

		new_beams: List[Beam] = []
		for beam, responses in zip(active_beams, generated):
			for response in responses:
				step_text = response.strip()
				if not step_text:
					continue
				new_steps = list(beam.steps) + [step_text]
				full_response = args.step_separator.join(new_steps)
				completed = extract_boxed(full_response) is not None
				new_beams.append(
					Beam(
						prompt=beam.prompt,
						steps=new_steps,
						score=0.0,
						completed=completed,
						response=full_response,
						vllm_input="",
					)
				)

		if not new_beams:
			break
        
        # score new beams
		if args.memory_mode == "swap":
			prm_model = build_prm_model(args, device)
			prm_model.config.pad_token_id = prm_tokenizer.pad_token_id
		pool_scores = score_with_prm(
			prm_model,
			prm_tokenizer,
			[beam.prompt.prompt for beam in new_beams],
			[beam.steps for beam in new_beams],
			batch_size=args.prm_batch_size,
			max_length=args.prm_max_length,
			separator=args.prm_separator,
			apply_chat_template=args.prm_apply_chat_template,
			device=device,
		)
		if args.memory_mode == "swap":
			del prm_model
			prm_model = None
			release_cuda_memory()
		for score, beam in zip(pool_scores, new_beams):
			beam.score = float(score)
		
        # sort and prune beams
		new_beams = [beam for beam in beams if beam.completed] + new_beams

		grouped: Dict[int, List[Beam]] = {item.idx: [] for item in items}
		for beam in new_beams:
			grouped[beam.prompt.idx].append(beam)

		beams = []
		for item in items:
			prompt_beams = grouped[item.idx]
			if not prompt_beams:
				beams.append(
		            Beam(prompt=item)
				)
				continue
			prompt_beams.sort(key=lambda beam: beam.score, reverse=True)
			beams.extend(prompt_beams[: args.beam_size])

		if all(beam.completed for beam in beams):
			break


	results: List[Dict[str, Any]] = []
	grouped_final: Dict[int, List[Beam]] = {item.idx: [] for item in items}
	for beam in beams:
		grouped_final[beam.prompt.idx].append(beam)

	for item in items:
		prompt_beams = grouped_final[item.idx]
		if not prompt_beams:
			prompt_beams = [
				Beam(prompt=item)
			]
		reference_boxed = extract_boxed(item.reference) or item.reference
		beam_records = []
		for beam in prompt_beams:
			extracted = extract_boxed(beam.response)
			correct = False
			if extracted is not None:
				correct = normalize_text(extracted) == normalize_text(reference_boxed)
			beam_records.append(
				{
					"steps": beam.steps,
					"response": beam.response,
					"score": beam.score,
					"extracted_output": extracted,
					"correctness": correct,
				}
			)

		best_beam = max(beam_records, key=lambda record: record["score"])
		results.append(
			{
				"prompt": item.prompt,
				"reference": reference_boxed,
				"beams": beam_records,
				"best": best_beam,
			}
		)

	return results


def build_prm_model(args: argparse.Namespace, device: torch.device):
	model_class = ARCHITECTURE_TO_MODEL_CLASS.get(args.prm_architecture)
	if model_class is None:
		raise ValueError(
			f"Unsupported prm_architecture '{args.prm_architecture}'. "
			f"Supported: {list(ARCHITECTURE_TO_MODEL_CLASS.keys())}"
		)
	model = model_class.from_pretrained(
		args.prm_model,
		trust_remote_code=args.trust_remote_code,
	)
	model.to(device)
	return model


def build_llm(args: argparse.Namespace) -> LLM:
	return LLM(
		model=args.model,
		tensor_parallel_size=args.tensor_parallel_size,
		gpu_memory_utilization=args.gpu_memory_utilization,
		max_model_len=args.max_model_len,
		dtype=args.dtype,
		trust_remote_code=args.trust_remote_code,
	)


def release_cuda_memory() -> None:
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.cuda.ipc_collect()


def worker_run(
	shard_id: int,
	num_shards: int,
	args_dict: Dict[str, Any],
	prompts_data: List[PromptItem],
	queue,
) -> None:
	os.environ["CUDA_VISIBLE_DEVICES"] = str(shard_id)
	args = argparse.Namespace(**args_dict)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	llm_tokenizer = AutoTokenizer.from_pretrained(
		args.model,
		trust_remote_code=args.trust_remote_code,
		cache_dir=args.cache_dir,
	)
	if args.chat_template is not None:
		llm_tokenizer.chat_template = args.chat_template

	prm_tokenizer = AutoTokenizer.from_pretrained(
		args.prm_model,
		trust_remote_code=args.trust_remote_code,
		cache_dir=args.cache_dir,
	)
	if prm_tokenizer.pad_token_id is None:
		prm_tokenizer.pad_token_id = prm_tokenizer.eos_token_id

	prm_model = None
	llm = None
	if args.memory_mode == "resident":
		prm_model = build_prm_model(args, device)
		prm_model.config.pad_token_id = prm_tokenizer.pad_token_id
		llm = build_llm(args)

	shard_data = shard_list(prompts_data, shard_id, num_shards)
	if not shard_data:
		queue.put([])
		return

	results = beam_search_batch(
		llm_tokenizer,
		prm_tokenizer,
		shard_data,
		args,
		device,
		llm=llm,
		prm_model=prm_model,
	)
	records = [
		{
			"task": "math",
			"idx": item.idx,
			**result,
		}
		for item, result in zip(shard_data, results)
	]

	queue.put(records)


def compute_accuracy(records: Sequence[Dict[str, Any]]) -> float:
	if not records:
		return 0.0
	correct = sum(1 for record in records if record["best"]["correctness"])
	return correct / len(records)


def main() -> None:
	args = parse_args()
	prompts_data = load_prompts(
		dataset_name=args.dataset_name,
		dataset_config=args.dataset_config,
		split=args.split,
		prompt_column=args.prompt_column,
		answer_column=args.answer_column,
		max_examples=args.max_examples,
		cache_dir=args.cache_dir,
	)

	if args.num_gpus <= 1:
		args_dict = vars(args)
		queue = get_context("spawn").Queue()
		worker_run(0, 1, args_dict, prompts_data, queue)
		records = queue.get()
	else:
		ctx = get_context("spawn")
		queue = ctx.Queue()
		processes = []
		args_dict = vars(args)
		for shard_id in range(args.num_gpus):
			process = ctx.Process(
				target=worker_run,
				args=(shard_id, args.num_gpus, args_dict, prompts_data, queue),
			)
			process.start()
			processes.append(process)

		records = []
		for _ in range(args.num_gpus):
			records.extend(queue.get())

		for process in processes:
			process.join()

	records = sorted(records, key=lambda item: item["idx"])
	accuracy = compute_accuracy(records)
	output_payload = {
		"accuracy": accuracy,
		"num_examples": len(records),
		"records": records,
	}

	output_dir = os.path.dirname(args.output_path)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)
	with open(args.output_path, "w", encoding="utf-8") as f:
		json.dump(output_payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
	main()
