import argparse
import json
import os
import re
from multiprocessing import get_context
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer

try:
	from vllm import LLM, SamplingParams
except Exception as exc:  # pragma: no cover - runtime import guard
	raise RuntimeError(
		"vLLM is required for rollout generation. "
		"Please install vllm before running this script."
	) from exc


INSTRUCTION_TEMPLATE = (
	"Let's think step by step. Separate each step with "
	"{step_separator!r}, and output the final answer within \\boxed{{}}."
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate rollouts with vLLM")
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
	parser.add_argument(
		"--n",
		type=int,
		default=1,
		help="Number of rollouts per prompt",
	)
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
	parser.add_argument("--temperature", type=float, default=0.7)
	parser.add_argument("--top_p", type=float, default=0.95)
	parser.add_argument("--max_tokens", type=int, default=1024)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--tensor_parallel_size", type=int, default=1)
	parser.add_argument(
		"--num_gpus",
		type=int,
		default=1,
		help="Number of GPUs / processes for data-parallel rollouts",
	)
	parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
	parser.add_argument("--max_model_len", type=int, default=2048)
	parser.add_argument("--dtype", type=str, default="auto")
	parser.add_argument("--trust_remote_code", action="store_true")
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
	parser.add_argument(
		"--stop_strings",
		type=str,
		nargs="*",
		default=None,
		help="Optional stop strings for sampling",
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
	return [response.strip()]


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
) -> List[Tuple[int, str, str]]:
	dataset = load_dataset(
		dataset_name,
		dataset_config,
		split=split,
		cache_dir=cache_dir,
	)
	prompts: List[Tuple[int, str, str]] = []
	for idx, example in enumerate(dataset):
		prompt = coerce_to_str(example[prompt_column])
		reference = coerce_to_str(example[answer_column])
		prompts.append((idx, prompt, reference))
		if max_examples is not None and len(prompts) >= max_examples:
			break
	return prompts


def generate_rollouts(
	llm: LLM,
	prompts: Iterable[str],
	sampling_params: SamplingParams,
) -> List[List[str]]:
	outputs = llm.generate(list(prompts), sampling_params)
	generations: List[List[str]] = []
	for output in outputs:
		generations.append([item.text for item in output.outputs])
	return generations


def shard_list(items: List[Any], shard_id: int, num_shards: int) -> List[Any]:
	if num_shards <= 1:
		return items
	return items[shard_id::num_shards]


def worker_run(
	shard_id: int,
	num_shards: int,
	args_dict: Dict[str, Any],
	prompts_data: List[Tuple[int, str, str]],
	queue,
) -> None:
	os.environ["CUDA_VISIBLE_DEVICES"] = str(shard_id)
	args = argparse.Namespace(**args_dict)
	instruction = build_instruction(args.step_separator)

	tokenizer = AutoTokenizer.from_pretrained(
		args.model,
		trust_remote_code=args.trust_remote_code,
		cache_dir=args.cache_dir,
	)
	if args.chat_template is not None:
		tokenizer.chat_template = args.chat_template

	shard_data = shard_list(prompts_data, shard_id, num_shards)
	if not shard_data:
		queue.put([])
		return

	llm = LLM(
		model=args.model,
		tensor_parallel_size=args.tensor_parallel_size,
		gpu_memory_utilization=args.gpu_memory_utilization,
		max_model_len=args.max_model_len,
		dtype=args.dtype,
		trust_remote_code=args.trust_remote_code,
	)

	stop_token_ids = (
		[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None
	)
	sampling_params = SamplingParams(
		temperature=args.temperature,
		top_p=args.top_p,
		max_tokens=args.max_tokens,
		n=args.n,
		seed=args.seed + shard_id,
		stop=args.stop_strings,
		stop_token_ids=stop_token_ids,
		ignore_eos=False,
	)

	if args.use_chat_template:
		prompts_with_instruction = []
		for _, prompt, _ in shard_data:
			messages = [
				{
					"role": "user",
					"content": f"{prompt}\n\n{instruction}",
				}
			]
			prompt_text = tokenizer.apply_chat_template(
				messages,
				tokenize=False,
				add_generation_prompt=True,
			)
			prompts_with_instruction.append(prompt_text)
	else:
		prompts_with_instruction = [
			f"{prompt}\n\n{instruction}" for _, prompt, _ in shard_data
		]
	generations = generate_rollouts(llm, prompts_with_instruction, sampling_params)

	records: List[Dict[str, Any]] = []
	for (idx, prompt, reference), responses in zip(shard_data, generations):
		for response in responses:
			records.append(
				build_record(
					idx=idx,
					prompt=prompt,
					response=response,
					reference=reference,
					step_separator=args.step_separator,
				)
			)

	queue.put(records)


def build_record(
	idx: int,
	prompt: str,
	response: str,
	reference: str,
	step_separator: str,
) -> Dict[str, Any]:
	steps = split_steps(response, step_separator)
	extracted = extract_boxed(response)
	reference_boxed = extract_boxed(reference) or reference
	correctness = False
	if extracted is not None:
		correctness = normalize_text(extracted) == normalize_text(reference_boxed)
	return {
		"task": "math",
		"idx": idx,
		"prompt": prompt,
		"response": response,
		"steps": steps,
		"extracted_output": extracted,
		"reference": reference_boxed,
		"correctness": correctness,
	}


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

	output_dir = os.path.dirname(args.output_path)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)
	with open(args.output_path, "w", encoding="utf-8") as f:
		json.dump(records, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
	main()
