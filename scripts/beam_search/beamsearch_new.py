import argparse
import json
import logging
import os
import re
import sys
from logging.handlers import QueueHandler, QueueListener
from queue import Empty
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
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


@dataclass(frozen=True)
class PromptItem:
	idx: int
	prompt: str
	reference: str


@dataclass
class Beam:
	prompt: PromptItem
	steps: List[str] = field(default_factory=list)
	score: Optional[float] = None
	completed: bool = False
	response: str = ""
	vllm_input: str = ""


@dataclass(eq=False)
class BeamGroup:
	prompt: PromptItem
	beams: List[Beam] = field(default_factory=list)
	round_id: int = 0


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
	parser.add_argument("--max_tokens", type=int, default=1024)
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
	parser.add_argument(
		"--batch_size",
		type=int,
		default=16,
		help="Batch size for vLLM generation",
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
		"--cuda_visible_devices",
		type=str,
		default=None,
		help=(
			"Comma-separated GPU indices to expose via CUDA_VISIBLE_DEVICES "
			"(e.g., '0,1'). If set, shard_id maps to this list."
		),
	)
	parser.add_argument(
		"--num_producers",
		type=int,
		default=1,
		help="Number of producer workers",
	)
	parser.add_argument(
		"--num_consumers",
		type=int,
		default=1,
		help="Number of consumer workers",
	)
	parser.add_argument("--cache_dir", type=str, default=None)

	return parser.parse_args()


def setup_main_logger() -> logging.Logger:
	logger = logging.getLogger("beamsearch")
	logger.setLevel(logging.INFO)
	handler = logging.StreamHandler()
	formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
	handler.setFormatter(formatter)
	logger.handlers = [handler]
	logger.propagate = False
	return logger


def setup_worker_logger(log_queue) -> logging.Logger:
	logger = logging.getLogger("beamsearch.worker")
	logger.setLevel(logging.INFO)
	logger.handlers = [QueueHandler(log_queue)]
	logger.propagate = False
	return logger


def silence_worker_output() -> None:
	sys.stdout = open(os.devnull, "w")
	sys.stderr = open(os.devnull, "w")


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
		user_content = f"{prompt}\n\n{instruction}"
		if steps_text:
			messages = [
				{"role": "user", "content": user_content},
				{"role": "assistant", "content": steps_text},
			]
			formatted = tokenizer.apply_chat_template(
				messages,
				tokenize=False,
				add_generation_prompt=False,
			)
			last_index = formatted.rfind(steps_text)
			if last_index != -1:
				return formatted[: last_index + len(steps_text)]
			return formatted
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
	dataset = TokenizedDocumentDataset(
		samples,
		tokenizer,
		max_length=max_length,
		separator=separator,
		apply_chat_template=apply_chat_template,
	)
	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		drop_last=False,
		collate_fn=collator,
	)
	scores: List[float] = []
	model.eval()
	with torch.no_grad():
		for batch in tqdm(dataloader, desc="Scoring with PRM", unit="batch"):
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


def producer_worker(
	producer_id: int,
	args_dict: Dict[str, Any],
	input_queue,
	output_queue,
	log_queue,
) -> None:
	# silence_worker_output()
	logger = setup_worker_logger(log_queue)
	args = argparse.Namespace(**args_dict)
	if args.cuda_visible_devices:
		visible = [item.strip() for item in args.cuda_visible_devices.split(",") if item.strip()]
		if visible:
			device_id = visible[producer_id % len(visible)]
			os.environ["CUDA_VISIBLE_DEVICES"] = device_id
		else:
			os.environ["CUDA_VISIBLE_DEVICES"] = str(producer_id)
	else:
		os.environ["CUDA_VISIBLE_DEVICES"] = str(producer_id)

	llm = build_llm(args)
	llm_tokenizer = AutoTokenizer.from_pretrained(
		args.model,
		trust_remote_code=args.trust_remote_code,
		cache_dir=args.cache_dir,
	)
	if args.chat_template is not None:
		llm_tokenizer.chat_template = args.chat_template

	sampling_params = build_sampling_params(args, args.step_separator)
	instruction = build_instruction(args.step_separator)

	logger.info("[Producer %s] Initialized and waiting for input...", producer_id)

	while True:
		# fetch all the groups in input_queue
		first_group = input_queue.get()
		if first_group is None:
			output_queue.put(None)
			break

		groups: List[BeamGroup] = [first_group]
		while True:
			try:
				next_group = input_queue.get_nowait()
			except Empty:
				break
			if next_group is None:
				output_queue.put(None)
				return
			groups.append(next_group)

		# flatten all active beams
		candidate_inputs: List[str] = []
		beam_refs: List[Beam] = []
		group_refs: List[BeamGroup] = []
		pending_counts: Dict[BeamGroup, int] = {}
		for group in groups:
			active_beams = [beam for beam in group.beams if not beam.completed]
			if not active_beams:
				output_queue.put(group)
				continue
			pending_counts[group] = len(active_beams)
			prompt_text = group.prompt.prompt
			for beam in active_beams:
				beam.vllm_input = build_candidate_prompts(
					prompt=prompt_text,
					steps=beam.steps,
					instruction=instruction,
					step_separator=args.step_separator,
					use_chat_template=args.use_chat_template,
					tokenizer=llm_tokenizer,
				)
				candidate_inputs.append(beam.vllm_input)
				beam_refs.append(beam)
				group_refs.append(group)

		new_beams_by_group: Dict[BeamGroup, List[Beam]] = {}
		batch_size = max(1, int(args.batch_size))
		for start in range(0, len(candidate_inputs), batch_size):
			batch_inputs = candidate_inputs[start : start + batch_size]
			batch_beams = beam_refs[start : start + batch_size]
			batch_groups = group_refs[start : start + batch_size]
			generated = generate_next_steps(llm, batch_inputs, sampling_params)
			for beam, group, responses in zip(batch_beams, batch_groups, generated):
				for response in responses:
					step_text = response.strip()
					if not step_text:
						continue
					new_steps = list(beam.steps) + [step_text]
					full_response = args.step_separator.join(new_steps)
					completed = extract_boxed(full_response) is not None
					new_beams_by_group.setdefault(group, []).append(
						Beam(
							prompt=beam.prompt,
							steps=new_steps,
							score=None,
							completed=completed,
							response=full_response,
							vllm_input="",
						)
					)
				pending_counts[group] -= 1
				if pending_counts[group] == 0:
					new_beams = new_beams_by_group.get(group, [])
					if new_beams:
						group.beams = [beam for beam in group.beams if beam.completed] + new_beams
						group.round_id += 1
					output_queue.put(group)
					logger.info(
						"[Producer %s] Round [%s] Group %s expanded.",
						producer_id,
						group.round_id,
						group.prompt.idx,
					)


def consumer_worker(
	consumer_id: int,
	args_dict: Dict[str, Any],
	input_queue,
	output_queue,
	log_queue,
) -> None:
	silence_worker_output()
	logger = setup_worker_logger(log_queue)
	args = argparse.Namespace(**args_dict)
	if args.cuda_visible_devices:
		visible = [item.strip() for item in args.cuda_visible_devices.split(",") if item.strip()]
		if visible:
			global_id = args.num_producers + consumer_id
			device_id = visible[global_id % len(visible)]
			os.environ["CUDA_VISIBLE_DEVICES"] = device_id
		else:
			os.environ["CUDA_VISIBLE_DEVICES"] = str(args.num_producers + consumer_id)
	else:
		os.environ["CUDA_VISIBLE_DEVICES"] = str(args.num_producers + consumer_id)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	prm_tokenizer = AutoTokenizer.from_pretrained(
		args.prm_model,
		trust_remote_code=args.trust_remote_code,
		cache_dir=args.cache_dir,
	)
	if prm_tokenizer.pad_token_id is None:
		prm_tokenizer.pad_token_id = prm_tokenizer.eos_token_id

	prm_model = build_prm_model(args, device)
	prm_model.config.pad_token_id = prm_tokenizer.pad_token_id

	logger.info("[Consumer %s] Initialized and waiting for input...", consumer_id)

	while True:
		first_group = input_queue.get()
		if first_group is None:
			output_queue.put(None)
			break

		groups: List[BeamGroup] = [first_group]
		while True:
			try:
				next_group = input_queue.get_nowait()
			except Empty:
				break
			if next_group is None:
				output_queue.put(None)
				return
			groups.append(next_group)

		beams_to_score: List[Beam] = []
		prompt_texts: List[str] = []
		candidate_steps: List[List[str]] = []
		group_refs: List[BeamGroup] = []
		pending_counts: Dict[BeamGroup, int] = {}
		for group in groups:
			unscored = [beam for beam in group.beams if beam.score is None]
			if not unscored:
				group.beams.sort(key=lambda beam: beam.score, reverse=True)
				group.beams = group.beams[: args.beam_size]
				output_queue.put(group)
				continue
			pending_counts[group] = len(unscored)
			for beam in unscored:
				beams_to_score.append(beam)
				prompt_texts.append(group.prompt.prompt)
				candidate_steps.append(beam.steps)
				group_refs.append(group)

		chunk_size = max(1, int(args.prm_batch_size))
		for start in range(0, len(beams_to_score), chunk_size):
			batch_beams = beams_to_score[start : start + chunk_size]
			batch_prompts = prompt_texts[start : start + chunk_size]
			batch_steps = candidate_steps[start : start + chunk_size]
			batch_groups = group_refs[start : start + chunk_size]
			scores = score_with_prm(
				prm_model,
				prm_tokenizer,
				batch_prompts,
				batch_steps,
				batch_size=args.prm_batch_size,
				max_length=args.prm_max_length,
				separator=args.prm_separator,
				apply_chat_template=args.prm_apply_chat_template,
				device=device,
			)
			for beam, group, score in zip(batch_beams, batch_groups, scores):
				beam.score = float(score)
				pending_counts[group] -= 1
				if pending_counts[group] == 0:
					group.beams.sort(key=lambda item: item.score or float("-inf"), reverse=True)
					group.beams = group.beams[: args.beam_size]
					output_queue.put(group)
					logger.info(
						"[Consumer %s] Round [%s] Group %s scored.",
						consumer_id,
						group.round_id,
						group.prompt.idx,
					)


def build_initial_groups(prompts: Sequence[PromptItem]) -> List[BeamGroup]:
	groups: List[BeamGroup] = []
	for item in prompts:
		groups.append(
			BeamGroup(
				prompt=item,
				beams=[Beam(prompt=item)],
				round_id=0,
			)
		)
	return groups


def all_groups_completed(groups: Sequence[BeamGroup]) -> bool:
	for group in groups:
		if not group.beams:
			return False
		if any(not beam.completed for beam in group.beams):
			return False
	return True


def group_completed(group: BeamGroup) -> bool:
	return bool(group.beams) and all(beam.completed for beam in group.beams)


def controller_main() -> None:
	logger = setup_main_logger()
	args = parse_args()
	prompts = load_prompts(
		dataset_name=args.dataset_name,
		dataset_config=args.dataset_config,
		split=args.split,
		prompt_column=args.prompt_column,
		answer_column=args.answer_column,
		max_examples=args.max_examples,
		cache_dir=args.cache_dir,
	)
	groups = build_initial_groups(prompts)
	if not groups:
		return
	if args.cuda_visible_devices:
		visible = [item.strip() for item in args.cuda_visible_devices.split(",") if item.strip()]
		required = max(1, args.num_producers) + max(1, args.num_consumers)
		if visible and len(visible) < required:
			raise ValueError(
				"Not enough CUDA devices for exclusive assignment: "
				f"required={required}, available={len(visible)}"
			)

	ctx = get_context("spawn")
	log_queue = ctx.Queue()
	log_listener = QueueListener(log_queue, *logger.handlers)
	log_listener.start()
	producer_output = ctx.Queue()
	consumer_output = ctx.Queue()
	producer_inputs = [ctx.Queue() for _ in range(max(1, args.num_producers))]
	consumer_inputs = [ctx.Queue() for _ in range(max(1, args.num_consumers))]

	args_dict = vars(args)
	producer_args = dict(args_dict)
	consumer_args = dict(args_dict)

	producers = []
	for producer_id in range(max(1, args.num_producers)):
		process = ctx.Process(
			target=producer_worker,
			args=(producer_id, producer_args, producer_inputs[producer_id], producer_output, log_queue),
		)
		process.start()
		producers.append(process)

	consumers = []
	for consumer_id in range(max(1, args.num_consumers)):
		process = ctx.Process(
			target=consumer_worker,
			args=(consumer_id, consumer_args, consumer_inputs[consumer_id], consumer_output, log_queue),
		)
		process.start()
		consumers.append(process)

	producer_index = 0
	consumer_index = 0
	producer_buffer = [group for group in groups]
	consumer_buffer = []
	completed_groups = []
	while len(completed_groups) < len(groups):
		# distribute producer_buffer to producers
		while producer_buffer:
			group = producer_buffer.pop()
			if group is None:
				continue
			if group_completed(group) or group.round_id >= args.max_steps:
				completed_groups.append(group)
				logger.info(
					"[Controller] Round [%s] Group %s completed with %s beams. Active groups: %s, Completed groups: %s",
					group.round_id,
					group.prompt.idx,
					len(group.beams),
					len(groups) - len(completed_groups),
					len(completed_groups)
				)
				continue
			producer_inputs[producer_index].put(group)
			logger.info(
				"[Controller] Round [%s] Dispatched group %s to producer %s",
				group.round_id,
				group.prompt.idx,
				producer_index,
			)
			producer_index = (producer_index + 1) % args.num_producers
		# collect producer output
		while True:
			try:
				group = producer_output.get_nowait()
				consumer_buffer.append(group)
			except Empty:
				break
		# distribute consumer_buffer to consumers
		while consumer_buffer:
			group = consumer_buffer.pop()
			consumer_inputs[consumer_index].put(group)
			logger.info(
				"[Controller] Round [%s] Dispatched group %s to consumer %s",
				group.round_id,
				group.prompt.idx,
				consumer_index,
			)
			consumer_index = (consumer_index + 1) % args.num_consumers
		# collect consumer output
		while True:			
			try:
				group = consumer_output.get_nowait()
				producer_buffer.append(group)
			except Empty:
				break

	for queue in producer_inputs:
		queue.put(None)
	for queue in consumer_inputs:
		queue.put(None)
	for process in producers:
		process.join()
	for process in consumers:
		process.join()
	log_listener.stop()

	results: List[Dict[str, Any]] = []
	for group in completed_groups:
		item = group.prompt
		prompt_beams = group.beams
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
				"id": item.idx,
				"prompt": item.prompt,
				"reference": reference_boxed,
				"beams": beam_records,
				"best": best_beam,
			}
		)
	
	results.sort(key=lambda record: record["id"])
	output_dir = os.path.dirname(args.output_path)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)
	with open(args.output_path, "w", encoding="utf-8") as f:
		json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
	controller_main()

