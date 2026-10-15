"""Generate short answers and persist token-level model statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from load_data import DEFAULT_CONFIG_PATH, load_config, resolve_project_path


SYSTEM_PROMPT = (
    "Answer each question with only the short answer, using at most a few words. "
    "Do not explain your answer."
)

# These examples are fixed independently of the sampled TriviaQA evaluation set.
FEW_SHOT_EXAMPLES = (
    ("What is the capital of Japan?", "Tokyo"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What is the chemical symbol for gold?", "Au"),
)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible generation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _question_seed(base_seed: int, question_id: str) -> int:
    """Derive a stable per-question seed so resumed runs remain reproducible."""
    digest = hashlib.sha256(f"{base_seed}:{question_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def load_model_and_tokenizer(
    model_name: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal language model and tokenizer on the available device."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read records from a UTF-8 JSON Lines file."""
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _existing_ids(path: Path) -> set[str]:
    """Read completed question IDs from an existing generations file."""
    if not path.exists():
        return set()

    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                completed.add(str(record["id"]))
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(
                    f"Invalid JSONL record at {path}:{line_number}."
                ) from error
    return completed


def _validate_few_shot_examples(questions: list[dict[str, Any]]) -> None:
    """Ensure no fixed demonstration question appears in the evaluation subset."""
    evaluation_questions = {item["question"].strip().casefold() for item in questions}
    overlap = {
        question
        for question, _ in FEW_SHOT_EXAMPLES
        if question.strip().casefold() in evaluation_questions
    }
    if overlap:
        raise ValueError(f"Few-shot examples overlap with evaluation questions: {overlap}")


def _prompt_inputs(
    question: str,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build and tokenize a three-example few-shot chat prompt."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_question, example_answer in FEW_SHOT_EXAMPLES:
        messages.extend(
            [
                {"role": "user", "content": example_question},
                {"role": "assistant", "content": example_answer},
            ]
        )
    messages.append({"role": "user", "content": question})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def _content_token_count(
    generated_ids: list[int],
    tokenizer: PreTrainedTokenizerBase,
) -> int:
    """Return the number of answer tokens before EOS, padding, or newline."""
    stop_token_ids = {
        token_id
        for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id)
        if token_id is not None
    }
    kept_ids: list[int] = []
    for token_id in generated_ids:
        if token_id in stop_token_ids:
            break
        candidate_ids = [*kept_ids, token_id]
        if "\n" in tokenizer.decode(candidate_ids, skip_special_tokens=True):
            break
        kept_ids.append(token_id)
    return len(kept_ids)


def _primary_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
) -> tuple[str, list[dict[str, int | float | str]]]:
    """Generate greedily and collect aligned token log-probabilities and entropies."""
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            stop_strings=["\n"],
            tokenizer=tokenizer,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output.sequences[0, prompt_length:].tolist()
    content_length = _content_token_count(generated_ids, tokenizer)
    content_ids = generated_ids[:content_length]

    token_statistics: list[dict[str, int | float | str]] = []
    aligned_scores = output.scores[:content_length]
    for token_id, step_scores in zip(content_ids, aligned_scores, strict=True):
        logits = step_scores[0].float()
        log_distribution = torch.log_softmax(logits, dim=-1)
        distribution = log_distribution.exp()
        token_statistics.append(
            {
                "token_id": token_id,
                "token_text": tokenizer.decode([token_id]),
                "log_probability": float(log_distribution[token_id].cpu()),
                "entropy": float(-(distribution * log_distribution).sum().cpu()),
            }
        )

    answer = tokenizer.decode(content_ids, skip_special_tokens=True).strip()
    return answer, token_statistics


def _sampled_answers(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    n_samples: int,
    temperature: float,
) -> list[str]:
    """Generate additional temperature-sampled answers from the same prompt."""
    with torch.inference_mode():
        sequences = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=n_samples,
            stop_strings=["\n"],
            tokenizer=tokenizer,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    answers: list[str] = []
    for sequence in sequences:
        generated_ids = sequence[prompt_length:].tolist()
        content_length = _content_token_count(generated_ids, tokenizer)
        answers.append(
            tokenizer.decode(
                generated_ids[:content_length],
                skip_special_tokens=True,
            ).strip()
        )
    return answers


def _append_record(record: dict[str, Any], output_path: Path) -> None:
    """Append one completed question to the generations JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def generate_answers(
    questions_path: Path,
    output_path: Path,
    config: dict[str, Any],
) -> int:
    """Generate missing answers and append each completed record to disk."""
    experiment = config["experiment"]
    max_new_tokens = int(experiment["max_new_tokens"])
    n_samples = int(experiment["n_samples"])
    temperature = float(experiment["sampling_temperature"])
    base_seed = int(experiment["seed"])
    if max_new_tokens <= 0 or n_samples <= 0 or temperature <= 0:
        raise ValueError("Generation parameters must be positive.")

    questions = _read_jsonl(questions_path)
    _validate_few_shot_examples(questions)
    completed_ids = _existing_ids(output_path)
    pending = [item for item in questions if str(item["id"]) not in completed_ids]
    if not pending:
        print(f"All {len(questions)} questions are already present in {output_path}")
        return 0

    model, tokenizer = load_model_and_tokenizer(config["model"]["name"])
    device = next(model.parameters()).device

    for question in tqdm(pending, desc="Generating answers", unit="question"):
        question_id = str(question["id"])
        set_seed(_question_seed(base_seed, question_id))
        inputs = _prompt_inputs(question["question"], tokenizer, device)
        answer, token_statistics = _primary_generation(
            model, tokenizer, inputs, max_new_tokens
        )
        sampled_answers = _sampled_answers(
            model, tokenizer, inputs, max_new_tokens, n_samples, temperature
        )
        _append_record(
            {
                "id": question_id,
                "question": question["question"],
                "gold_aliases": question["gold_aliases"],
                "answer": answer,
                "tokens": token_statistics,
                "sampled_answers": sampled_answers,
            },
            output_path,
        )
    return len(pending)


def parse_args() -> argparse.Namespace:
    """Parse the configuration path from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML experiment configuration.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate and incrementally save answers for configured questions."""
    args = parse_args()
    config = load_config(args.config)
    questions_path = resolve_project_path(config["outputs"]["questions"])
    output_path = resolve_project_path(config["outputs"]["generations"])
    generated_count = generate_answers(questions_path, output_path, config)
    print(f"Added {generated_count} generation records to {output_path}")


if __name__ == "__main__":
    main()
