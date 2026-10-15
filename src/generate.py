"""Generate short answers and persist token-level model statistics."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from load_data import DEFAULT_CONFIG_PATH, load_config, resolve_project_path


SYSTEM_PROMPT = "Answer the question with only a short answer. Do not explain."


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible generation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _prompt_inputs(
    question: str,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Format one question with the model's chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def _generated_tokens(token_ids: list[int], stop_token_ids: set[int]) -> list[int]:
    """Discard EOS and padding tokens from the generated suffix."""
    output: list[int] = []
    for token_id in token_ids:
        if token_id in stop_token_ids:
            break
        output.append(token_id)
    return output


def _primary_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
) -> tuple[str, list[int], list[float], list[float]]:
    """Generate a greedy answer with per-token log-probability and entropy."""
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    suffix = output.sequences[0, prompt_length:].tolist()
    stop_ids = {
        token_id
        for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id)
        if token_id is not None
    }
    token_ids = _generated_tokens(suffix, stop_ids)

    log_probs: list[float] = []
    entropies: list[float] = []
    for token_id, logits in zip(token_ids, output.scores, strict=False):
        token_logits = logits[0].float()
        token_log_probs = torch.log_softmax(token_logits, dim=-1)
        token_probs = token_log_probs.exp()
        log_probs.append(float(token_log_probs[token_id].cpu()))
        entropies.append(float(-(token_probs * token_log_probs).sum().cpu()))

    answer = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    return answer, token_ids, log_probs, entropies


def _sampled_answers(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    n_samples: int,
    temperature: float,
) -> list[str]:
    """Generate repeated stochastic answers for response-consistency analysis."""
    with torch.inference_mode():
        sequences = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=n_samples,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    return [
        answer.strip()
        for answer in tokenizer.batch_decode(
            sequences[:, prompt_length:],
            skip_special_tokens=True,
        )
    ]


def generate_answers(
    questions_path: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate one greedy answer and repeated sampled answers per question."""
    experiment = config["experiment"]
    max_new_tokens = int(experiment["max_new_tokens"])
    n_samples = int(experiment["n_samples"])
    temperature = float(experiment["sampling_temperature"])
    if max_new_tokens <= 0 or n_samples <= 0 or temperature <= 0:
        raise ValueError("Generation parameters must be positive.")

    set_seed(int(experiment["seed"]))
    model, tokenizer = load_model_and_tokenizer(config["model"]["name"])
    questions = _read_jsonl(questions_path)
    device = next(model.parameters()).device

    records: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        inputs = _prompt_inputs(question["question"], tokenizer, device)
        answer, token_ids, token_log_probs, token_entropies = _primary_generation(
            model, tokenizer, inputs, max_new_tokens
        )
        samples = _sampled_answers(
            model, tokenizer, inputs, max_new_tokens, n_samples, temperature
        )
        records.append(
            {
                "id": question["id"],
                "question": question["question"],
                "gold_aliases": question["gold_aliases"],
                "answer": answer,
                "generated_token_ids": token_ids,
                "token_log_probs": token_log_probs,
                "token_entropies": token_entropies,
                "sampled_answers": samples,
            }
        )
        print(f"Generated {index}/{len(questions)}", end="\r", flush=True)

    print()
    return records


def save_generations(records: list[dict[str, Any]], output_path: Path) -> None:
    """Save generations and token-level statistics as JSON Lines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    """Generate and save answers for the configured question file."""
    args = parse_args()
    config = load_config(args.config)
    questions_path = resolve_project_path(config["outputs"]["questions"])
    output_path = resolve_project_path(config["outputs"]["generations"])
    records = generate_answers(questions_path, config)
    save_generations(records, output_path)
    print(f"Saved {len(records)} generation records to {output_path}")


if __name__ == "__main__":
    main()
