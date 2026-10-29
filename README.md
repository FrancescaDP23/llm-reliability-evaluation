# Evaluating Confidence and Reliability in Open-Source Language Models

## Status

**Working prototype.** The repository currently provides a reproducible TriviaQA
sampling stage, few-shot answer generation with
`Qwen/Qwen2.5-1.5B-Instruct`, token-level log-probability collection, mean token
probability, and a PyTorch implementation of single-parameter temperature
scaling. A 20-question generation run has completed successfully on Google
Colab and produced resumable JSONL output. Unit tests cover the confidence
calculation and temperature scaler.

This is not yet a completed research study. The full 500-question experiment,
the remaining reliability metrics, statistical analysis, plots, and final
conclusions have not been run or implemented. Generated data are intentionally
not committed to Git.

## Research question

Can token probabilities and response consistency predict when an open-source
language model gives an incorrect short answer?

## Methodology

The pipeline samples TriviaQA questions deterministically, prompts a small
instruction-tuned model with three fixed examples, and saves one greedy answer
plus repeated temperature-sampled answers. For the greedy answer it records each
generated token's log-probability and full-distribution entropy.

`src/confidence.py` computes the arithmetic mean token probability and the
geometric mean sequence confidence. It converts sequence confidence to binary
logits and learns one positive temperature on a deterministic calibration split
by minimizing binary cross-entropy with PyTorch. The remaining split is marked
as evaluation data and is not used to fit the temperature.

## Models and datasets

- Model: `Qwen/Qwen2.5-1.5B-Instruct` from Hugging Face.
- Dataset: TriviaQA `rc.nocontext`, validation split.
- Quick test: 20 questions; planned experiment: 500 questions.

## Reproducibility

All experiment parameters and output paths are stored in YAML. Generation uses
fixed global and per-question seeds, and JSONL output supports resuming without
changing later sampled answers.

On Google Colab, select a GPU runtime and run:

```bash
git clone https://github.com/FrancescaDP23/llm-reliability-evaluation.git
cd llm-reliability-evaluation
pip install -r requirements.txt
python src/load_data.py --config configs/quick_test.yaml
python src/generate.py --config configs/quick_test.yaml
python src/confidence.py --config configs/quick_test.yaml
```

The quick run writes questions and generations as JSONL and confidence features
to `results/confidence_quick.csv`. If Colab contains an incompatible optional
`torchvision` installation, it can be removed with `pip uninstall -y torchvision`;
this text-only project does not require it.

Run unit tests from the repository root with:

```bash
pytest -q
```

## Results

The 20-question generation smoke test runs and saves its output. Quantitative
results are deliberately not reported yet: the complete experiment and formal
evaluation have not been performed.

## Figures

Not available yet.

## Limitations

- Exact match is a strict correctness proxy even after standard normalization.
- A 20-question smoke test is too small for conclusions about calibration.
- Mean token probability can be high for fluent but factually incorrect text.
- Temperature scaling calibrates confidence but cannot improve the model's answers.
- Only one small language model and one QA dataset are currently configured.

## Future work

- Run the complete 500-question experiment.
- Add response-consistency, ECE, Brier score, and error-detection AUROC.
- Report bootstrap confidence intervals and calibration plots.
- Compare raw and calibrated confidence on the held-out evaluation split.
