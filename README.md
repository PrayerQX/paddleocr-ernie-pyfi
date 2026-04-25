# Paddle PyFi

Paddle PyFi is a local `uv` managed project for a reproducible
PaddleOCR-VL/PaddleOCR remote OCR plus ERNIE 4.5 document intelligence
pipeline. The first domain adapter is `finance`, inspired by
`AgenticFinLab/PyFi`, and the same architecture can be reused for contracts,
invoices, research papers, government documents, and other document types.

The evaluation path now includes:

- a `QuestionRouter` that selects a profile by capability and question type;
- an `OCRPresetResolver` that maps the selected profile to `light`, `medium`,
  or `heavy` PaddleOCR option sets;
- short structured answers for benchmark scoring;
- chart consistency checks between OCR chart-to-table output and the original image;
- stratified manifest runs that keep held-out answer-like fields away from the model.

## Benchmark Summary

Two 301-sample PyFi runs were completed with the same ERNIE model but different
PaddleOCR settings:

- Round 1
  - OCR: PaddleOCR remote `layout-parsing`, effective preset `medium`
  - Key OCR settings: `useChartRecognition=true`, `useSealRecognition=false`,
    `mergeTables=false`, `relevelTitles=false`, `promptLabel="chart"`,
    `restructurePages=false`
  - LLM: `ernie-4.5-21b-a3b`
  - Result: `163/295` correct, accuracy `0.552542`

- Round 2
  - OCR: PaddleOCR remote `layout-parsing`, preset `baidu_sample`
  - Key OCR settings: `useChartRecognition=true`, `useSealRecognition=true`,
    `mergeTables=true`, `relevelTitles=true`, `promptLabel="ocr"`,
    `restructurePages=true`
  - LLM: `ernie-4.5-21b-a3b`
  - Result: `157/300` correct, accuracy `0.523333`

Main observations:

- Heavier OCR reduced missing predictions (`6 -> 1`) but lowered overall
  accuracy (`0.552542 -> 0.523333`).
- The largest drops were in `Calculation_analysis` and `Data_extraction`.
- The dominant failure mode is still loss of visual evidence during OCR
  textification, with a secondary class of errors where ERNIE misjudges even
  when the OCR evidence is already sufficient.

Detailed writeups:

- [Two-round results summary](docs/two_round_results_summary.md)
- [Error analysis summary](output-pyfi301-ernie45-baidu-sample-smoke-20260423/error_analysis_summary.md)

## Environment

All Python dependencies must stay in the project-local `.venv`.

```powershell
uv sync
```

Create local credentials from the template:

```powershell
Copy-Item .env.example .env
notepad .env
```

Do not commit `.env`, user documents, OCR JSON, downloaded images, or model
outputs.

## Commands

List available commands:

```powershell
uv run python -m paddle_pyfi --help
```

Inspect the PyFi-600K dataset metadata without downloading the full dataset:

```powershell
uv run python -m paddle_pyfi dataset-info
```

Download the core files needed for a first real run:

```powershell
uv run python -m paddle_pyfi dataset-download --selection core --output-dir data/pyfi-600k
```

Download all published dataset files:

```powershell
uv run python -m paddle_pyfi dataset-download --selection full --output-dir data/pyfi-600k
```

Run remote PaddleOCR layout parsing:

```powershell
uv run python -m paddle_pyfi parse .\samples\report.pdf --output-dir output
```

Run OCR and ERNIE 4.5 analysis with the finance adapter:

```powershell
uv run python -m paddle_pyfi analyze .\samples\report.pdf --domain finance --output-dir output
```

For image inputs, `analyze` sends both OCR evidence and the original image to
ERNIE by default. Use `--no-image` to force text-only analysis.

For chart inputs, the prompt asks ERNIE to compare PaddleOCR chart-to-table
evidence with the original image. If the reconstructed OCR table conflicts with
the visible chart trend, the answer should mark `chart_consistency.status` as
`needs_human_review`, `inconsistent`, or `uncertain`.

Use the auto router for benchmark-style runs:

```powershell
uv run python -m paddle_pyfi analyze .\samples\chart.jpg --domain finance --profile auto --ocr-preset auto --max-completion-tokens 8192 --no-web-search
```

Override the auto-selected OCR strength when needed:

```powershell
uv run python -m paddle_pyfi analyze .\samples\chart.jpg --domain finance --profile pattern_visual_consistency --ocr-preset heavy --prompt-label chart
```

Ask a question over an existing OCR run:

```powershell
uv run python -m paddle_pyfi ask .\output\report-abc123 --domain finance --question "提取关键财务指标并说明证据。"
```

Attach image evidence to an existing OCR run:

```powershell
uv run python -m paddle_pyfi ask .\output\report-abc123 --domain finance --question "图例中无负债类别是什么颜色？" --image .\samples\chart.jpg
```

Export the reusable architecture document:

```powershell
uv run python -m paddle_pyfi export-architecture
```

## PyFi-600K Dataset

The dataset lives at:

```text
https://huggingface.co/datasets/AgenticFinLab/PyFi-600K
```

It contains CSV/JSON question-answer files, chain datasets, CoT data, and
`images.zip`. The full dataset is large, so this project defaults to metadata
inspection and local manifest creation rather than automatic full download.

If you manually download `PyFi-600K-dataset.csv`, create a small manifest:

```powershell
uv run python -m paddle_pyfi manifest .\data\PyFi-600K-dataset.csv --limit 100 --output .\data\pyfi_manifest.json
```

Create a reproducible 301-sample stratified manifest by PyFi capability:

```powershell
uv run python -m paddle_pyfi manifest .\data\pyfi-600k\PyFi-600K-dataset.csv --stratify capability --sample-size 301 --seed 20260419 --exclude-stratum None --output .\data\pyfi-600k\pyfi301_manifest.json
```

Run the manifest with the best chart pipeline. The runner writes a JSONL status
record after each sample and resumes completed samples by default:

```powershell
uv run python -m paddle_pyfi run-manifest .\data\pyfi-600k\pyfi301_manifest.json --output-dir output-pyfi301 --domain finance --profile auto --ocr-preset auto --max-completion-tokens 8192
```

For smoke tests, use `--limit`:

```powershell
uv run python -m paddle_pyfi run-manifest .\data\pyfi-600k\pyfi301_manifest.json --output-dir output-pyfi301 --domain finance --profile auto --ocr-preset auto --max-completion-tokens 8192 --limit 3
```

The runner disables web search by default to keep evaluation grounded in the
dataset evidence. Add `--web-search` only when you explicitly want that.

Score a completed manifest run:

```powershell
uv run python -m paddle_pyfi score-manifest .\data\pyfi-600k\pyfi301_manifest.json --run-output-dir output-pyfi301
```

## Output Structure

Each input file creates a stable run directory:

```text
output/
└── report-<sha12>/
    ├── ocr_result.json
    ├── doc_0.md
    ├── markdown_images/
    ├── output_images/
    ├── run_meta.json
    ├── prompt.md
    ├── analysis_finance.md
    └── analysis_finance.json
```

## Domains

Bundled domain adapters:

```powershell
uv run python -m paddle_pyfi domains
```

The core pipeline is domain-neutral. Add a domain by creating a YAML adapter
with document types, tasks, output schema, forbidden claims, instructions, and
evaluation notes.

## Tests

```powershell
uv run pytest
```

Unit tests must not call the real PaddleOCR or ERNIE APIs. Real integration
tests should be opt-in and controlled by environment variables.
