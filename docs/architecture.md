# PaddleOCR-VL + ERNIE 4.5 Reproducible Document Intelligence Architecture

## Goal

This project uses PaddleOCR-VL/PaddleOCR remote layout parsing as the evidence
layer and ERNIE 4.5 as the evidence-constrained reasoning layer. PyFi finance
document understanding is the first validation domain, but the pipeline is
designed for reusable document intelligence across contracts, invoices,
research papers, government documents, technical documents, and other domains.

## Data Flow

```text
PDF/Image
  -> Document Ingestion
  -> PaddleOCR-VL Remote Layout Parsing
  -> Evidence Store
  -> Question Router / Profile Selection
  -> OCR Preset Resolver / LayoutOptionsFactory
  -> Domain Adapter
  -> ERNIE 4.5 Reasoning Agent
  -> Validation Layer
  -> Markdown/JSON Export
  -> Evaluation and Reproducibility Records
```

## Module Boundaries

- Document Ingestion: file type detection, input hashing, stable run directory.
- OCR Layout Parsing: remote PaddleOCR API call and raw result persistence.
- Evidence Store: OCR markdown, images, raw JSON, run metadata, prompt and output.
- Question Router: selects a task profile based on capability, question style,
  and option structure.
- OCR Preset Resolver / LayoutOptionsFactory: maps profiles to light, medium,
  or heavy PaddleOCR configurations while allowing explicit per-run overrides.
- Domain Adapter: prompt, schema, forbidden claims, domain tasks and evaluation notes.
- Reasoning Agent: ERNIE 4.5 call over OCR evidence and optional user context.
- Validation Layer: path safety, API errors, missing evidence, uncertainty handling,
  and chart consistency checks between OCR chart-to-table output and original
  image evidence.
- Export Layer: markdown, JSON and architecture documentation.

## Chart Consistency

For chart documents, PaddleOCR chart-to-table output is treated as reconstructed
evidence rather than unquestionable ground truth. When original image evidence
is available, the reasoning agent must compare:

- legend colors and labels;
- line, bar, or area colors and styles;
- visible trend direction and inflection points;
- axis labels and date/category order;
- OCR table values and image-visible values.

If OCR table values conflict with the visible chart, the output must mark
`chart_consistency.status` as `needs_human_review`, `inconsistent`, or
`uncertain`, and should not overstate the conclusion. Visual questions about
color, shape, line style, or layout should prioritize the original image while
using OCR as supporting structure.

## Reproducibility

Each run should preserve input hash, API URL, model name, domain adapter, layout
options, prompt, OCR raw JSON and model output. Benchmark improvements must be
based on academically accepted practices such as validation-set error analysis,
ablation, schema improvement and evidence-grounded prompting. Test labels,
hidden answers and benchmark leakage must not be used.

For PyFi evaluation, create stratified manifests from the public CSV using a
fixed seed and `capability` as the default stratum. The runner keeps held-out
fields such as actions and victory counts out of the model prompt, writes one
JSONL status record per sample, and supports resume so long evaluations can be
continued without rerunning completed samples.

The evaluation runner defaults to an `auto` profile, short structured answers,
and web search disabled so that benchmark outputs are driven by dataset
evidence rather than external retrieval.

## Domain Migration

To add a new domain, create a YAML adapter under `src/paddle_pyfi/domains_data/`
or pass a custom adapter path to the CLI. The core PaddleOCR client, ERNIE
client and pipeline should not be copied for each domain.
