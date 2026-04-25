from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .architecture import export_architecture
from .config import load_settings
from .dataset import (
    create_manifest_from_csv,
    create_stratified_manifest_from_csv,
    download_pyfi_files,
    fetch_pyfi_dataset_info,
)
from .domains import available_domains
from .file_types import is_supported_document
from .manifest_runner import run_manifest
from .ocr_presets import build_layout_options
from .pipeline import analyze_document, analyze_run_dir, parse_document
from .reporting import build_manifest_report
from .scoring import score_manifest_run


def _layout_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if getattr(args, "use_doc_orientation_classify", None) is not None:
        overrides["use_doc_orientation_classify"] = args.use_doc_orientation_classify
    if getattr(args, "use_doc_unwarping", None) is not None:
        overrides["use_doc_unwarping"] = args.use_doc_unwarping
    if getattr(args, "use_chart_recognition", None) is not None:
        overrides["use_chart_recognition"] = args.use_chart_recognition
    if getattr(args, "prompt_label", "auto") != "auto":
        overrides["prompt_label"] = args.prompt_label
    return overrides


def _add_ocr_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ocr-preset", default="auto", choices=["auto", "light", "medium", "heavy", "baidu_sample"])
    parser.add_argument("--use-doc-orientation-classify", action="store_true", default=None)
    parser.add_argument("--use-doc-unwarping", action="store_true", default=None)
    parser.add_argument("--use-chart-recognition", action="store_true", default=None)
    parser.add_argument("--prompt-label", default="auto", choices=["auto", "ocr", "chart", "table", "formula", "seal"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paddle-pyfi",
        description="PaddleOCR-VL + ERNIE 4.5 reproducible document intelligence pipeline.",
    )
    parser.add_argument("--env-file", default=None, help="Optional .env file path.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse", help="Run remote PaddleOCR layout parsing.")
    parse_parser.add_argument("file", help="PDF or image file.")
    parse_parser.add_argument("--output-dir", default="output")
    _add_ocr_options(parse_parser)

    analyze_parser = subparsers.add_parser("analyze", help="Parse and analyze a document.")
    analyze_parser.add_argument("file", help="PDF or image file.")
    analyze_parser.add_argument("--output-dir", default="output")
    analyze_parser.add_argument("--domain", default="finance", help="Domain adapter name or YAML path.")
    analyze_parser.add_argument(
        "--profile",
        default="auto",
        choices=["auto", "perception_visual", "data_extraction_precise", "calculation_formula", "pattern_visual_consistency", "logical_options", "decision_support_evidence"],
    )
    analyze_parser.add_argument("--question", default=None)
    analyze_parser.add_argument("--context", default=None)
    analyze_parser.add_argument("--max-completion-tokens", type=int, default=8192)
    analyze_parser.add_argument("--no-image", action="store_true", help="Do not send image inputs to ERNIE.")
    analyze_parser.add_argument("--no-web-search", action="store_true")
    _add_ocr_options(analyze_parser)

    ask_parser = subparsers.add_parser("ask", help="Ask over an existing OCR run directory.")
    ask_parser.add_argument("run_dir", help="Directory containing doc_*.md from parse.")
    ask_parser.add_argument("--domain", default="finance", help="Domain adapter name or YAML path.")
    ask_parser.add_argument(
        "--profile",
        default="auto",
        choices=["auto", "perception_visual", "data_extraction_precise", "calculation_formula", "pattern_visual_consistency", "logical_options", "decision_support_evidence"],
    )
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--context", default=None)
    ask_parser.add_argument("--capability", default=None)
    ask_parser.add_argument("--options-text", default=None, help="Raw multiple-choice options text.")
    ask_parser.add_argument("--max-completion-tokens", type=int, default=8192)
    ask_parser.add_argument("--image", action="append", dest="images", help="Image evidence to send to ERNIE.")
    ask_parser.add_argument("--no-web-search", action="store_true")

    batch_parser = subparsers.add_parser("batch", help="Batch parse or analyze supported files in a directory.")
    batch_parser.add_argument("directory")
    batch_parser.add_argument("--output-dir", default="output")
    batch_parser.add_argument("--domain", default="finance")
    batch_parser.add_argument(
        "--profile",
        default="auto",
        choices=["auto", "perception_visual", "data_extraction_precise", "calculation_formula", "pattern_visual_consistency", "logical_options", "decision_support_evidence"],
    )
    batch_parser.add_argument("--analyze", action="store_true")
    batch_parser.add_argument("--limit", type=int, default=None)
    batch_parser.add_argument("--max-completion-tokens", type=int, default=8192)
    batch_parser.add_argument("--no-image", action="store_true", help="Do not send image inputs to ERNIE.")
    batch_parser.add_argument("--no-web-search", action="store_true")
    _add_ocr_options(batch_parser)

    domains_parser = subparsers.add_parser("domains", help="List bundled domain adapters.")
    domains_parser.set_defaults(list_domains=True)

    dataset_parser = subparsers.add_parser("dataset-info", help="Inspect the remote PyFi-600K dataset metadata.")
    dataset_parser.add_argument("--json", action="store_true", dest="as_json")

    download_parser = subparsers.add_parser("dataset-download", help="Download PyFi-600K dataset files.")
    download_parser.add_argument("--output-dir", default="data/pyfi-600k")
    download_parser.add_argument("--selection", choices=["core", "full"], default="core")
    download_parser.add_argument("--file", action="append", dest="files", help="Specific dataset file to download.")
    download_parser.add_argument("--overwrite", action="store_true")

    manifest_parser = subparsers.add_parser("manifest", help="Create a small manifest from a local PyFi CSV file.")
    manifest_parser.add_argument("csv_path")
    manifest_parser.add_argument("--output", default="data/pyfi_manifest.json")
    manifest_parser.add_argument("--limit", type=int, default=100)
    manifest_parser.add_argument("--stratify", default=None, help="Column for stratified sampling, e.g. capability.")
    manifest_parser.add_argument("--sample-size", type=int, default=301)
    manifest_parser.add_argument("--seed", type=int, default=20260419)
    manifest_parser.add_argument("--allow-missing-images", action="store_true")
    manifest_parser.add_argument("--exclude-stratum", action="append", dest="exclude_strata", default=[])

    run_manifest_parser = subparsers.add_parser("run-manifest", help="Run real OCR+ERNIE calls from a manifest.")
    run_manifest_parser.add_argument("manifest_path")
    run_manifest_parser.add_argument("--output-dir", default="output-pyfi301")
    run_manifest_parser.add_argument("--domain", default="finance")
    run_manifest_parser.add_argument(
        "--profile",
        default="auto",
        choices=["auto", "perception_visual", "data_extraction_precise", "calculation_formula", "pattern_visual_consistency", "logical_options", "decision_support_evidence"],
    )
    run_manifest_parser.add_argument("--limit", type=int, default=None)
    run_manifest_parser.add_argument("--start", type=int, default=0)
    run_manifest_parser.add_argument("--max-completion-tokens", type=int, default=8192)
    run_manifest_parser.add_argument("--workers", type=int, default=2)
    run_manifest_parser.add_argument("--retries", type=int, default=2)
    run_manifest_parser.add_argument("--retry-delay-seconds", type=float, default=5.0)
    run_manifest_parser.add_argument("--no-resume", action="store_true")
    run_manifest_parser.add_argument("--no-image", action="store_true")
    run_manifest_parser.set_defaults(no_web_search=True)
    run_manifest_parser.add_argument("--web-search", action="store_true", help="Enable web search for manifest runs.")
    _add_ocr_options(run_manifest_parser)

    score_parser = subparsers.add_parser("score-manifest", help="Score a manifest run against held-out answers.")
    score_parser.add_argument("manifest_path")
    score_parser.add_argument("--run-output-dir", default="output-pyfi301")
    score_parser.add_argument("--domain", default="finance")
    score_parser.add_argument("--output", default=None)

    report_parser = subparsers.add_parser("report-manifest", help="Build a detailed per-sample report for a manifest run.")
    report_parser.add_argument("manifest_path")
    report_parser.add_argument("--run-output-dir", default="output-pyfi301")
    report_parser.add_argument("--domain", default="finance")
    report_parser.add_argument("--output", default=None)

    arch_parser = subparsers.add_parser("export-architecture", help="Export reproducible architecture docs.")
    arch_parser.add_argument("--output", default="docs/architecture.md")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings(args.env_file)

    try:
        if args.command == "parse":
            ocr_preset, options = build_layout_options(
                profile_name=None,
                requested_preset=args.ocr_preset,
                explicit_overrides=_layout_overrides(args),
                default_preset="medium",
            )
            run_dir = parse_document(
                input_path=args.file,
                output_root=args.output_dir,
                settings=settings,
                options=options,
                ocr_preset=ocr_preset,
            )
            print(run_dir)
            return 0

        if args.command == "analyze":
            answer_path = analyze_document(
                input_path=args.file,
                output_root=args.output_dir,
                settings=settings,
                domain_name=args.domain,
                requested_profile=args.profile,
                requested_ocr_preset=args.ocr_preset,
                question=args.question,
                context=args.context,
                web_search=not args.no_web_search,
                include_image=not args.no_image,
                max_completion_tokens=args.max_completion_tokens,
                layout_overrides=_layout_overrides(args),
            )
            print(answer_path)
            return 0

        if args.command == "ask":
            answer_path = analyze_run_dir(
                run_dir=args.run_dir,
                domain_name=args.domain,
                settings=settings,
                question=args.question,
                context=args.context,
                capability=args.capability,
                raw_options=args.options_text,
                requested_profile=args.profile,
                image_paths=args.images,
                web_search=not args.no_web_search,
                max_completion_tokens=args.max_completion_tokens,
            )
            print(answer_path)
            return 0

        if args.command == "batch":
            directory = Path(args.directory)
            files = [path for path in sorted(directory.rglob("*")) if path.is_file() and is_supported_document(path)]
            if args.limit is not None:
                files = files[: args.limit]
            for path in files:
                if args.analyze:
                    result = analyze_document(
                        input_path=path,
                        output_root=args.output_dir,
                        settings=settings,
                        domain_name=args.domain,
                        requested_profile=args.profile,
                        requested_ocr_preset=args.ocr_preset,
                        web_search=not args.no_web_search,
                        include_image=not args.no_image,
                        max_completion_tokens=args.max_completion_tokens,
                        layout_overrides=_layout_overrides(args),
                    )
                else:
                    ocr_preset, options = build_layout_options(
                        profile_name=None,
                        requested_preset=args.ocr_preset,
                        explicit_overrides=_layout_overrides(args),
                        default_preset="medium",
                    )
                    result = parse_document(
                        input_path=path,
                        output_root=args.output_dir,
                        settings=settings,
                        options=options,
                        ocr_preset=ocr_preset,
                    )
                print(result)
            return 0

        if args.command == "domains":
            for domain in available_domains():
                print(domain)
            return 0

        if args.command == "dataset-info":
            info = fetch_pyfi_dataset_info(timeout=settings.request_timeout)
            if args.as_json:
                print(json.dumps(info, ensure_ascii=False, indent=2))
            else:
                print(f"id: {info['id']}")
                print(f"sha: {info['sha']}")
                print(f"last_modified: {info['last_modified']}")
                print(f"license: {info['license']}")
                print("files:")
                for item in info["files"]:
                    print(f"  - {item['name']}")
            return 0

        if args.command == "dataset-download":
            paths = download_pyfi_files(
                output_dir=args.output_dir,
                selection=args.selection,
                explicit_files=args.files,
                timeout=settings.request_timeout,
                overwrite=args.overwrite,
            )
            for path in paths:
                print(path)
            return 0

        if args.command == "manifest":
            if args.stratify:
                output = create_stratified_manifest_from_csv(
                    args.csv_path,
                    args.output,
                    sample_size=args.sample_size,
                    stratify_column=args.stratify,
                    seed=args.seed,
                    require_existing_image=not args.allow_missing_images,
                    exclude_strata=args.exclude_strata,
                )
            else:
                output = create_manifest_from_csv(args.csv_path, args.output, limit=args.limit)
            print(output)
            return 0

        if args.command == "run-manifest":
            output = run_manifest(
                manifest_path=args.manifest_path,
                output_dir=args.output_dir,
                settings=settings,
                domain=args.domain,
                layout_overrides=_layout_overrides(args),
                requested_profile=args.profile,
                requested_ocr_preset=args.ocr_preset,
                limit=args.limit,
                start=args.start,
                resume=not args.no_resume,
                include_image=not args.no_image,
                web_search=args.web_search,
                max_completion_tokens=args.max_completion_tokens,
                workers=args.workers,
                retries=args.retries,
                retry_delay_seconds=args.retry_delay_seconds,
            )
            print(output)
            return 0

        if args.command == "score-manifest":
            output = score_manifest_run(
                manifest_path=args.manifest_path,
                run_output_dir=args.run_output_dir,
                domain=args.domain,
                output_path=args.output,
            )
            print(output)
            return 0

        if args.command == "report-manifest":
            output = build_manifest_report(
                manifest_path=args.manifest_path,
                run_output_dir=args.run_output_dir,
                domain=args.domain,
                output_path=args.output,
            )
            print(output)
            return 0

        if args.command == "export-architecture":
            output = export_architecture(args.output)
            print(output)
            return 0

    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 1
