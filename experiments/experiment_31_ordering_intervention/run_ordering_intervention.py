import argparse
import csv
import json
from pathlib import Path
import sys

from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.experiment_31_ordering_intervention.synthetic_ordering_dataset import (
    FixedPositionSyntheticSedOrderingDataset,
)
from experiments.experiment_31_ordering_intervention.swap_tokens_intervention import (
    DEFAULT_MODEL_IDS,
    INTERVENTION_TOKEN_TYPES,
    OrderingIntervention,
    build_model_interface,
    default_model_id,
    model_output_slug,
)


CANDIDATE_TEXTS = [" before", " after"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run before/after activation swapping on fixed-position SyntheticSED examples."
    )
    parser.add_argument(
        "--model",
        default="af3",
        choices=tuple(DEFAULT_MODEL_IDS),
        help="Audio language model backend to run.",
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "dataset"))
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-auto-download", action="store_true")
    parser.add_argument(
        "--intervention-layer",
        type=int,
        required=True,
        help="Decoder layer to intervene on.",
    )
    parser.add_argument(
        "--intervention-tokens",
        nargs="+",
        default=["audio"],
        choices=INTERVENTION_TOKEN_TYPES,
        help="Token positions to replace.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--csv-name", default=None)
    return parser.parse_args()


def intervention_tokens_label(intervention_tokens):
    if isinstance(intervention_tokens, str):
        return intervention_tokens
    return "+".join(intervention_tokens)


def default_csv_name(args):
    model_slug = args.model.replace("-", "_")
    part = intervention_tokens_label(args.intervention_tokens)
    return (
        "ordering_intervention_syntheticsed_10s_"
        f"{model_slug}_layer{args.intervention_layer}_{part}.csv"
    )


def probability_by_candidate(result):
    return {
        item["text"].strip(): item["probability"]
        for item in result["keyword_probabilities"]
    }


def belief_shift(normal_correct_probability, intervened_correct_probability, swapped_correct_probability):
    denominator = normal_correct_probability - swapped_correct_probability
    if denominator == 0:
        return None
    return (normal_correct_probability - intervened_correct_probability) / denominator


def build_dataset(args):
    return FixedPositionSyntheticSedOrderingDataset(
        root=args.dataset_root,
        sample_rate=args.sample_rate,
        size=args.size,
        random_seed=args.random_seed,
        auto_download=not args.no_auto_download,
    )


def write_rows(csv_path, rows):
    fieldnames = [
        "id",
        "model_id",
        "query_event",
        "reference_event",
        "prompt",
        "teacher_forced_input",
        "answer",
        "before_probability",
        "after_probability",
        "correct_probability",
        "swapped_correct_probability",
        "intervened_correct_probability",
        "belief_shift",
        "intervention_layer",
        "intervention_token_type",
        "intervention_token_indices",
        "intervention_tokens",
        "top10_tokens",
        "first_event",
        "second_event",
        "first_source_id",
        "second_source_id",
        "background_id",
    ]

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score_rows(args, dataset, model, model_id):
    rows = []
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=list,
    )

    for batch in tqdm(dataloader, desc="Scoring SyntheticSED ordering examples"):
        prompts = [
            f"Does {sample['query_label']} occur before or after {sample['reference_label']}? "
            for sample in batch
        ]
        teacher_forced_inputs = [
            f"{sample['query_label']} occurs"
            for sample in batch
        ]
        correct_keywords = [f" {sample['answer']}" for sample in batch]
        sampling_rates = [sample["sample_rate"] for sample in batch]

        results = model.forward_intervention(
            [sample["waveform"] for sample in batch],
            [sample["swapped_waveform"] for sample in batch],
            prompts,
            correct_keywords=correct_keywords,
            layer=args.intervention_layer,
            token_type=args.intervention_tokens,
            teacher_forced_texts=teacher_forced_inputs,
            sampling_rates=sampling_rates,
            keywords=CANDIDATE_TEXTS,
            top_k=10,
        )

        for sample, prompt, teacher_forced_input, result in zip(
            batch,
            prompts,
            teacher_forced_inputs,
            results,
        ):
            probabilities = probability_by_candidate(result["normal"])
            correct_probability = result["normal_correct_probability"]
            swapped_correct_probability = result["swapped_correct_probability"]
            intervened_correct_probability = result["intervened_correct_probability"]
            intervention = result["intervention"]

            rows.append({
                "id": sample["id"],
                "model_id": model_id,
                "query_event": sample["query_label"],
                "reference_event": sample["reference_label"],
                "prompt": prompt,
                "teacher_forced_input": teacher_forced_input,
                "answer": sample["answer"],
                "before_probability": probabilities["before"],
                "after_probability": probabilities["after"],
                "correct_probability": correct_probability,
                "swapped_correct_probability": swapped_correct_probability,
                "intervened_correct_probability": intervened_correct_probability,
                "belief_shift": belief_shift(
                    normal_correct_probability=correct_probability,
                    intervened_correct_probability=intervened_correct_probability,
                    swapped_correct_probability=swapped_correct_probability,
                ),
                "intervention_layer": args.intervention_layer,
                "intervention_token_type": intervention.get(
                    "token_type",
                    intervention_tokens_label(args.intervention_tokens),
                ),
                "intervention_token_indices": json.dumps(intervention.get("token_indices", [])),
                "intervention_tokens": json.dumps(intervention.get("tokens", [])),
                "top10_tokens": json.dumps(result["normal"]["top_tokens"]),
                "first_event": sample["first_label"],
                "second_event": sample["second_label"],
                "first_source_id": sample["first_source_id"],
                "second_source_id": sample["second_source_id"],
                "background_id": sample["background_id"],
            })

    return rows


def run_analysis(args, dataset=None, model=None):
    model_id = args.model_id or default_model_id(args.model)
    output_dir = Path(args.output_dir) / model_output_slug(model_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset is None:
        dataset = build_dataset(args)
    if model is None:
        model = OrderingIntervention(
            build_model_interface(args.model, model_id=model_id, device=args.device)
        )

    rows = score_rows(args, dataset, model, model_id=model_id)
    csv_path = output_dir / (args.csv_name or default_csv_name(args))
    write_rows(csv_path, rows)

    print(f"Wrote CSV: {csv_path}")
    return csv_path


def main():
    args = parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
