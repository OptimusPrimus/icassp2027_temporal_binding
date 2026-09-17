import argparse
from argparse import Namespace
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from experiments.experiment_31_ordering_intervention.swap_tokens_intervention import (
    DEFAULT_MODEL_IDS,
    INTERVENTION_TOKEN_TYPES,
    OrderingIntervention,
    build_model_interface,
)
from run_ordering_intervention import (
    DEFAULT_OUTPUT_DIR,
    build_dataset,
    intervention_tokens_label,
    run_analysis,
)


DEFAULT_LAYER_COUNTS = {
    "af3": 28,
    "af-next": 28,
    "moss-audio": 36,
    "qwen3-omni": 48,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run fixed-position SyntheticSED ordering interventions for multiple layers and token groups."
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
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--intervention-tokens",
        nargs="+",
        default=None,
        help=(
            "Intervention token groups to run. Tokens in the same group are combined; "
            "separate groups with --intervention-token-separator. "
            "Example: --intervention-tokens part4 part6 , events"
        ),
    )
    parser.add_argument(
        "--intervention-token-separator",
        default=",",
        help="Separator token used inside --intervention-tokens to split intervention groups.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Decoder layers to run. Defaults to every layer for the selected backend.",
    )
    return parser.parse_args()


def intervention_token_groups(args):
    if args.intervention_tokens is None:
        return [("audio",), ("text",), ("events",), ("part11",), ("control",)]

    groups = []
    current_group = []
    for item in args.intervention_tokens:
        if item == args.intervention_token_separator:
            if not current_group:
                raise ValueError("intervention token separator cannot start or follow a group")
            groups.append(tuple(current_group))
            current_group = []
            continue
        if item not in INTERVENTION_TOKEN_TYPES:
            raise ValueError(
                "intervention tokens must be one of "
                f"{', '.join(INTERVENTION_TOKEN_TYPES)} or separator "
                f"{args.intervention_token_separator!r}; got {item!r}"
            )
        current_group.append(item)

    if not current_group:
        raise ValueError("intervention token separator cannot end the token list")
    groups.append(tuple(current_group))
    return groups


def analysis_args(args, layer, intervention_tokens):
    return Namespace(
        model=args.model,
        model_id=args.model_id,
        device=args.device,
        dataset_root=args.dataset_root,
        size=args.size,
        sample_rate=args.sample_rate,
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        no_auto_download=args.no_auto_download,
        intervention_layer=layer,
        intervention_tokens=list(intervention_tokens),
        output_dir=args.output_dir,
        csv_name=None,
    )


def validate_layers(layers, layer_count):
    invalid_layers = [layer for layer in layers if layer < 0 or layer >= layer_count]
    if invalid_layers:
        raise ValueError(
            f"layers must be in [0, {layer_count - 1}], got {invalid_layers}"
        )


def default_layers(model_name, layer_count):
    return list(range(DEFAULT_LAYER_COUNTS.get(model_name, layer_count)))


def main():
    args = parse_args()

    dataset = build_dataset(args)
    model = OrderingIntervention(
        build_model_interface(args.model, model_id=args.model_id, device=args.device)
    )
    layer_count = model.decoder_layer_count()
    layers = args.layers if args.layers is not None else default_layers(args.model, layer_count)
    validate_layers(layers, layer_count)
    token_groups = intervention_token_groups(args)

    for intervention_tokens in token_groups:
        intervention_tokens_name = intervention_tokens_label(intervention_tokens)
        for layer in layers:
            print(
                f"Running {intervention_tokens_name} intervention "
                f"for layer {layer}/{layer_count - 1}"
            )
            run_analysis(
                analysis_args(
                    args,
                    layer=layer,
                    intervention_tokens=intervention_tokens,
                ),
                dataset=dataset,
                model=model,
            )


if __name__ == "__main__":
    main()
