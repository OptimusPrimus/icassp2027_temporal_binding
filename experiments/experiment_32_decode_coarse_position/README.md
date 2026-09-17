# Experiment 32: Decode Coarse Position

This experiment captures decoder activations for yes/no event-presence prompts
on fixed-length 30 second synthetic recordings with two ESC-50 events and trains predictors
to decode the event's temporal position.


## Data

Synthetic examples use `SyntheticSoundEventDetectionDataset` with:

- foreground dataset: ESC-50
- two foreground events per recording
- fixed recording length of 30 seconds
- random event placement inside the recording
- trimmed ESC-50 foreground clips
- `unique_event_classes=True`, so the two events are not from the same class
- train: 2,000 examples
- validation: 1,000 examples
- test: 1,000 examples

## Prompt

One of the two events is selected randomly as the query event. Each example
uses:

```text
Is there {query_event}?
```

The model also performs greedy decoding for up to 10 new tokens. The decoded
answer is stored with the activation bundle.

## Activations

For each example, the script captures the text tokens from the prompt plus the
part of the chat template that follows it. It stores:

- hidden states before the first decoder layer
- hidden states after every decoder layer
- token indices and token strings
- full input tokenization
- generated answer text
- synthetic metadata, including foreground events, source files, event
  positions, SNRs, and background file metadata

The output is a single `.pt` file. Activations are stored as a ragged list with
one tensor per example:

```text
[layer_slot, selected_token, hidden]
```

Layer slot `0` is the input to the first decoder layer. Slots `1..N` are the
outputs after decoder layers `0..N-1`.

## Run

```bash
python experiments/experiment_32_decode_coarse_position/run_collect_activations.py \
  --model af3 \
  --device auto
```

To run another supported audio language model:

```bash
python experiments/experiment_32_decode_coarse_position/run_collect_activations.py \
  --model qwen3-omni \
  --device auto
```

The default output is written under:

```text
experiments/experiment_32_decode_coarse_position/outputs/activations/{model_id_slug}/
```

Use `--max-output-gb` to guard against unexpectedly large activation files, or
`--allow-large-output` to override the guard.

## Decode Event Time

After activations have been captured, train layer-wise midpoint predictors from
the saved text embeddings:

```bash
python experiments/experiment_32_decode_coarse_position/run_decoding_event_time_from_saved_embeddings.py \
  --model af3
```

The script fits the same small neural midpoint regressor used by
`reproduce_midpoint_bins.py` for every decoder layer slot:

- target event midpoint from the final token of the prompted event class
- alternative event midpoint from the same prompted event-class token
- target event midpoint from the first prompt token

For each condition and layer slot, the input is one hidden-state vector from the
selected text token. Features are standardized with the mean and standard
deviation computed only on that training fold. The regressor is:

```text
Linear(hidden_size, 96)
GELU
Dropout(p=0.20)
Linear(96, 1)
30 * sigmoid
```

The output is constrained to the 30 second recording range. Training uses AdamW
with learning rate `1e-3`, weight decay `3e-3`, 12 epochs, batch size 1024, and
Smooth L1 loss with `beta=1.0`. All training examples are used. The regressor is
always `midpoint_nn`; `torch.set_num_threads(min(8, os.cpu_count()))` is used to
match the midpoint-bin reproduction script.

By default it also repeats the layer-wise regression with ESC-50 class-heldout
cross-validation. The 50 ESC-50 classes are split into 5 folds; each fold trains
on `--train-split` examples from 40 classes and evaluates on `--eval-split`
examples from the 10 held-out classes. Use `--class-fold-count` to change the
number of folds or `--skip-class-cv` to only run the original split evaluation.

Results are written under:

```text
experiments/experiment_32_decode_coarse_position/outputs/regular_split/{model_id_slug}/
```

The class-heldout results are written under
`experiments/experiment_32_decode_coarse_position/outputs/class_cv_split/{model_id_slug}/`
to `event_midpoint_class_cv_metrics_by_layer.csv`,
`event_midpoint_class_cv_fold_metrics_by_layer.csv`,
`event_midpoint_class_cv_predictions.csv`, and
`event_midpoint_class_cv_summary.json`.

Create the per-model MAE plot from the saved CSVs:

```bash
python experiments/experiment_32_decode_coarse_position/plot_decoding_event_time_from_saved_embeddings.py \
  --model af3
```

The default plot is written under:

```text
experiments/experiment_32_decode_coarse_position/plots/event_time_decoding/{model_id_slug}/
```

Plot the compact all-model paper MAE figure:

```bash
python experiments/experiment_32_decode_coarse_position/plot_all_models_paper_mae_lines.py
```
