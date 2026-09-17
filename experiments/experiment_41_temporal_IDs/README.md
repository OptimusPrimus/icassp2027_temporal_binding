# Experiment 41: Temporal IDs

This experiment captures decoder activations for yes/no event-presence prompts
on fixed-length 30 second one-event synthetic ESC-50 recordings.

## Data

Synthetic examples use `SyntheticSoundEventDetectionDataset` with:

- foreground dataset: ESC-50
- one foreground event per recording
- fixed recording length of 30 seconds
- random event placement inside the recording
- trimmed ESC-50 foreground clips
- train: 2,000 examples
- validation: 1,000 examples
- test: 1,000 examples

With the background metadata recorded for this experiment, the synthetic setup
uses 200 FreeSound ambience files:

| split | ESC-50 source clips | event classes | eligible background files | generated examples |
| --- | ---: | ---: | ---: | ---: |
| train | 1,200 | 50 | 119 | 2,000 |
| validation | 400 | 50 | 42 | 1,000 |
| test | 400 | 50 | 39 | 1,000 |


## Prompt

Each example uses:

```text
Is there {event}?
```

No response is generated. The script only performs one forward pass and captures
activations.

## Activations

For each example, the script captures the text tokens from the prompt plus the
part of the chat template that follows it. It stores:

- hidden states before the first decoder layer
- hidden states after every decoder layer
- token indices and token strings
- full input tokenization
- synthetic and RealDESED metadata

The output is a single `.pt` file. Activations are stored as a ragged list with
one tensor per example:

```text
[layer_slot, selected_token, hidden]
```

Layer slot `0` is the input to the first decoder layer. Slots `1..N` are the
outputs after decoder layers `0..N-1`; for example, slot `14` is
`decoder_layer_13_output`.

## Run

Default capture settings:

- model key: `af3` (`nvidia/audio-flamingo-3-hf`)
- sample rate: 16 kHz
- recording length: 30 seconds
- synthetic examples: train 2,000, validation 1,000, test 1,000
- RealDESED examples: skipped by default
- RealDESED filters: recordings up to 30 seconds; unique queried event class;
  queried event duration from 3 to 7 seconds
- random seed: 0
- batch size: 1
- activation storage dtype: `float16`

```bash
python experiments/experiment_41_temporal_IDs/run_save_activations.py \
  --model af3 \
  --device auto
```

To run another supported audio language model:

```bash
python experiments/experiment_41_temporal_IDs/run_save_activations.py \
  --model moss-audio \
  --device auto
```

The default output is written under:

```text
experiments/experiment_41_temporal_IDs/outputs/activations/{model_id_slug}/
```

Use `--max-output-gb` to guard against unexpectedly large activation files, or
`--allow-large-output` to override the guard.

To explicitly include RealDESED activations:

```bash
python experiments/experiment_41_temporal_IDs/run_save_activations.py \
  --model af3 \
  --device auto \
  --real-desed-examples 1000
```

## Analyze Temporal IDs

After capturing activations, extract 2.5 second temporal ID bins from the
synthetic train split, remove train-set class means, plot PCA variance for
ranks 1 through 5, plot the 2D PCA projection of temporal IDs only, and evaluate
linear, quadratic, and 15 second peak piecewise-linear fits. By default, the
analysis runs a model-specific layer range and writes an overview plot of
train/validation R2 by layer:

```bash
python experiments/experiment_41_temporal_IDs/plot_temporal_id_analysis.py \
  --model af3
```

Default analysis settings:

- token: `query_event`
- time field: event center
- bin width: 2.5 seconds
- first bin center: 2.5 seconds
- minimum examples per split/bin temporal ID: 10
- excluded bin centers: 0.0 and 30.0 seconds
- peak time for the piecewise-linear fit: 15 seconds
- plot DPI: 200

Default layer-slot ranges are:

- `af3` and `af-next`: 14 through 20
- `moss-audio`: 16 through 22
- `qwen3-audio` / `qwen3-omni`: 22 through 28

Override the range with:

```bash
python experiments/experiment_41_temporal_IDs/plot_temporal_id_analysis.py \
  --model moss-audio \
  --min-layer 18 \
  --max-layer 24
```

For one layer only, use `--layer`.

By default `--token query_event` selects the token corresponding to the queried
event name. If the event name spans multiple tokens, the last event-name token
is used.

Bins centered at 0.0 and 30.0 seconds are excluded by default with
`--exclude-bin-centers-sec 0.0 30.0`. Sparse temporal ID bins are also filtered
with `--min-bin-count 10` by default. Use `--exclude-bin-centers-sec` with no
values and `--min-bin-count 1` to reproduce unfiltered plots.


CSV and JSON outputs are written under:

```text
experiments/experiment_41_temporal_IDs/outputs/temporal_id_analysis/{model_id_slug}/
```

Plots are written under:

```text
experiments/experiment_41_temporal_IDs/plots/pca_rank_variance/{model_id_slug}/
experiments/experiment_41_temporal_IDs/plots/pca_2d_projection/{model_id_slug}/
experiments/experiment_41_temporal_IDs/plots/model_fits_pca_2d/{model_id_slug}/
experiments/experiment_41_temporal_IDs/plots/r2_overview/{model_id_slug}/
```

## Paper PCA Figure

Create the compact one-column 2D PCA figure for the paper with:

```bash
python experiments/experiment_41_temporal_IDs/plot_paper_temporal_id_pca.py
```

The default panels are AF-next decoder layer 16, MOSS-audio decoder layer 18,
and qwen3-omni decoder layer 24, corresponding to layer slots 17, 19, and 25.
The script writes PDF and PNG outputs under:

```text
experiments/experiment_41_temporal_IDs/plots/paper/
```
