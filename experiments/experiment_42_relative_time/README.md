# Experiment 42: Relative Time

This experiment captures decoder activations for yes/no event-presence prompts
when the RealDESED input window varies between 15 and 30 seconds, capped by the
actual recording length.

The setup is similar to `experiment_41_temporal_IDs`, but it uses RealDESED recordings 
with a recordings in between 15 and 30 seconds.


## Data

By default, activation capture uses only RealDESED. Synthetic examples can be
included with `--include-synthetic`; when enabled they use
`SyntheticSoundEventDetectionDataset` with:

- foreground dataset: ESC-50
- one foreground event per recording
- random recording length sampled from 15 to 30 seconds
- trimmed ESC-50 foreground clips
- train: 2,000 examples
- validation: 1,000 examples
- test: 1,000 examples

RealDESED examples are included by default. The script selects recordings from
train, validation, and test by default, keeping examples with a selected event
that:

- has duration between 3 and 7 seconds
- belongs to a class that occurs only once in the recording

The script samples a random 15-30 second window that contains
the selected event. The window length is capped to the actual recording length,
so RealDESED examples are not zero-padded. Event timings are stored relative to
the sampled window, and the original recording timing is preserved in metadata.

With the default RealDESED dataset, the number of eligible recordings is:

| split | recordings | eligible recordings | default selected |
| --- | ---: | ---: | ---: |
| train | 3,704 | 1,351 | 1,351 |
| validation | 999 | 320 | 320 |
| test | 1,007 | 354 | 354 |
| total | 5,710 | 2,025 | 2,025 |


If `--real-desed-root` is not provided, the script uses the first existing root
from:

- `/home/paul/repos/domestic_sed_dataset/data/release`
- `/opt/scratch/paul/data/real_desed`

None of which will work for you unless you are named Paul :)
Find the RealDESED dataset at https://zenodo.org/records/20056072 and set
`--real-desed-root` to the path where you unpacked it.

## Prompt

Each example uses:

```text
Is there {event}?
```

No response is generated. The script runs one forward pass and captures
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

Defaults:

- model: `af3` (`nvidia/audio-flamingo-3-hf`)
- sample rate: 16 kHz
- synthetic examples: skipped unless `--include-synthetic` is passed
- synthetic examples with `--include-synthetic`: 2,000 train, 1,000 validation, 1,000 test
- RealDESED examples: all eligible examples from all splits
- RealDESED selected-event duration: 3-7 seconds
- input duration: random 15-30 second window, capped to recording length
- random seed: 0
- activation dtype: `float16`
- batch size: 1

```bash
python experiments/experiment_42_relative_time/run_save_relative_time_activations.py \
  --model af3 \
  --device auto
```

To run another supported audio language model:

```bash
python experiments/experiment_42_relative_time/run_save_relative_time_activations.py \
  --model moss-audio \
  --device auto
```

The default output is written under:

```text
experiments/experiment_42_relative_time/outputs/activations/{model_id_slug}/
```

To also capture synthetic ESC-50 activations in the same bundle:

```bash
python experiments/experiment_42_relative_time/run_save_relative_time_activations.py \
  --model af3 \
  --device auto \
  --include-synthetic
```

Use `--max-output-gb` to guard against unexpectedly large activation files, or
`--allow-large-output` to override the guard.

## RealDESED Relative-Time Projection

After capturing activations, project RealDESED class-token activations onto the
synthetic training temporal IDs:

```bash
python experiments/experiment_42_relative_time/plot_temporal_id_relative_analysis.py \
  --model af3
```

The analyzer uses:

- model: `af3` unless `--model` is set
- synthetic reference bundle: newest fixed-30s `experiment_41_temporal_IDs` activation bundle for the model
- RealDESED relative-time bundle: newest `experiment_42_relative_time` activation bundle for the model
- layer slots: 14-20 for AF3/AF-Next, 16-22 for MOSS-Audio, 22-28 for Qwen3
- token: `query_event`
- event time: center
- synthetic split: train
- bin width: 2.5 seconds
- excluded bin centers: 0.0 and 30.0 seconds
- minimum bin count: 10
- synthetic reference duration: 30 seconds
- bootstrap iterations: 2,000

It builds class-mean-centered temporal IDs from the synthetic training split,
fits a 2D PCA plus quadratic trajectory, and compares RealDESED event
coordinates against:

- absolute event center time in seconds
- event center divided by input duration
- event center rescaled to the synthetic ID axis

Per-layer CSV/JSON outputs are written under:

```text
experiments/experiment_42_relative_time/outputs/realdesed_relative_time_validation/{model_id_slug}/
```

PNG plots are written under plot-type-specific directories:

```text
experiments/experiment_42_relative_time/plots/{plot_type}/{model_id_slug}/
```
