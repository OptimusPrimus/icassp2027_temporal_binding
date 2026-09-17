# Experiment 52: Pairwise Relative Temporal-ID Intervention

This experiment tests whether event-position-dependent temporal-ID directions
causally change RealDESED before/after predictions. The supported entrypoint is:

```bash
python experiments/experiment_52_intervention/run_before_after_intervention.py \
  --model af3 \
  --intervention-layer 16 \
  --alpha 1.0
```

## Method

Experiment 41 stores decoder activations as:

```text
slot 0 = decoder_input
slot N + 1 = decoder_layer_N_output
```

Experiment 52 maps `--intervention-layer N` to `decoder_layer_N_output` when
extracting temporal-ID vectors, so the stored vector and hooked decoder layer
refer to the same layer output.

For each RealDESED pair, the model is prompted with:

```text
Does <query event> occur before or after <reference event>?
```

and teacher-forced with:

```text
<query event> occurs
```

The script scores next-token probabilities for `" before"` and `" after"`.

The default relative intervention interpolates the synthetic temporal-ID table on
the relative-time axis. For an event at relative time `r_event`:

```text
tau_event = tau(r_event)
direction = tau_target - tau_event
```

The script runs two query-token interventions from the same baseline:

```text
query_to_end
query_to_beginning
```

Query edits target `part2` and `part10`. The edit is applied at the requested
decoder layer as:

```text
hidden_state <- hidden_state + alpha * mean_norm(selected_hidden_states) * direction
```

By default `tau_beginning` is `3.75 / 30` and `tau_end` is `26.25 / 30`.

## RealDESED Example Counts

The configured RealDESED sample size is 1,000 examples with `split=all`,
`min_event_duration_sec=2.5`, `max_event_duration_sec=5.5`, `min_gap_sec=1.0`,
`max_audio_length_sec=30.0`, and `footsteps` excluded. The table below reports
the aligned examples used by the selected query-only paper runs; these counts
come from the existing query intervention outputs and do not require rerunning
the extraction scripts.

| Model | Layer | Alpha | Aligned pairwise examples per direction | Baseline before examples | Baseline after examples | Plotted direction-example rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AF-next | 16 | 0.5 | 133 | 86 | 47 | 266 |
| MOSS-audio | 18 | 0.75 | 133 | 77 | 56 | 266 |
| Qwen3-omni | 24 | 1.25 | 133 | 65 | 68 | 266 |

## Outputs

Primary probability CSVs and metadata are written to:

```text
experiments/experiment_52_intervention/outputs/intervention/<model_slug>/
```

Derived CSV outputs are grouped by type:

```text
experiments/experiment_52_intervention/outputs/aligned/<model_slug>/
experiments/experiment_52_intervention/outputs/summary/<model_slug>/
```

Plot-side outputs are grouped by type:

```text
experiments/experiment_52_intervention/plots/probability_mass_change/<model_slug>/
```

Each intervention gets its own aligned CSV, summary CSV, and
probability-mass-change plot.
