# Experiment 53: Relative RealDESED Onset Temporal-ID Intervention

This experiment runs RealDESED single-event onset detection with event-relative
temporal-ID edits. The supported entrypoint is:

```bash
python experiments/experiment_53_sed_intervention/run_sed_intervention.py \
  --model moss-audio \
  --intervention-layer 18 \
  --alpha 1.0
```

## Method

The script scores the same RealDESED onset examples twice:

1. baseline onset generation
2. onset generation with a per-example temporal-ID direction added at the
   queried event-label tokens

Temporal IDs are reconstructed from experiment 41 through experiment 52’s
relative temporal-ID utilities. Stored activation bundles are indexed as:

```text
slot 0 = decoder_input
slot N + 1 = decoder_layer_N_output
```

The script maps `--intervention-layer N` to `decoder_layer_N_output` when
loading temporal-ID vectors, then hooks decoder layer `N` during generation.

For each example, the queried event center is converted to relative time within
the RealDESED audio window. The direction is:

```text
direction = tau_target - tau_event_center
```

with linear interpolation over the synthetic temporal-ID table. The default
target is `end` at `26.25 / 30`; use `--target beginning` to steer toward
`3.75 / 30`.

## RealDESED Example Counts

The configured RealDESED onset sample size is 1,000 examples with `split=all`,
`min_event_duration_sec=2.5`, `max_event_duration_sec=5.5`,
`max_overlap_fraction=0.1`, `max_audio_length_sec=30.0`, and `footsteps`
excluded. The table below reports the aligned examples used by the selected
paper runs in
`outputs/paper/relative_temporal_id_endpoint_steering_onset_shift_per_box_stats.csv`;
these counts come from the existing outputs and do not require rerunning the
extraction scripts.

| Model | Layer | Alpha | Aligned onset examples per target | Baseline early examples | Baseline late examples |
| --- | ---: | ---: | ---: | ---: | ---: |
| AF-Next | 16 | 1.0 | 998 | 803 | 195 |
| MOSS-Audio | 18 | 0.5 | 998 | 663 | 335 |
| Qwen3-Omni | 24 | 1.0 | 998 | 641 | 357 |

## Outputs

Primary prediction CSVs and metadata are written to:

```text
experiments/experiment_53_sed_intervention/outputs/intervention/<model_slug>/
```

Derived CSV outputs are grouped by type:

```text
experiments/experiment_53_sed_intervention/outputs/aligned/<model_slug>/
experiments/experiment_53_sed_intervention/outputs/summary/<model_slug>/
```

Plot-side outputs are grouped by type:

```text
experiments/experiment_53_sed_intervention/plots/prediction_delta/<model_slug>/
experiments/experiment_53_sed_intervention/plots/predicted_onset_scatter/<model_slug>/
```
