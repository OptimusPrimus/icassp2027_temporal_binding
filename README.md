# Temporal Binding in Large Audio-Language Models

This repository contains code, experiment scripts, saved outputs, and
plots for our ICASSP 2027 paper "On Termporal Binding in Large Audio-Language Models".

## Repository Layout

- `audio_lm_interfaces/`: shared interfaces for the supported audio language
  models.
- `dataset/`: dataset loaders and utilities for ESC-50,  background ambience from 
Freesound, synthetic sound event detection, and RealDESED.
- `experiments/`: experiment-specific scripts, READMEs, outputs, and plots.
- `environment.yml`, `environment-moss.yml`, `environment-qwen3-omni.yml`:
  conda environments for the main setup and model-specific dependencies.

## Experiments

Each experiment directory contains its own `README.md` with run commands,
method details, and output conventions.

- [`experiments/experiment_31_ordering_intervention/`](experiments/experiment_31_ordering_intervention/):
  synthetic two-event before/after ordering intervention with token-group and
  layer sweeps.
- [`experiments/experiment_32_decode_coarse_position/`](experiments/experiment_32_decode_coarse_position/):
  activation capture and layer-wise decoding of coarse event position from
  synthetic 30-second two-event recordings.
- [`experiments/experiment_41_temporal_IDs/`](experiments/experiment_41_temporal_IDs/):
  fixed-length synthetic activation capture and temporal-ID analysis, including
  PCA projections and paper figures.
- [`experiments/experiment_42_relative_time/`](experiments/experiment_42_relative_time/):
  RealDESED relative-time activation analysis with variable 15-30 second input
  windows and synthetic temporal-ID references.
- [`experiments/experiment_52_intervention/`](experiments/experiment_52_intervention/):
  pairwise RealDESED before/after temporal-ID interventions using query-token
  edits.
- [`experiments/experiment_53_sed_intervention/`](experiments/experiment_53_sed_intervention/):
  RealDESED onset prediction interventions with event-relative temporal-ID
  steering.

## Outputs and Plots

Generated files are kept with the corresponding experiment:

- `experiments/*/outputs/`: saved activations, CSV predictions, summaries, and
  intermediate analysis products.
- `experiments/*/plots/`: generated analysis figures and paper plots.