# Experiment 31: Ordering Intervention

This experiment builds 10-second SyntheticSED examples with two ESC-50 foreground events placed at fixed positions: one at 0 seconds and one at 5 seconds.
It then swaps residual stream activation of a forward pass on the original audio with a forward pass on the audio with swapped events. 
The experiment measures how much the model's next-token prediction for `" before"` and `" after"` changes as a result of the intervention.

The dataset construction lives in `synthetic_ordering_dataset.py`.

The swap-token activation replacement logic lives in `swap_tokens_intervention.py`.

The two run scripts parse arguments, build the dataset/model, score examples, and write CSV rows.

Default setup:

- Dataset root: `dataset`
- Foreground dataset: ESC-50
- Split: `train`
- Clip length: 10.0 seconds
- Event onsets: 0.0 seconds and 5.0 seconds
- Event count: 2 unique event classes
- Sample rate: 16000 Hz
- Dataset size: 1000 examples
- Experiment random seed: 0
- Single-run batch size: 1
- Multi-run batch size: 1
- Worker count: 4
- Default model key: `af3` (`nvidia/audio-flamingo-3-hf`)
- Single-run default intervention tokens: `audio`
- Multi-run default intervention groups: `audio`, `text`, `events`, `part11`, and `control`

For each example, the prompt asks:

```text
Does <query event> occur before or after <reference event>?
```

The model is teacher-forced with:

```text
<query event> occurs
```

and the experiment scores the next-token probabilities for `" before"` and `" after"`.

## Intervention

Each example is paired with a swapped-audio counterfactual where the same two event clips trade time positions while the text prompt stays unchanged. The experiment runs three forwards:

- Normal audio: original event order.
- Swapped audio: counterfactual event order.
- Intervened audio: original input, but selected hidden states at one decoder layer are replaced with hidden states from the swapped-audio forward.

The intervention can target audio tokens, all text tokens, event-name spans, the `control` relation phrase (`before or after`), or specific prompt/teacher-forced parts (`part1` through `part11`). The CSV reports how far the intervened probability moves from the normal prediction toward the swapped-audio prediction as `belief_shift`.

## Usage

Run intervention for one layer/token group:

```bash
python experiments/experiment_31_ordering_intervention/run_ordering_intervention.py \
  --model af3 \
  --intervention-layer 16 \
  --intervention-tokens audio
```

Run multiple layers/token groups:

```bash
python experiments/experiment_31_ordering_intervention/run_multiple_ordering_intervention.py \
  --model af3 \
  --layers 0 8 16 24 \
  --intervention-tokens audio , events , control , part4 part6
```

Run complete interventions for every layer and every token group by omitting
`--layers` and passing each token group explicitly:

```bash
python experiments/experiment_31_ordering_intervention/run_multiple_ordering_intervention.py \
  --model af3 \
  --intervention-tokens audio , text , events , control , part1 , part2 , part3 , part4 , part5 , part6 , part7 , part8 , part9 , part10 , part11
```

```bash
python experiments/experiment_31_ordering_intervention/run_multiple_ordering_intervention.py \
  --model af-next \
  --intervention-tokens audio , text , events , control , part1 , part2 , part3 , part4 , part5 , part6 , part7 , part8 , part9 , part10 , part11
```

```bash
python experiments/experiment_31_ordering_intervention/run_multiple_ordering_intervention.py \
  --model moss-audio \
  --intervention-tokens audio , text , events , control , part1 , part2 , part3 , part4 , part5 , part6 , part7 , part8 , part9 , part10 , part11
```

```bash
python experiments/experiment_31_ordering_intervention/run_multiple_ordering_intervention.py \
  --model qwen3-omni \
  --intervention-tokens audio , text , events , control , part1 , part2 , part3 , part4 , part5 , part6 , part7 , part8 , part9 , part10 , part11
```

Plot all available model outputs:

```bash
python experiments/experiment_31_ordering_intervention/plot_belief_shift_by_layer.py
```


Plot the compact wide paper line plot with `audio`, `text`, event-name,
control, and last-token interventions. By default this includes the paper
models `af-next`, `moss-audio`, and `qwen3-omni`

```bash
python experiments/experiment_31_ordering_intervention/plot_compact_results_for_paper.py
```

Save 5 default synthetic examples and metadata for inspection:

```bash
python experiments/experiment_31_ordering_intervention/synthetic_ordering_dataset.py --num-examples 5
```

This writes normal WAVs, swapped WAVs, per-example JSON files, and a
`metadata.jsonl` index to:

```text
outputs/synthetic_examples/
```


## Outputs

Results are written to:

```text
outputs/<model_id_slug>/ordering_intervention_syntheticsed_10s_<model>_layer<layer>_<tokens>.csv
```

Each row includes the model id, prompt, teacher-forced text, before/after probabilities, correct/swapped/intervened probabilities, intervention layer/token metadata, selected token indices, selected token strings, and top-10 next-token predictions.

Plots are written to `plots`