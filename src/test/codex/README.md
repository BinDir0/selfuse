# LegendVLA Codex Checks

This directory contains the pre-flight checks added for LegendVLA before long industrial-scale training.

## Pytest contracts

- `test_state_padding_contracts.py`
- `test_inference_wrapper_parity.py`
- `test_embedding_contracts.py`

Run them with:

```bash
pytest src/test/codex -q
```

## Real batch report

Exports one real batch, validates raw interface contracts, and writes an HTML report.

```bash
python -m src.test.codex.run_real_batch_report \
  --output_dir /tmp/legendvla_real_batch_report
```

## Embedding health check

Runs one or more real batches through the model, saves distribution plots, norm curves, and a t-SNE summary.

```bash
python -m src.test.codex.run_embedding_health_check \
  --output_dir /tmp/legendvla_embedding_report \
  --device cuda
```

## Activation health check

Runs one or more real batches through the full training forward path with hooks and saves activation statistics.

```bash
python -m src.test.codex.run_activation_health_check \
  --output_dir /tmp/legendvla_activation_report \
  --device cuda
```
