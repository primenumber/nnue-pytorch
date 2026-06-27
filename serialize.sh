#!/bin/sh
set -eux

# Pick the most recently modified checkpoint under lightning_logs
ckpt=$(find lightning_logs -name '*.ckpt' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

uv run serialize.py --features="HalfKAHm" "$ckpt" eval_issen/nn.nnue
