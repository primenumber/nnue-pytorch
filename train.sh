#!/bin/sh
set -eux

#uv run train.py --accelerator="gpu" --devices=1 --features="HalfKP" --num-workers=32 --batch-size=8192 --epoch-size=10000000 --max_epochs=1 /mnt/d/Shogi/Suishopsv-150m.bin /mnt/d/Shogi/ForValidation-Depth22-48k/ken.bin
#uv run train.py fit --trainer.accelerator="gpu" --trainer.devices=1 --model.features_name="HalfKP" --data.num_workers=32 --data.batch_size=8192 --data.epoch_size=10000000 --trainer.max_epochs=1 --data.train_filename=/mnt/d/Shogi/Suishopsv-150m.bin --data.val_filename=/mnt/d/Shogi/ForValidation-Depth22-48k/ken.bin --data.filtered=False --data.features_name="HalfKP" --data.random_fen_skipping=False --model.lr=8.75e-4 --model.start_lambda=1.0 --model.gamma=0.992 --model.in_scaling=240 --model.out_scaling=280 --model.offset=270 --model.adjust_loss=0.1 --seed_everything=42
uv run main.py fit --config=config.yaml
