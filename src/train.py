"""
Train the GRU on per card transaction sequences.

Run with:
    python -m src.train

Useful flags:
    --subsample-neg 0.1   keep every fraud and 10 percent of legitimate rows,
                          for training only. The test set is never touched.
    --epochs 3            shorten the run if time is tight.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src import config
from src.dataset import SequenceDataset, load_cache, split_indices, subsample_negatives
from src.model import GRUFraudModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the sequence fraud model.")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--seq-len", type=int, default=config.SEQ_LEN)
    p.add_argument("--threads", type=int, default=config.NUM_THREADS)
    p.add_argument(
        "--subsample-neg",
        type=float,
        default=1.0,
        help="Fraction of legitimate TRAINING rows to keep. 1.0 keeps all of them.",
    )
    return p.parse_args()


@torch.no_grad()
def score(model: nn.Module, loader: DataLoader) -> np.ndarray:
    """Return fraud probabilities in the loader's (unshuffled) row order."""
    model.eval()
    out = []
    for x_num, x_cat, _ in loader:
        logits = model(x_num, x_cat)
        out.append(torch.sigmoid(logits).numpy())
    return np.concatenate(out)


def main() -> None:
    args = parse_args()
    config.ensure_dirs()

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    # The i5-1235U exposes 12 threads across 2 performance and 8 efficiency
    # cores. Using them explicitly is the difference between a 2 minute epoch
    # and a 10 minute one, since there is no GPU to fall back on.
    torch.set_num_threads(args.threads)

    cache = load_cache()
    x_num, x_cat, y = cache["x_num"], cache["x_cat"], cache["y"]
    block_start, is_test = cache["block_start"], cache["is_test"]
    n_categories = int(x_cat.max())

    train_rows, test_rows = split_indices(is_test)
    if args.subsample_neg < 1.0:
        before = len(train_rows)
        train_rows = subsample_negatives(train_rows, y, args.subsample_neg)
        print(f"Subsampled training rows: {before:,} -> {len(train_rows):,} "
              f"(all frauds kept, {args.subsample_neg:.0%} of legitimate rows)")

    train_ds = SequenceDataset(x_num, x_cat, y, block_start, train_rows, args.seq_len)
    test_ds = SequenceDataset(x_num, x_cat, y, block_start, test_rows, args.seq_len)

    # num_workers=0 on purpose. Windows spawns rather than forks, so worker
    # processes would each re-pickle the feature arrays, which costs more than
    # the slicing they save on a dataset this size.
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=1024, shuffle=False, num_workers=0, drop_last=False
    )

    model = GRUFraudModel(n_numeric=x_num.shape[1] + 1, n_categories=n_categories)

    # The fraud rate is well under one percent, so an unweighted loss would be
    # minimised by predicting "legitimate" for everything. pos_weight rescales
    # the positive class by the negative to positive ratio.
    n_pos = float(y[train_rows].sum())
    n_neg = float(len(train_rows) - n_pos)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0), dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_batches = len(train_loader)
    print("-" * 72)
    print(f"Model parameters : {model.count_parameters():,}")
    print(f"Training rows    : {len(train_rows):,}  ({int(n_pos):,} fraud)")
    print(f"Test rows        : {len(test_rows):,}")
    print(f"Sequence length  : {args.seq_len}")
    print(f"pos_weight       : {pos_weight.item():.1f}")
    print(f"Batches per epoch: {n_batches:,}")
    print(f"Torch threads    : {torch.get_num_threads()}")
    print("-" * 72)

    step_losses: list[float] = []
    epoch_losses: list[float] = []
    report_every = max(1, n_batches // 20)
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen, epoch_total = 0.0, 0, 0.0
        t_epoch = time.time()

        for i, (x_n, x_c, target) in enumerate(train_loader, start=1):
            optimiser.zero_grad(set_to_none=True)
            logits = model(x_n, x_c)
            loss = criterion(logits, target)
            loss.backward()
            optimiser.step()

            running += loss.item()
            epoch_total += loss.item()
            seen += 1

            if i % report_every == 0:
                elapsed = time.time() - t_epoch
                rate = i / elapsed
                eta = (n_batches - i) / rate
                print(
                    f"  epoch {epoch}  batch {i:>5}/{n_batches}  "
                    f"loss {running / seen:.4f}  "
                    f"{rate:5.1f} batch/s  eta {eta / 60:4.1f} min"
                )
                step_losses.append(running / seen)
                running, seen = 0.0, 0

        mean_loss = epoch_total / n_batches
        epoch_losses.append(mean_loss)
        print(f"epoch {epoch} complete  mean loss {mean_loss:.4f}  "
              f"({(time.time() - t_epoch) / 60:.1f} min)")

    print(f"Training finished in {(time.time() - t_start) / 60:.1f} min")

    torch.save(model.state_dict(), config.GRU_CHECKPOINT)
    np.save(config.CACHE_DIR / "gru_step_losses.npy", np.array(step_losses))
    np.save(config.CACHE_DIR / "gru_epoch_losses.npy", np.array(epoch_losses))

    print("Scoring the test set ...")
    t0 = time.time()
    scores = score(model, test_loader)
    np.save(config.SCORES["gru"], scores)
    print(f"Wrote {config.SCORES['gru'].name} "
          f"({len(scores):,} scores, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
