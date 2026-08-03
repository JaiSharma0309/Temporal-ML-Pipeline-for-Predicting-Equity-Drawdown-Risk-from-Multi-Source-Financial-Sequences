"""
train_sequence_model_pytorch.py
===============================

PyTorch sequence model for the equity drawdown risk project.

This script reuses the existing stage-1 modeling dataset and temporal split
logic, then trains a compact GRU classifier on rolling per-stock feature
windows.  It is meant to complement the tabular sklearn models in
train_drawdown_risk_models.py with a model that can learn temporal patterns
across the last N trading days.

Run:
    python src/train_sequence_model_pytorch.py

Fast smoke test:
    python src/train_sequence_model_pytorch.py --epochs 1 --max-train-samples 2000 --max-eval-samples 1000
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import train_drawdown_risk_models as stage1


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
OUT_DIR = REPO_ROOT / "results/stage1"
TABLES_DIR = OUT_DIR / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "pytorch_sequence_gru"
RANDOM_STATE = 42
DEFAULT_DATA_CANDIDATES = [
    REPO_ROOT / "data/processed/stage1_modeling_data.csv",
    REPO_ROOT / "data/processed/stage1_modeling_data_shortened.csv",
]


def set_seed(seed: int) -> None:
    """
    Make training as reproducible as the local device backend allows.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """
    Resolve a requested torch device, supporting an automatic best-effort mode.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit_transform_features(
    df: pd.DataFrame,
    train_row_ids: set[int],
    numeric_features: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Median-impute and standardize numeric features using training endpoints only,
    then append one-hot country/sector features.
    """
    numeric = df[numeric_features].replace([np.inf, -np.inf], np.nan)
    train_mask = df["_row_id"].isin(train_row_ids)

    medians = numeric.loc[train_mask].median()
    filled = numeric.fillna(medians).fillna(0.0)

    means = filled.loc[train_mask].mean()
    stds = filled.loc[train_mask].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    scaled = ((filled - means) / stds).astype(np.float32)

    cats = pd.get_dummies(
        df[stage1.CATEGORICAL_FEATURES].fillna("Unknown"),
        columns=stage1.CATEGORICAL_FEATURES,
        dtype=np.float32,
    )

    features = pd.concat([scaled, cats], axis=1)
    return features.to_numpy(dtype=np.float32), features.columns.tolist()


def resolve_data_path(requested: str | None) -> Path:
    """
    Use an explicit processed dataset path, or fall back to available local
    stage-1 exports.
    """
    if requested:
        path = Path(requested)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Requested data path does not exist: {path}")
        return path

    for path in DEFAULT_DATA_CANDIDATES:
        if path.exists():
            return path

    candidates = "\n  ".join(str(p) for p in DEFAULT_DATA_CANDIDATES)
    raise FileNotFoundError(
        "No processed modeling dataset found. Expected one of:\n"
        f"  {candidates}\n"
        "Run src/build_modeling_dataset.py or pass --data-path."
    )


def load_sequence_data(data_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """
    Load a processed stage-1 modeling table and add lightweight sequence-model
    features that do not require external data merges.
    """
    df = pd.read_csv(data_path)
    df[stage1.DATE_COL] = pd.to_datetime(df[stage1.DATE_COL])
    df = df.sort_values(stage1.DATE_COL).reset_index(drop=True)

    required = {
        stage1.DATE_COL,
        "symbol",
        *stage1.CATEGORICAL_FEATURES,
        stage1.TARGET_CLF,
        *stage1.BASE_NUMERIC_FEATURES,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{data_path} is missing required columns: {sorted(missing)}")

    df, regime_cols = stage1.add_regime_features(df)
    df, rank_cols = stage1.add_cross_sectional_ranks(df, stage1.FEATURES_TO_RANK.copy())

    numeric_features = [
        c for c in [*stage1.BASE_NUMERIC_FEATURES, *regime_cols, *rank_cols]
        if c in df.columns
    ]
    return df, numeric_features


def build_sequence_endpoints(
    df_sorted: pd.DataFrame,
    split_row_ids: set[int],
    lookback: int,
    max_samples: int | None = None,
) -> list[int]:
    """
    Build endpoint positions whose previous lookback-1 rows are from the same
    symbol and whose endpoint belongs to the requested split.
    """
    endpoints: list[int] = []

    for _, group in df_sorted.groupby("symbol", sort=False):
        positions = group.index.to_numpy()
        row_ids = group["_row_id"].to_numpy()
        eligible = np.isin(row_ids, list(split_row_ids))
        local_idxs = np.flatnonzero(eligible)
        local_idxs = local_idxs[local_idxs >= lookback - 1]
        endpoints.extend(positions[local_idxs].tolist())

    endpoints = sorted(endpoints)

    if max_samples is not None and len(endpoints) > max_samples:
        rng = np.random.default_rng(RANDOM_STATE)
        endpoints = sorted(rng.choice(endpoints, size=max_samples, replace=False).tolist())

    return endpoints


class EquitySequenceDataset(Dataset):
    """
    Lazy rolling-window dataset backed by one 2D feature matrix.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        endpoints: list[int],
        lookback: int,
    ) -> None:
        self.features = features
        self.labels = labels.astype(np.float32)
        self.endpoints = endpoints
        self.lookback = lookback

    def __len__(self) -> int:
        return len(self.endpoints)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = self.endpoints[idx]
        start = end - self.lookback + 1
        x = self.features[start : end + 1]
        y = self.labels[end]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


class GRUDrawdownClassifier(nn.Module):
    """
    Compact GRU classifier for rolling equity feature windows.
    """

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        last_hidden = h[-1]
        return self.head(last_hidden).squeeze(-1)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def top_k_event_rate(y_true: np.ndarray, y_score: np.ndarray, k_frac: float = 0.10) -> dict:
    k = max(1, int(np.ceil(len(y_true) * k_frac)))
    top_idx = np.argsort(-y_score)[:k]
    top_rate = float(np.mean(y_true[top_idx]))
    base_rate = float(np.mean(y_true))
    return {
        "k_frac": float(k_frac),
        "top_k_n": int(k),
        "top_k_event_rate": top_rate,
        "base_rate": base_rate,
        "lift": float(top_rate / base_rate) if base_rate > 0 else np.nan,
    }


def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    lift = top_k_event_rate(y_true, y_score, k_frac=0.10)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": safe_auc(y_true, y_score),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "top10_lift": lift["lift"],
        "top10_event_rate": lift["top_k_event_rate"],
        "base_rate": lift["base_rate"],
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += float(loss.item()) * len(yb)
        total_n += len(yb)

    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    scores = []

    for xb, yb in loader:
        logits = model(xb.to(device))
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        ys.append(yb.numpy())
        scores.append(prob)

    return np.concatenate(ys).astype(int), np.concatenate(scores)


def save_predictions(
    df_sorted: pd.DataFrame,
    endpoints: list[int],
    split_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> None:
    out = df_sorted.loc[endpoints, ["date", "symbol", "country", "sector"]].copy()
    out["y_true"] = y_true
    out["y_pred"] = (y_score >= 0.5).astype(int)
    out["y_score"] = y_score
    out.to_csv(TABLES_DIR / f"{MODEL_NAME}_{split_name}_predictions.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PyTorch GRU drawdown-risk sequence model.")
    parser.add_argument("--lookback", type=int, default=120, help="Trading-day window length.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Processed stage-1 CSV. Defaults to stage1_modeling_data.csv, then stage1_modeling_data_shortened.csv.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    args = parser.parse_args()

    set_seed(RANDOM_STATE)
    device = resolve_device(args.device)

    data_path = resolve_data_path(args.data_path)
    print(f"Loading stage-1 data and features for PyTorch sequence model...")
    print(f"  data: {data_path}")
    df, numeric_features = load_sequence_data(data_path)
    df["_row_id"] = np.arange(len(df))

    train_df, val_df, test_df = stage1.split_data(df)
    train_row_ids = set(train_df["_row_id"].tolist())
    val_row_ids = set(val_df["_row_id"].tolist())
    test_row_ids = set(test_df["_row_id"].tolist())

    df_sorted = df.sort_values(["symbol", stage1.DATE_COL]).reset_index(drop=True)
    features, feature_names = fit_transform_features(df_sorted, train_row_ids, numeric_features)
    labels = df_sorted[stage1.TARGET_CLF].astype(int).to_numpy()

    train_endpoints = build_sequence_endpoints(
        df_sorted, train_row_ids, args.lookback, max_samples=args.max_train_samples
    )
    val_endpoints = build_sequence_endpoints(
        df_sorted, val_row_ids, args.lookback, max_samples=args.max_eval_samples
    )
    test_endpoints = build_sequence_endpoints(
        df_sorted, test_row_ids, args.lookback, max_samples=args.max_eval_samples
    )

    if not train_endpoints or not val_endpoints or not test_endpoints:
        raise ValueError("One or more splits has no sequence windows. Try a shorter --lookback.")

    print(
        f"  windows: train={len(train_endpoints):,} "
        f"val={len(val_endpoints):,} test={len(test_endpoints):,}"
    )
    print(f"  lookback={args.lookback} | features/window={len(feature_names)} | device={device}")

    train_ds = EquitySequenceDataset(features, labels, train_endpoints, args.lookback)
    val_ds = EquitySequenceDataset(features, labels, val_endpoints, args.lookback)
    test_ds = EquitySequenceDataset(features, labels, test_endpoints, args.lookback)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = GRUDrawdownClassifier(
        n_features=features.shape[1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    train_labels = labels[train_endpoints]
    pos = max(float(train_labels.sum()), 1.0)
    neg = max(float(len(train_labels) - train_labels.sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_pr = -np.inf
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        y_val, s_val = predict(model, val_loader, device)
        val_metrics = evaluate_scores(y_val, s_val)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        print(
            f"  epoch {epoch:02d} | loss={train_loss:.4f} "
            f"val_PR={val_metrics['pr_auc']:.4f} "
            f"val_ROC={val_metrics['roc_auc']:.4f} "
            f"lift@10%={val_metrics['top10_lift']:.2f}x"
        )

        if val_metrics["pr_auc"] > best_pr:
            best_pr = val_metrics["pr_auc"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics_rows = []
    for split_name, loader, endpoints in [
        ("train", train_loader, train_endpoints),
        ("validation", val_loader, val_endpoints),
        ("test", test_loader, test_endpoints),
    ]:
        y_true, y_score = predict(model, loader, device)
        metrics = evaluate_scores(y_true, y_score)
        metrics.update({"model": MODEL_NAME, "split": split_name, "model_type": "sequence_classifier"})
        metrics_rows.append(metrics)
        save_predictions(df_sorted, endpoints, split_name, y_true, y_score)

    pd.DataFrame(history).to_csv(TABLES_DIR / f"{MODEL_NAME}_training_history.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(TABLES_DIR / f"{MODEL_NAME}_metrics.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "feature_names": feature_names,
            "numeric_features": numeric_features,
        },
        OUT_DIR / f"{MODEL_NAME}.pt",
    )

    test_metrics = next(row for row in metrics_rows if row["split"] == "test")
    print(
        f"\nSaved PyTorch sequence artifacts → {OUT_DIR}\n"
        f"  test PR={test_metrics['pr_auc']:.4f} "
        f"test ROC={test_metrics['roc_auc']:.4f} "
        f"test lift@10%={test_metrics['top10_lift']:.2f}x"
    )


if __name__ == "__main__":
    main()
