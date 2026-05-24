"""Train a model on the configured language(s).

Examples:
    python scripts/train.py --model tfidf_logreg --language afr
    python scripts/train.py --model afro_xlmr_base --language all
    python scripts/train.py --model afro_xlmr_base --language swa --override train.num_epochs=3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omegaconf import OmegaConf  # noqa: E402

from emotion_analysis import EMOTION_LABELS  # noqa: E402
from emotion_analysis.data.datasets import build_split, resolve_languages  # noqa: E402
from emotion_analysis.models.registry import build_model  # noqa: E402
from emotion_analysis.utils.config import load_config  # noqa: E402
from emotion_analysis.utils.seeds import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train emotion classifier")
    p.add_argument("--model", default=None, help="Model key from configs/models.yaml")
    p.add_argument("--language", default="all", help="Target ISO code, or 'all' for every target")
    p.add_argument("--override", nargs="*", default=[], help="Dotlist overrides, e.g. train.batch_size=8")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config("training", "data", "languages", "models", overrides=args.override)
    set_seed(cfg.seed)

    model_key = args.model or cfg.default
    spec = cfg.models[model_key]
    languages = resolve_languages(cfg, args.language)
    label_names = list(cfg.task.labels) or EMOTION_LABELS
    lang_tag = "all" if len(languages) > 1 else languages[0]

    print(f"[train] model={model_key} type={spec.type} languages={languages} seed={cfg.seed}")

    train_texts, train_labels, _ = build_split(cfg, languages, "train")
    dev_texts, dev_labels, _ = build_split(cfg, languages, "dev")
    print(f"[train] train={len(train_texts)} dev={len(dev_texts)}")

    output_dir = Path(cfg.paths.output_dir) / f"{model_key}_{lang_tag}"

    if spec.type == "sklearn":
        from emotion_analysis.training.trainer import train_baseline

        pipeline = build_model(model_key, num_labels=len(label_names))
        artifacts = train_baseline(
            pipeline,
            list(train_texts),
            train_labels,
            list(dev_texts),
            dev_labels,
            output_dir=output_dir,
            label_names=label_names,
        )
    elif spec.type == "transformer":
        import numpy as np

        from emotion_analysis.models.transformer import build_hf_dataset
        from emotion_analysis.training.trainer import train_transformer

        model, tokenizer = build_model(model_key, num_labels=len(label_names))
        train_ds = build_hf_dataset(train_texts, train_labels, tokenizer, spec.max_length)
        dev_ds = build_hf_dataset(dev_texts, dev_labels, tokenizer, spec.max_length)

        arr = np.asarray(train_labels, dtype="float32")
        pos_counts = arr.sum(axis=0)
        neg_counts = len(arr) - pos_counts
        pos_weight = (neg_counts / np.maximum(pos_counts, 1)).tolist()
        print(f"[train] pos_weight={[round(w,2) for w in pos_weight]}")

        artifacts = train_transformer(
            model,
            tokenizer,
            train_ds,
            dev_ds,
            config=cfg,
            output_dir=output_dir,
            label_names=label_names,
            pos_weight=pos_weight,
        )
    elif spec.type == "bilstm":
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from emotion_analysis.training.metrics import metrics_from_predictions
        from emotion_analysis.training.trainer import TrainArtifacts

        model, tokenizer = build_model(model_key, num_labels=len(label_names))

        def encode(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
            enc = tokenizer(list(texts), max_length=spec.max_length)
            return enc["input_ids"], enc["lengths"]

        train_ids, train_lens = encode(list(train_texts))
        dev_ids, dev_lens = encode(list(dev_texts))
        train_lbl = torch.tensor(np.asarray(train_labels, dtype="float32"))
        dev_lbl = torch.tensor(np.asarray(dev_labels, dtype="float32"))

        arr = np.asarray(train_labels, dtype="float32")
        pos_counts = arr.sum(axis=0)
        neg_counts = len(arr) - pos_counts
        pos_weight = torch.tensor(neg_counts / np.maximum(pos_counts, 1), dtype=torch.float32)
        print(f"[train] pos_weight={[round(w, 2) for w in pos_weight.tolist()]}")

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.num_epochs)

        train_loader = DataLoader(
            TensorDataset(train_ids, train_lens, train_lbl),
            batch_size=cfg.train.batch_size,
            shuffle=True,
        )
        best_f1, best_state = 0.0, None
        num_epochs = cfg.train.num_epochs
        patience, no_improve = cfg.train.early_stopping_patience, 0

        for epoch in range(1, num_epochs + 1):
            model.train()
            total_loss = 0.0
            for ids_b, lens_b, lbl_b in train_loader:
                optimizer.zero_grad()
                out = model(ids_b, lengths=lens_b)
                loss = criterion(out["logits"], lbl_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                dev_out = model(dev_ids, lengths=dev_lens)
                preds = (torch.sigmoid(dev_out["logits"]) >= float(cfg.task.threshold)).numpy().astype(int)
            metrics = metrics_from_predictions(preds, dev_lbl.numpy().astype(int), label_names=label_names)
            f1 = metrics["f1_macro"]
            print(f"[train] epoch={epoch} loss={total_loss/len(train_loader):.4f} dev_f1_macro={f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[train] early stopping at epoch {epoch}")
                    break

        if best_state:
            model.load_state_dict(best_state)

        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / "model.pt")
        tokenizer.save_pretrained(str(output_dir))
        import json as _json
        (output_dir / "bilstm_config.json").write_text(_json.dumps({
            "num_labels": model.config.num_labels,
            "vocab_size": model.config.vocab_size,
            "embed_dim": model.config.embed_dim,
            "hidden_dim": model.config.hidden_dim,
            "num_layers": model.config.num_layers,
            "dropout": model.config.dropout,
            "max_length": model.config.max_length,
        }))

        model.eval()
        with torch.no_grad():
            dev_out = model(dev_ids, lengths=dev_lens)
            preds = (torch.sigmoid(dev_out["logits"]) >= float(cfg.task.threshold)).numpy().astype(int)
        final_metrics = metrics_from_predictions(preds, dev_lbl.numpy().astype(int), label_names=label_names)
        artifacts = TrainArtifacts(model=model, tokenizer=tokenizer, metrics=final_metrics, output_dir=output_dir)
    else:
        raise SystemExit(f"Unknown model type: {spec.type}")

    (artifacts.output_dir / "dev_metrics.json").write_text(json.dumps(artifacts.metrics, indent=2))
    run_meta = {
        "model_key": model_key,
        "model_type": spec.type,
        "languages": languages,
        "label_names": label_names,
        "n_train": len(train_texts),
        "n_dev": len(dev_texts),
    }
    (artifacts.output_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2))
    OmegaConf.save(cfg, artifacts.output_dir / "resolved_config.yaml")

    print(f"[train] dev f1_macro={artifacts.metrics.get('f1_macro'):.4f}")
    print(f"[train] artifacts -> {artifacts.output_dir}")


if __name__ == "__main__":
    main()
