"""CUDA DDP supervised fine-tuning of Laya on MPC labels, telemetry-only prompt; MLX-compatible export.

Copied from lunar-laya/training/train.py; only the question set and the training metadata differ.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from laya.common import build_model, build_sequence, collate_items

from training.prompt import QUESTIONS

SOURCE = "aac6fef/laya-multilingual-mlx"
REVISION = "f2b4faf51023039425946074e2cf1361d2db11d5"


def torch_names(weights):
    """Inverse of the pinned Laya-MLX parameter-name conversion."""
    result = {}
    for name, value in weights.items():
        name = name.replace(".in_proj.weight", ".in_proj_weight").replace(".in_proj.bias", ".in_proj_bias")
        name = name.replace("scorer.layers.", "scorer.").replace("act_head.layers.", "act_head.")
        if name in result:
            raise ValueError(f"duplicate parameter {name}")
        result[name] = value
    return result


def tokenize(rows, tokenizer, cfg):
    items = []
    for row in rows:
        q = QUESTIONS[row["question"]]
        internal = {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}
        ids, markers = build_sequence(tokenizer, row["state"], internal, cfg["max_len"], cfg["head_max_len"])
        if len(markers) != 3:
            raise ValueError("expected three intact choice markers")
        items.append({"ids": ids, "markers": markers, "qtype": 0,
                      "label": list(q["criteria"]).index(row["label"])})
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoder-layers", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.encoder_layers < 0:
        parser.error("positive epochs/batch size and nonnegative encoder layers required")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    torch.manual_seed(718)
    random.seed(718)
    torch.set_num_threads(4)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    source = Path(snapshot_download(SOURCE, revision=REVISION, allow_patterns=[
        "model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/*"]))
    cfg = json.loads((source / "rl_agent_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(source / "tokenizer")
    model = build_model(cfg, str(source / "encoder"))
    model.load_state_dict(torch_names(load_file(str(source / "model.safetensors"))), strict=True)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    if args.encoder_layers:
        for layer in model.encoder.layers[-args.encoder_layers:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        for p in model.encoder.final_norm.parameters():
            p.requires_grad_(True)
    # Action/escalation head is unused by flight controls and excluded from this loss.
    for p in model.act_head.parameters():
        p.requires_grad_(False)
    model.cuda(local_rank)
    train_rows = [json.loads(s) for s in (args.data / "train.jsonl").read_text().splitlines()]
    val_rows = [json.loads(s) for s in (args.data / "validation.jsonl").read_text().splitlines()]
    train_items, val_items = tokenize(train_rows, tok, cfg), tokenize(val_rows, tok, cfg)

    def collate(batch):
        return collate_items([batch], tok.pad_token_id)

    sampler = DistributedSampler(train_items, num_replicas=world, rank=rank, shuffle=True, seed=718)
    loader = DataLoader(train_items, batch_size=args.batch_size, sampler=sampler, collate_fn=collate)
    val_loader = DataLoader(val_items, batch_size=args.batch_size, collate_fn=collate)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": [p for n,p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")], "lr": 1e-5},
        {"params": [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")], "lr": 1e-4},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(loader)*args.epochs, eta_min=1e-6)
    wrapped = DistributedDataParallel(model, device_ids=[local_rank]) if world > 1 else model
    history, best = [], -1
    start = time.perf_counter()

    def forward(batch):
        return {k: batch[k].cuda(local_rank) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        wrapped.train()
        loss_sum = 0.0
        for step, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = wrapped(**forward(batch))
                loss = torch.nn.functional.cross_entropy(logits, batch["label"].cuda(local_rank))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
            if rank == 0 and step % 25 == 0:
                print(json.dumps({"epoch": epoch+1, "step": step, "steps": len(loader), "loss": loss.item()}), flush=True)
        if world > 1:
            dist.barrier()
        if rank == 0:
            model.eval()
            predicted = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for batch in val_loader:
                    logits, _ = model(**forward(batch))
                    predicted.extend(logits.argmax(-1).cpu().tolist())
            accuracy = sum(p == r["label"] for p,r in zip(predicted, val_items)) / len(val_items)
            row = {"epoch": epoch+1, "rank0_train_loss": loss_sum/len(loader),
                   "validation_accuracy": accuracy,
                   "per_question": {q: sum(p == it["label"] for p,it,r in zip(predicted,val_items,val_rows) if r["question"] == q) /
                                    sum(r["question"] == q for r in val_rows) for q in QUESTIONS},
                   "elapsed_seconds": time.perf_counter()-start}
            history.append(row)
            print(json.dumps(row), flush=True)
            if accuracy > best:
                best = accuracy
                save_file({k:v.detach().cpu().half().contiguous() for k,v in model.state_dict().items()}, str(args.output / "model.safetensors"))
            (args.output / "history.json").write_text(json.dumps(history, indent=2)+"\n")
        if world > 1:
            dist.barrier()
    if rank == 0:
        cfg["temperature"] = [1,1,1]
        cfg["temperature_by_options"] = {}
        cfg["training"] = {"method": "supervised adaptive-MPC imitation, telemetry-only prompt; cross entropy",
                           "source": SOURCE, "source_revision": REVISION, "world_size": world,
                           "epochs": args.epochs, "encoder_layers": args.encoder_layers,
                           "trainable_parameters": sum(p.numel() for p in params),
                           "dataset": json.loads((args.data / "manifest.json").read_text()),
                           "best_validation_accuracy": best, "torch": torch.__version__}
        (args.output / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2)+"\n")
        shutil.copytree(source / "tokenizer", args.output / "tokenizer")
        shutil.copytree(source / "encoder", args.output / "encoder")
        model.load_state_dict(load_file(str(args.output / "model.safetensors")), strict=True)
        model.eval()
        fixtures = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for r,it in zip(val_rows[:32], val_items[:32]):
                logits, _ = model(**forward(collate([it])))
                fixtures.append({"row":r, "ids":it["ids"], "probabilities":logits.softmax(-1)[0].cpu().tolist()})
        (args.output / "parity.json").write_text(json.dumps(fixtures, indent=2)+"\n")
        with (args.output / "model.safetensors").open("rb") as weights:
            digest = hashlib.file_digest(weights, "sha256").hexdigest()
        (args.output / "weights.sha256").write_text(digest+"  model.safetensors\n")
        print(json.dumps({"complete": True, "best_validation_accuracy": best, "sha256": digest}), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
