"""Train GPT-2 on WikiText-2 with a checkpoint policy attached.

The batch for step i depends only on i and the seed, and the random state
is reset from the step number before every step. A restored job therefore
sees the same batches and draws the same dropout masks, so its losses can
be compared with the ones logged before the failure.

    python examples/train_gpt2.py --model gpt2-large --policy checkfreq:0.01 \
        --ckpt-dir /mnt/store/ckpt --record run/train.jsonl

Policies: none, fixed:N, dcp:N, checkfreq:BUDGET, failure-aware:MTBF_SECONDS.
`--model tiny` runs a two-layer model on random tokens, for tests.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import torch

from facp.checkpoint import make_checkpointer
from facp.policies import from_spec
from facp.record import Record


def load_tokens(model_name: str, data_dir: Path, seed: int) -> torch.Tensor:
    if model_name == "tiny":
        g = torch.Generator().manual_seed(seed)
        return torch.randint(0, 1000, (200_000,), generator=g)
    cache = data_dir / "wikitext2_train_gpt2.pt"
    if cache.exists():
        return torch.load(cache)
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    text = "\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"])
    tokens = torch.tensor(GPT2TokenizerFast.from_pretrained("gpt2")(text)["input_ids"])
    data_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tokens, cache)
    return tokens


def build_model(name: str, grad_ckpt: bool) -> torch.nn.Module:
    from transformers import GPT2Config, GPT2LMHeadModel

    if name == "tiny":
        config = GPT2Config(n_layer=2, n_embd=128, n_head=2, vocab_size=1000, n_positions=256)
    else:
        config = GPT2Config.from_pretrained(name)
    model = GPT2LMHeadModel(config)  # trained from scratch: the weights do not matter here
    if grad_ckpt:
        model.gradient_checkpointing_enable()
    return model


def batch_for(step: int, tokens: torch.Tensor, batch: int, seq: int, seed: int) -> torch.Tensor:
    n_blocks = (len(tokens) - 1) // seq
    g = torch.Generator().manual_seed(seed * 1_000_003 + step)
    idx = torch.randint(0, n_blocks, (batch,), generator=g)
    return torch.stack([tokens[i * seq:(i + 1) * seq] for i in idx.tolist()])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt2-large")
    p.add_argument("--policy", default="none")
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--record", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--steps", type=int, default=10**9, help="stop after this many steps")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad-ckpt", action="store_true", help="recompute activations to save GPU memory")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    record = Record(args.record, "train")
    record.write("start", incarnation=int(os.environ.get("FACP_INCARNATION", "0")),
                 pid=os.getpid(), args=vars(args))

    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))

    device = torch.device(args.device)
    tokens = load_tokens(args.model, Path(args.data_dir), args.seed)
    # seeded, so a job killed before its first checkpoint restarts from the same weights
    torch.manual_seed(args.seed)
    model = build_model(args.model, args.grad_ckpt).to(device)
    # foreach=False updates one tensor at a time; the default batched update
    # needs about 3 GB of temporaries for GPT-2 large and does not fit a T4
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=False)

    ckpt = None
    step = 0
    if args.policy != "none":
        ckpt = make_checkpointer(args.policy, args.ckpt_dir, model, optimizer, record, from_spec)
        step = ckpt.restore()
    record.write("ready", step=step)

    model.train()
    while step < args.steps and not stop["now"]:
        step += 1
        began = time.time()
        torch.manual_seed(args.seed * 7919 + step)
        x = batch_for(step, tokens, args.batch, args.seq, args.seed).to(device)
        loss = model(x, labels=x).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        value = loss.item()  # synchronises with the GPU, so the step time is real
        seconds = time.time() - began
        record.write("loss", step=step, loss=value)
        if ckpt is not None:
            ckpt.after_step(step, seconds)
        else:
            record.write("step_time", step=step, seconds=seconds, writing=False)

    if stop["now"]:
        record.write("stopped", step=step)
        record.close()
        os._exit(0)  # the run's time is up; an unfinished write is not waited for
    if ckpt is not None:
        ckpt.finish()
    record.write("done", step=step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
