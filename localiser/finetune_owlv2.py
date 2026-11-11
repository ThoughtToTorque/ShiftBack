#!/usr/bin/env python3
"""Fine-tune OWLv2 as a phrase-conditioned point localiser on LIBERO pre-grasp frames.

OWLv2's per-patch class scores are text-vs-appearance dot products computed per patch, so the
architecture is translation-equivariant by construction; fine-tuning only teaches name <-> appearance
for this object set, with labels derived from the gripper alone (``build_dataset.py``). A generic
whole-frame translate/scale augmentation and an optional label-patch relocation augmentation
(cut the patch around the label pixel, paste it elsewhere, fill the hole from another frame) remove
what little absolute-position signal the demonstrations leave.

Loss: CenterNet penalty-reduced focal loss on the 60x60 patch grid against a Gaussian around the
label pixel. Trainable: vision backbone (low lr) + class head. Frozen: text tower, box head,
objectness head. Point read-out: argmax patch centre (4.3 px at the 256 scale).

Inputs: ``--data`` (npz from build_dataset.py); ``--val`` / ``--val-base`` first frames of the
relocated and the demonstrated layouts dumped by ``sim/dump_first_frames.py --sim-truth``. The
simulator object position in those files is used ONLY to score the localiser (hit rate at < 20 px)
and, together with the held-out demonstration frames, to pick the checkpoint; the method itself
never reads it. Without validation frames the checkpoint is picked on the held-out frames alone.
Outputs: ``--out/best`` and ``--out/last`` (HuggingFace checkpoints) and ``--out/log.json``.

Example
    python localiser/finetune_owlv2.py --out outputs/localiser/owlv2_ft --reloc-p 0.5 --reloc-fill label
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from shiftback.libero_phrases import first_grasp_phrase  # noqa: E402

MID = "google/owlv2-base-patch16-ensemble"
GRID = 60  # 960 / 16


def load_val(path):
    """Frames with a scoring target (meta ``true`` from --sim-truth); rows without one are skipped."""
    if not path or not os.path.exists(path):
        return None, []
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    rows = []
    for i, m in enumerate(meta):
        ph = first_grasp_phrase(m["prompt"])
        if ph is None or m.get("true") is None:
            continue
        P = np.asarray(m["w2p"]); pt = P @ np.append(np.asarray(m["true"]), 1.0)
        rows.append(dict(i=i, suite=m["suite"], task=m["task"], phrase=ph, true=(pt[:2] / max(pt[2], 1e-6))))
    return d["imgs"], rows


def affine_aug(img, uv, rng, extra=None):
    """Whole-frame random translate/scale (label co-transformed). Generic detection aug, nothing object-specific."""
    s = float(rng.uniform(0.8, 1.25)); tx, ty = rng.uniform(-48, 48, 2)
    pil = Image.fromarray(img)
    # PIL affine maps OUTPUT (x,y) -> INPUT via the 6 coefficients: in = (a x + b y + c, d x + e y + f)
    a = 1.0 / s; c = 128 - a * (128 + tx); f = 128 - a * (128 + ty)
    out = pil.transform((256, 256), Image.AFFINE, (a, 0, c, 0, a, f), resample=Image.BILINEAR, fillcolor=(120, 110, 100))
    uv2 = np.array([(uv[0] - c) / a, (uv[1] - f) / a])
    if rng.uniform() < 0.8:
        out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.8, 1.2))
        out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.8, 1.2))
    if extra is not None:  # second target (duplicate-phrase rows), co-transformed
        return np.asarray(out), uv2, np.array([(extra[0] - c) / a, (extra[1] - f) / a])
    return np.asarray(out), uv2


def reloc_aug(img, uv, donor, rng, k=44, min_shift=48, feather=6, donor_uv=None):
    """Cut the k x k patch around the LABEL PIXEL (the only thing we know: where the gripper will close), paste it
    at a random place >= min_shift away, and fill the hole with a k x k crop of another training frame (so the old
    spot gets whatever happens to be there -- table, or another object, as in a real relocation). Nothing
    object-specific: no mask, no colour, no name. Feathered edges so the seam is not the cue."""
    H = 256
    cx, cy = int(round(uv[0])), int(round(uv[1]))
    # k may be a list of candidate sizes -> sample one, shrinking until the patch fits. Large patches carry
    # fixtures (cabinet top, rack, stove) with the label, so a release anchor cannot be learned as a location.
    if isinstance(k, (list, tuple)):
        k = int(rng.choice(k))
    while k > 24 and not (k // 2 <= cx < H - k // 2 and k // 2 <= cy < H - k // 2):
        k -= 8
    r = k // 2
    if not (r <= cx < H - r and r <= cy < H - r):
        return img, uv
    out = img.astype(np.float32).copy()
    patch = img[cy - r:cy + r, cx - r:cx + r].astype(np.float32)
    w = np.ones((k, k, 1), np.float32)
    for i in range(feather):
        a = (i + 1) / (feather + 1)
        w[i, :], w[k - 1 - i, :] = np.minimum(w[i, :], a), np.minimum(w[k - 1 - i, :], a)
        w[:, i], w[:, k - 1 - i] = np.minimum(w[:, i], a), np.minimum(w[:, k - 1 - i], a)
    # hole fill from a donor frame: a random crop, or the donor's OWN label patch (another demo's grasp target)
    # so that "the thing sitting at the demo spot" is an explicit negative -- exactly the relocation condition
    if donor_uv is not None and r <= int(round(donor_uv[0])) < H - r and r <= int(round(donor_uv[1])) < H - r:
        dx, dy = int(round(donor_uv[0])), int(round(donor_uv[1]))
    else:
        dx, dy = rng.integers(r, H - r, 2)
    fill = donor[dy - r:dy + r, dx - r:dx + r].astype(np.float32)
    out[cy - r:cy + r, cx - r:cx + r] = w * fill + (1 - w) * out[cy - r:cy + r, cx - r:cx + r]
    for _ in range(20):
        nx, ny = rng.integers(r, H - r, 2)
        if np.hypot(nx - cx, ny - cy) >= min_shift:
            break
    out[ny - r:ny + r, nx - r:nx + r] = w * patch + (1 - w) * out[ny - r:ny + r, nx - r:nx + r]
    return out.clip(0, 255).astype(np.uint8), np.array([uv[0] + (nx - cx), uv[1] + (ny - cy)], np.float64)


def gaussian_target(uv, sigma_patches=1.5):
    ys, xs = np.mgrid[0:GRID, 0:GRID]
    cx, cy = uv[0] / 256 * GRID - 0.5, uv[1] / 256 * GRID - 0.5
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma_patches ** 2)).reshape(-1)
    g[int(np.argmax(g))] = 1.0  # exactly one positive patch (CenterNet convention)
    return g.astype(np.float32)


def focal_loss(logits, target, alpha=2.0, beta=4.0):
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = (target >= 0.999).float()
    pos_loss = -(torch.log(p) * (1 - p) ** alpha * pos).sum()
    neg_loss = -(torch.log(1 - p) * p ** alpha * (1 - target) ** beta * (1 - pos)).sum()
    return (pos_loss + neg_loss) / pos.sum().clamp(min=1.0)


@torch.no_grad()
def predict(model, proc, imgs, phrases, device, bs=16):
    """-> (N,2) point in 256-px frame (argmax patch centre) and (N,) max prob."""
    pts, prob = [], []
    for k in range(0, len(imgs), bs):
        ims = [Image.fromarray(x) for x in imgs[k:k + bs]]
        inp = proc(text=[[f"a photo of a {p}"] for p in phrases[k:k + bs]], images=ims, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(**inp).logits[:, :, 0].float()
        j = lg.argmax(1)
        pts.append(torch.stack([(j % GRID + 0.5) / GRID * 256, (j // GRID + 0.5) / GRID * 256], 1).cpu().numpy())
        prob.append(torch.sigmoid(lg.max(1).values).cpu().numpy())
    return np.concatenate(pts), np.concatenate(prob)


def evaluate(model, proc, imgs, rows, device, tag):
    """Per-suite hit rate at < 20 px (~8 cm) against the simulator pixel; scoring only. {} without rows."""
    if not rows:
        return {}
    pts, prob = predict(model, proc, imgs[[r["i"] for r in rows]], [r["phrase"] for r in rows], device)
    err = np.array([np.hypot(*(p - r["true"])) for p, r in zip(pts, rows)])
    out = {}
    for s in sorted(set(r["suite"] for r in rows)):
        m = np.array([r["suite"] == s for r in rows])
        per_task = {}
        for t in sorted(set(r["task"] for r in rows if r["suite"] == s)):
            mt = m & np.array([r["task"] == t for r in rows])
            per_task[t] = float((err[mt] < 20).mean())
        out[s] = dict(hit20=float((err[m] < 20).mean()), median_px=float(np.median(err[m])), per_task=per_task)
        print(f"[{tag}] {s}: hit<20px {100*out[s]['hit20']:.0f}%  median {out[s]['median_px']:.1f}px  per-task "
              + " ".join(f"{int(10*v)}" for v in per_task.values()), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs", "localiser", "train_frames.npz"))
    ap.add_argument("--val", default=os.path.join(ROOT, "outputs", "first_frames", "swap_frames.npz"))
    ap.add_argument("--val-base", default=os.path.join(ROOT, "outputs", "first_frames", "base_frames.npz"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--aug-p", type=float, default=0.7)
    ap.add_argument("--reloc-p", type=float, default=0.0, help="probability of the label-patch relocation aug")
    ap.add_argument("--reloc-fill", default="random", choices=["random", "label"], help="hole fill: random donor crop | donor's label patch")
    ap.add_argument("--reloc-ks-release", default=None, help="patch sizes for role==1 (release) rows only, e.g. 44,64,96,128")
    ap.add_argument("--reloc-ks", default="44", help="comma list of relocation patch sizes, sampled per row")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--zero-shot-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    device = args.device
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    proc = Owlv2Processor.from_pretrained(MID)
    model = Owlv2ForObjectDetection.from_pretrained(MID).to(device)
    vimgs, vrows = load_val(args.val)
    bimgs, brows = load_val(args.val_base)
    log = []
    model.eval()
    r0 = evaluate(model, proc, vimgs, vrows, device, "zero-shot relocated")
    rb0 = evaluate(model, proc, bimgs, brows, device, "zero-shot base")
    log.append(dict(step=0, swap=r0, base=rb0))
    if args.zero_shot_only:
        return
    d = np.load(args.data)
    imgs, uv, phr, ep = d["image"], d["uv256"], d["phrase"], d["episode"]
    # optional SECOND target per row (duplicate phrase "moka pot" x2); NaN = none
    uvb = d["uv256_b"] if "uv256_b" in d.files else np.full_like(uv, np.nan)
    eps = np.unique(ep); rng.shuffle(eps)
    hold = set(eps[: max(1, len(eps) // 20)].tolist())
    tr = np.array([e not in hold for e in ep]); ho = ~tr
    tr_idx = np.nonzero(tr)[0]
    print(f"train frames {tr.sum()}  held-out frames {ho.sum()} ({len(hold)} episodes)", flush=True)
    # held-out in-distribution check set (sim-free: label = gripper pixel)
    ho_idx = np.nonzero(ho)[0][:: max(1, ho.sum() // 200)]
    # a RELOCATED copy of the held-out set, built once with a fixed seed from the same label-only reloc_aug: the
    # demo-spot shortcut can creep back late in training while the in-distribution score is blind to it
    _rr = np.random.default_rng(1234); ho_reloc = None
    reloc_ks = [int(x) for x in args.reloc_ks.split(",")]
    reloc_ks = reloc_ks[0] if len(reloc_ks) == 1 else reloc_ks
    role = d["role"] if "role" in d.files else np.zeros(len(uv), np.int64)
    reloc_ks_rel = [int(x) for x in args.reloc_ks_release.split(",")] if args.reloc_ks_release else None
    if args.reloc_p > 0:
        _ims, _uvs = [], []
        for k in ho_idx:
            _d = _rr.choice(tr_idx)
            im, pp = reloc_aug(imgs[k], uv[k].astype(np.float64), imgs[_d], _rr, donor_uv=(uv[_d] if args.reloc_fill == "label" else None)); _ims.append(im); _uvs.append(pp)
        ho_reloc = (np.stack(_ims), np.stack(_uvs))
    # freeze text tower, box head, objectness head
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("owlv2.vision_model") or n.startswith("class_head") or n.startswith("layer_norm")
    bb = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("owlv2.vision_model")]
    hd = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("owlv2.vision_model")]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr_backbone}, {"params": hd, "lr": args.lr_head}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    t0 = time.time(); best = -1.0
    for step in range(1, args.steps + 1):
        model.train()
        bi = rng.choice(tr_idx, args.bs, replace=False)
        ims, tg = [], []
        for k in bi:
            im, p = imgs[k], uv[k].astype(np.float64)
            pb = None if np.isnan(uvb[k][0]) else uvb[k].astype(np.float64)
            if rng.uniform() < args.reloc_p:
                _d = rng.choice(tr_idx)
                _ks = reloc_ks_rel if (reloc_ks_rel is not None and role[k] == 1) else reloc_ks
                im, p = reloc_aug(im, p, imgs[_d], rng, k=_ks, donor_uv=(uv[_d] if args.reloc_fill == "label" else None))
            if rng.uniform() < args.aug_p:
                if pb is None:
                    im, p = affine_aug(im, p, rng)
                else:
                    im, p, pb = affine_aug(im, p, rng, extra=pb)
            if not (0 <= p[0] < 256 and 0 <= p[1] < 256):
                im, p = imgs[k], uv[k].astype(np.float64); pb = None if np.isnan(uvb[k][0]) else uvb[k].astype(np.float64)
            t = gaussian_target(p)
            if pb is not None and 0 <= pb[0] < 256 and 0 <= pb[1] < 256:
                t = np.maximum(t, gaussian_target(pb))  # two positives (CenterNet multi-centre target)
            ims.append(Image.fromarray(im)); tg.append(t)
        inp = proc(text=[[f"a photo of a {phr[k]}"] for k in bi], images=ims, return_tensors="pt").to(device)
        target = torch.from_numpy(np.stack(tg)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**inp).logits[:, :, 0]
        loss = focal_loss(logits.float(), target)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(bb + hd, 1.0); opt.step(); sched.step()
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.5f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            pts, _ = predict(model, proc, imgs[ho_idx], [str(phr[k]) for k in ho_idx], device)
            e_ho = np.hypot(*(pts - uv[ho_idx]).T)
            _eb = np.hypot(*(pts - uvb[ho_idx]).T); e_ho = np.where(np.isnan(_eb), e_ho, np.minimum(e_ho, _eb))  # either target
            print(f"[held-out demos] hit<20px {100*(e_ho<20).mean():.0f}% median {np.median(e_ho):.1f}px", flush=True)
            r = evaluate(model, proc, vimgs, vrows, device, f"step {step} relocated")
            rb = evaluate(model, proc, bimgs, brows, device, f"step {step} base")
            hr = None
            if ho_reloc is not None:
                pts_r, _ = predict(model, proc, ho_reloc[0], [str(phr[k]) for k in ho_idx], device)
                e_r = np.hypot(*(pts_r - ho_reloc[1]).T); hr = float((e_r < 20).mean())
                print(f"[held-out RELOCATED] hit<20px {100*hr:.0f}% median {np.median(e_r):.1f}px", flush=True)
            log.append(dict(step=step, swap=r, base=rb, heldout_hit20=float((e_ho < 20).mean()), heldout_reloc_hit20=hr))
            (out / "log.json").write_text(json.dumps(log, indent=1))
            # checkpoint selection must NOT look at the relocated (test) frames: in-distribution signals only
            score = float((e_ho < 20).mean()) + (float(np.mean([v["hit20"] for v in rb.values()])) if rb else 0.0) + (hr if hr is not None else 0.0)
            if score > best:
                best = score; model.save_pretrained(out / "best"); proc.save_pretrained(out / "best")
                print(f"  saved best (score {score:.3f})", flush=True)
    model.save_pretrained(out / "last"); proc.save_pretrained(out / "last")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
