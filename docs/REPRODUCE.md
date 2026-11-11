# Reproducing the simulation results

All simulation numbers come from `sim/eval_paired.py`, which runs a frozen policy served by openpi
with and without ShiftBack on identical initial states and identical policy noise.

## One-time set-up

1. **openpi** (the policy server). Follow https://github.com/Physical-Intelligence/openpi; the
   released `pi05_libero` checkpoint is downloaded on first use. Nothing in openpi is modified.
2. **LIBERO-Pro** (the relocation suites). Clone https://github.com/Zxy-MLlab/LIBERO-PRO (we used
   commit `eafdb80`) and install it into the same Python environment as openpi's LIBERO example
   (`openpi/examples/libero`). The standard suites (`libero_object`, ...) are in the same checkout.
   `sim/eval_paired.py --libero-root <checkout>` writes LIBERO's `config.yaml` for you.
3. **This repository** on the Python path of that environment (`pip install -e .` or set
   `PYTHONPATH`), plus `openpi/packages/openpi-client`.

Environment for the evaluation process (MuJoCo off-screen rendering):

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_ROOT=/path/to/LIBERO-PRO
```

For the policy server we set `XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"`:
without it XLA may autotune a different kernel per process and the same episode can end differently
in two runs; with it two runs are bit-identical, so gained/lost counts measure the method and not
run-to-run noise.

## Level 1: one task, paired, with videos

```bash
uv run scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config pi05_libero --policy.dir gs://openpi-assets/checkpoints/pi05_libero   # in openpi/
python sim/eval_paired.py --suite libero_object_swap --task-ids 0 --trials 5 --port 8000 --save-videos --out outputs/smoke
```

`outputs/smoke/videos/` holds, per episode, the raw fixed-camera video of each mode and the video
of what the policy actually saw under ShiftBack (`*_policy_view.mp4`). `metrics.json` has the
success counts and the paired gained/lost numbers.

## Table I: relocation suites and retention (official protocol)

```bash
bash sim/run_paired.sh outputs/table1          # 50 initial states per task, four relocation + four standard suites
python sim/paired_stats.py outputs/table1/libero_object_swap/rows.json --cluster-bootstrap
```

Equivalent single commands:

```bash
for s in libero_object_swap libero_spatial_swap libero_goal_swap libero_10_swap libero_object libero_spatial libero_goal libero_10; do
  python sim/eval_paired.py --suite $s --task-ids 0-9 --trials 50 --port 8000 --out outputs/table1/$s
done
```

Seed robustness (bottom of Table I): `--trials 10 --policy-noise-seed {20260722,20261001,20261002}`.

## Table II: ablations (ten initial states per task, first seed)

| row | switches |
|---|---|
| image only, oracle Δ | `--no-arm-shift --no-place` (the paper's row used the simulator's displacement, see note) |
| start pose only | `--no-image-shift --no-place` |
| start pose + image | `--no-place` |
| + placement-side shift | default |
| placement: carry only | `--place-carry-only` |
| placement: image only | `--place-image-only` |
| fill of the vacated strip | `--fill gray` / `--fill mirror` |
| localiser: zero-shot VLM / OWLv2 grasp only / grasp + release | `--pixels <table>` built with `sim/build_localiser_pixels.py` from the respective model; the shipped table is grasp + release |

Note: the "oracle Δ" rows take the displacement from the simulator's object position instead of
the localiser. They are a diagnostic ceiling and the method never uses them; they are not part of
this release's evaluation path.

## Operating envelope (Fig. 5)

- Localiser error: `--pixel-noise 4|8|16|32` (Gaussian noise on the detected pixels, 256-px frame).
- Threshold: `--tau 0|0.01|0.02|0.04|0.06` on the relocation and standard suites.
- Calibration error: `--calib-perturb focal=0.05`, `focal=0.10`, `focal=-0.10`, `yaw_deg=2`, `yaw_deg=5`.

## Other frozen policies (Table IV)

Any server that speaks the openpi websocket protocol works: π0 (`--policy.config pi0_libero`,
`--action-horizon 50`), and policies that read the arm state need `--shift-state` so the
reading is expressed in the demonstrated frame.

## Building the tables yourself

The shipped `data/libero/*.json` can be regenerated from the LIBERO demonstrations:

```bash
python sim/build_anchors_libero.py --out data/libero                     # anchors_grasp.json, anchors_stages.json from the gripper events
python sim/dump_first_frames.py --suites libero_object_swap,libero_spatial_swap,libero_goal_swap,libero_10_swap --n-init 50 --out outputs/frames/swap.npz
python sim/dump_first_frames.py --suites libero_object,libero_spatial,libero_goal,libero_10 --n-init 50 --out outputs/frames/base.npz
python localiser/build_dataset.py --with-place --out outputs/localiser_train.npz   # pre-grasp frames labelled with the projected gripper events
python localiser/finetune_owlv2.py --data outputs/localiser_train.npz --out outputs/owlv2_ft
python sim/build_localiser_pixels.py --model outputs/owlv2_ft/best --frames outputs/frames/swap.npz outputs/frames/base.npz --out data/libero/localiser_pixels_mine.json
```

The checkpoints behind the shipped pixel table are on the Releases page (`owlv2_grasp_localiser`,
`owlv2_release_localiser`).

## Notes on regenerating the tables

- The shipped anchor tables were built with 79 demonstrations held out as a validation split; the
  builder uses every demonstration unless `--exclude-episodes` is given, which moves the anchors by
  a few millimetres and `n_episodes` by about two per prompt. The two prompts without any gripper
  closure (`push the plate ...`, `open the middle drawer ...`) have no grasp anchor and are left
  uncorrected by the evaluation.
- `sim/build_localiser_pixels.py` keeps the batched bfloat16 inference of the original so the
  pixels are bit-for-bit those of the shipped table when run with the released checkpoints.
- `localiser/finetune_owlv2.py` selects its checkpoint by the hit rate of the predicted pixel against
  the simulator's object pixel on held-out first frames (`sim/dump_first_frames.py --sim-truth`). This
  is validation only: no simulator quantity enters the method or the tables it uses at test time.
