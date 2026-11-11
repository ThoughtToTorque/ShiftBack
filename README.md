# ShiftBack

**Training-free correction of workpiece layout changes for imitation-learned robot manipulation policies.**

A policy taught by demonstration in one table layout fails when a part is moved: it reaches for
where the part used to be. ShiftBack leaves the policy frozen and moves its two inputs instead.
One detection on the first frame, projected onto the table at the height at which the gripper
closed in the demonstrations, gives the part's displacement. The arm then starts from a pose moved
by this displacement, and the fixed camera's image is moved back by it, so the policy sees the
layout it was taught in. At gripper closure the same is done for the placement.

No training, no object pose, no depth camera, no hand-made label. The only calibration is the
fixed camera's, done once (on our robot from the demonstrations themselves).

<p align="center">
<img src="assets/real_object_moved.gif" width="640" alt="tape moved 10 cm: frozen policy vs ShiftBack"><br>
<em>Tape moved 10 cm. Left: the frozen π0.5 policy closes its gripper where the tape used to be. Right: the same policy with ShiftBack.</em>
</p>
<p align="center">
<img src="assets/real_tray_moved.gif" width="640" alt="tray moved 9 cm: frozen policy vs ShiftBack"><br>
<em>Tray moved 9 cm. The correction is applied again at gripper closure, for the placement.</em>
</p>

| | frozen π0.5 | + ShiftBack |
|---|---|---|
| LIBERO-Pro relocation suites (object / spatial / goal / long), success % | 16 / 46 / 34 / 7 | 74 / 70 / 50 / 17 |
| Franka FR3, object exchanged with a neighbour, successes of 10 | 1 | 7 |
| Franka FR3, tray moved, successes of 10 | 0 | 7 |

## Three ways to try it

**Level 0, no GPU, a few seconds.** The whole computation on recorded robot frames: the shipped
projection matrix, the anchors from 40 demonstrations, the first frame of a trial in which the tape
was moved and the detections OWLv2 gave on it.

```bash
pip install -r requirements.txt            # numpy, pillow, matplotlib are enough for this level
python demos/00_shift_back_no_gpu.py       # object moved 10 cm
python demos/00_shift_back_no_gpu.py --scene tray_moved
python -m pytest tests                     # checks the core against the numbers the robot used
```

<p align="center"><img src="assets/demo0_object_moved.png" width="900"></p>

<p align="center">
<img src="assets/sim_object_swap.gif" width="768" alt="LIBERO-Pro object exchange: frozen policy, ShiftBack, and what the policy sees"><br>
<em>LIBERO-Pro, objects exchanged (level-1 demo output). Left: frozen π0.5. Middle: the same policy with ShiftBack. Right: the shifted frame the policy actually receives, the vacated strip filled with the edge pixels.</em>
</p>

**Level 1, one GPU, about half an hour.** A frozen π0.5 on one LIBERO-Pro task, with and without
ShiftBack, paired on identical initial states and policy noise, with side-by-side videos of what
the policy saw. See [docs/REPRODUCE.md](docs/REPRODUCE.md) for the one-time set-up (openpi, LIBERO-Pro).

```bash
# terminal 1 (openpi checkout): serve the released checkpoint, never changed
uv run scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config pi05_libero --policy.dir gs://openpi-assets/checkpoints/pi05_libero
# terminal 2
python sim/eval_paired.py --suite libero_object_swap --task-ids 0 --trials 5 --port 8000 --save-videos --out outputs/smoke
```

**Level 2, full reproduction.** `sim/run_paired.sh` runs the four relocation suites and the retention
suites under the official protocol; the ablation switches of `sim/eval_paired.py` give the rows of
the ablation table. [docs/REPRODUCE.md](docs/REPRODUCE.md) lists the commands for every table.

## What is in the box

```
shiftback/          the method, pure numpy (torch + transformers only for the localiser)
  geometry.py       pixel <-> table-plane points through the fixed camera; calibration from correspondences
  anchors.py        grasp / release anchors from the gripper events of the demonstrations (three-step rule)
  shift.py          the image shift with edge filling; the minimum-shift threshold
  localiser.py      phrase extraction from the instruction; OWLv2 as a phrase -> pixel localiser
  planner.py        first-frame detections -> start-pose shift, image shift, carry at closure
demos/              level-0 demo
sim/                LIBERO / LIBERO-Pro: paired evaluation, anchor and localiser-pixel tables, statistics
localiser/          fine-tuning OWLv2 as a point localiser on the demonstrations (benchmark only; the robot uses it zero-shot)
real/               Franka kit: calibration from demonstrations, anchors, detection service, episode runner
data/real_sample/   40 demonstration first frames + gripper logs, 25 clicked calibration frames, 3 recorded trials
data/libero/        the anchor and localiser-pixel tables used for the paper's numbers
```

The two fine-tuned OWLv2 checkpoints used on the benchmark (one for the grasp point, one for the
release point) are on the [Releases](https://github.com/ThoughtToTorque/ShiftBack/releases) page;
`sim/eval_paired.py` runs with the shipped pixel tables without them.

## How it works, in one screen

```python
from shiftback import ShiftBackPlanner, load_projection
P = load_projection("data/real_sample/calib/P.npy")           # fixed camera, robot base -> pixel
planner = ShiftBackPlanner(P, grasp_uv=[112, 252], grasp_z=0.066,   # where the gripper closed in the demos
                           release_uv=[365, 279], release_z=0.082, tau=0.04)
plan = planner.plan(det_obj={"uv": [60, 324]}, det_place={"uv": [366, 280]})   # one detection on the first frame
plan.pick.delta      # -> [-0.099, 0.018, 0]  move the arm's start pose by this (metres)
plan.pick.duv        # -> [52, -73]           move every fixed-camera frame by this (pixels)
plan.carry           # -> [0.099, -0.018, 0]  carry the held object by this at gripper closure
frame_for_policy = planner.shift_frame(frame, "pick")
```

## Citation

The paper is under review; a citation will be added here. Until then please cite this repository.

## License

MIT. OWLv2, openpi and LIBERO are used under their own licences.
