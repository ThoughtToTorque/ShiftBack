# ShiftBack on a real robot

This is what we ran on a Franka FR3 with a fixed RGB camera and a wrist camera, with a π0.5 policy
fine-tuned on 40 demonstrations of "put the tape on the plate". Everything below is in `real/` and
needs no object pose, no depth camera and no hand-made label; the only calibration is the fixed
camera's, done once from the demonstrations.

## What the method needs from you

| input | where it comes from |
|---|---|
| demonstrations that log the gripper position and the gripper command | `real/collect_franka_keyboard.py` (or your own recorder: `.npz` with `base_rgb`, `ee_pose` (T,4,4), `gripper_cmd`, `gripper_width`) |
| the fixed camera's projection matrix `P` (3x4, robot base -> pixel) | `real/calibrate_from_demos.py` (click the finger tips in ~25 demonstration frames) or any standard calibration |
| anchors: demonstrated object/placement pixels, grasp and release heights | `real/build_anchors.py` (gripper events + OWLv2 on the demonstrations' first frames) |
| a localiser for the two phrases of the instruction | released OWLv2, zero-shot (`shiftback.localiser.OwlLocaliser`) |

## Step by step

1. **Record demonstrations** in one layout. The recorder stores both camera streams, the joint
   state, the end-effector pose and the commanded gripper state at 15 Hz.

2. **Calibrate the camera from the demonstrations.**
   ```bash
   python real/calibrate_from_demos.py prepare --demos data_root/T1 --out calib_work   # dumps open-gripper frames
   python real/calibrate_from_demos.py serve   --out calib_work                        # click tips at http://127.0.0.1:8030
   python real/calibrate_from_demos.py fit     --out calib_work                        # -> calib_work/P.npy
   ```
   The finger tips' 3-D positions follow from the logged gripper pose and width, so the clicked
   pixels and the logged poses are the correspondences; a pinhole with the camera's nominal focal
   length is fitted to them. On our camera the fit has a 2.6 px rms residual. You can try the fit on
   the shipped clicks without a robot:
   ```bash
   python real/calibrate_from_demos.py fit --clicked data/real_sample/calib/clicked_frames.json --out /tmp/calib_demo
   ```
   The matrix it produces gives the same displacement for the recorded trial as the one we used
   (see `tests/test_core.py`).

3. **Build the anchors.**
   ```bash
   python real/build_anchors.py --demos data_root/T1 --calib calib_work/P.npy \
       --object-phrase "roll of tape" "a green roll of tape on the table" --place-phrase "black round tray" \
       --out calib_work/anchors.json
   ```
   The grasp point is the mean gripper position at the first commanded closure of the demonstrations
   (three-step rule), the release point the mean position at the following opening, and the two
   heights are their z components. The demonstrated pixels are the mean OWLv2 detections on the 40
   first frames. The phrases are the only text you write.

4. **Run an episode.** The policy runs on a GPU machine behind an openpi policy server; detection
   runs either on the same machine (`real/shiftback_service.py`) or on the robot PC.
   ```bash
   # GPU machine
   python real/shiftback_service.py --port 8010 --calib calib_work/P.npy --anchors calib_work/anchors.json
   # robot PC (next to your robot/camera primitives, see below)
   python real/run_episode.py --mode shiftback --tag trial_01 --policy-host <gpu-ip> --detector http://<gpu-ip>:8010
   python real/run_episode.py --mode baseline  --tag trial_01 --policy-host <gpu-ip>
   ```
   `run_episode.py` moves the arm from its home pose by the pick displacement with the fingers open,
   passes every fixed-camera frame through the image shift, and at the first commanded closure lifts
   the held object, carries it by the placement displacement and lowers it before handing control
   back to the policy with the image shift switched to the placement side. At the end it asks the
   operator for the outcome and stores everything under `--results/<tag>/`.

   `run_episode.py` and `collect_calib_frames.py` import three robot-side modules that are specific
   to our cell and are not part of this release: `auto_collect.AutoCollector` (robot, gripper,
   cameras, `move_to`, taught poses), `robot` (quaternion helpers) and `cameras`. Implement the same
   small interface for your robot (`get_state`, `move_to`, `gripper`, camera `get_all`) or replace
   the calls; everything ShiftBack-specific is in `shiftback_real.py` and the library.

## Recorded trials shipped with the release

`data/real_sample/live/` holds the first frames (fixed and wrist camera) of three recorded trials,
the detections OWLv2 returned on them and the displacements the robot used, and
`demos/00_shift_back_no_gpu.py` recomputes them. The side-by-side videos in `assets/` are the raw
camera recordings of the frozen policy and of ShiftBack in the same layouts.
