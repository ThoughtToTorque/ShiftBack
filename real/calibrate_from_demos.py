#!/usr/bin/env python3
"""Calibrate the fixed camera to the robot from the demonstrations themselves (no calibration target).

The demonstrations log the gripper pose at every step. In frames where the gripper is open and
visible, the two finger tips are at known 3-D positions (``shiftback.geometry.finger_tips``). Click
them in a couple of dozen frames and fit a pinhole camera; that is the only calibration ShiftBack
needs. Three steps:

    # 1. dump open-gripper frames + the logged poses from the demonstration files (.npz as written by
    #    collect_franka_keyboard.py: base_rgb, ee_pose (T,4,4), gripper_width, gripper_cmd)
    python real/calibrate_from_demos.py prepare --demos data_root/T1 --out calib_work

    # 2. click the finger tips in a browser (http://127.0.0.1:8030); the fit updates as you go
    python real/calibrate_from_demos.py serve --out calib_work

    # 3. (or offline) fit from the saved clicks -> calib_work/P.npy, calib_work/calib.json
    python real/calibrate_from_demos.py fit --out calib_work

The shipped sample ``data/real_sample/calib/clicked_frames.json`` holds 25 clicked frames of our
robot, so step 3 can be tried without a robot:

    python real/calibrate_from_demos.py fit --clicked data/real_sample/calib/clicked_frames.json --out /tmp/calib_demo

Model: pinhole with the principal point at the image centre and a nominal focal length ``--f0``
(the camera's spec value) held fixed, pose (6 dof) and the half finger thickness fitted with a
robust least squares (scipy). A free-focal fit is reported alongside for comparison. Holding the
intrinsics keeps the fit well conditioned when the clicked points cover a small volume.
"""
import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shiftback.geometry import finger_tips, fit_projection, lift, project, reprojection_error  # noqa: E402


# ------------------------------------------------------------------------------------------------- prepare
def prepare(a):
    from PIL import Image
    fdir = os.path.join(a.out, "frames")
    os.makedirs(fdir, exist_ok=True)
    eps = sorted(f for f in os.listdir(a.demos) if f.endswith(".npz"))
    eps = eps[::max(1, len(eps) // a.n_episodes)][:a.n_episodes]
    man = []
    for name in eps:
        d = np.load(os.path.join(a.demos, name))
        T, w, cmd, rgb = d["ee_pose"], d["gripper_width"], d["gripper_cmd"], d["base_rgb"]
        n = len(w)
        close = int(np.argmax(cmd > 0))
        opn = close + int(np.argmax(cmd[close:] < 0))
        slots = list(range(0, max(1, close - 6), a.step_down)) + [max(0, close - 6)]
        slots += list(range(opn + 25, n - 1, a.step_up)) + [n - 1]
        stem = name[:-4]
        for i in sorted(set(s for s in slots if 0 <= s < n)):
            if w[i] < a.min_width:  # fingers not fully open: tip positions unreliable
                continue
            fid = f"{stem}_f{i:04d}"
            Image.fromarray(rgb[i]).save(os.path.join(fdir, fid + ".jpg"), quality=92)
            man.append(dict(id=fid, img=f"frames/{fid}.jpg", episode=stem, frame=int(i),
                            phase="down" if i < close else "up", tcp=T[i, :3, 3].round(5).tolist(),
                            R=T[i, :3, :3].round(5).tolist(), width=float(w[i])))
        print(name, "close", close, "open", opn, "frames so far", len(man), flush=True)
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print("manifest:", len(man), "frames ->", a.out)


# ----------------------------------------------------------------------------------------------------- fit
def _rodrigues(r):
    r = np.asarray(r, np.float64)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _rotvec(R):
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-12:
        return np.zeros(3)
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def collect(rows, t=0.008):
    """Clicked rows -> (X, uv, finger axis per point, side per point)."""
    X, uv, axis, side = [], [], [], []
    for r in rows:
        if r.get("skip"):
            continue
        tl, tr = finger_tips(r["tcp"], r["R"], r["width"], tip_offset=t)
        ax = np.asarray(r["R"], np.float64)[:, 1]
        for key, X3, s in (("click_L", tl, -1), ("click_R", tr, 1)):
            if r.get(key):
                X.append(X3); uv.append(r[key]); axis.append(ax); side.append(s)
    return (np.asarray(X, np.float64).reshape(-1, 3), np.asarray(uv, np.float64).reshape(-1, 2),
            np.asarray(axis, np.float64).reshape(-1, 3), np.asarray(side, np.float64))


def fit_pinhole(rows, f0=615.0, cx=320.0, cy=240.0, fix_f=True, thr=5.0, t0=0.008):
    """Pose (+ optional focal length, + half finger thickness) by robust least squares."""
    from scipy.optimize import least_squares
    X0, uv, axis, side = collect(rows, t0)
    if len(X0) < 8:
        raise ValueError("need at least 8 clicked tips (4 frames)")
    K = np.array([[f0, 0, cx], [0, f0, cy], [0, 0, 1.0]])
    # initial pose from a DLT, decomposed against the nominal intrinsics
    Pd = fit_projection(X0, uv)
    M = np.linalg.inv(K) @ Pd
    s = np.sign(np.linalg.det(M[:, :3])) / np.cbrt(abs(np.linalg.det(M[:, :3])))
    M = M * s
    U, _, Vt = np.linalg.svd(M[:, :3])
    R0 = U @ Vt
    if np.linalg.det(R0) < 0:
        R0 = -R0
    t_init = M[:, 3]
    if (R0 @ X0.mean(0) + t_init)[2] < 0:  # camera must look at the points
        R0, t_init = -R0, -t_init
    p = np.concatenate([[f0], _rotvec(R0), t_init, [t0]])

    def build(p):
        f = f0 if fix_f else p[0]
        P = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]) @ np.concatenate([_rodrigues(p[1:4]), p[4:7, None]], 1)
        X = X0 + axis * ((p[7] - t0) * side)[:, None]
        return P, X, f

    inl = np.ones(len(X0), bool)

    def resid(p):
        P, X, _ = build(p)
        return (project(P, X[inl]).reshape(-1, 2) - uv[inl]).ravel()

    for _ in range(3):
        p = least_squares(resid, p, loss="soft_l1", f_scale=3.0).x
        P, X, f = build(p)
        err = np.linalg.norm(project(P, X).reshape(-1, 2) - uv, axis=1)
        inl = err < thr
    return P / P[2, 3], dict(f=float(f), cx=cx, cy=cy, rotvec=p[1:4].tolist(), tvec=p[4:7].tolist(),
                             t_finger_m=float(p[7]), rms_px=float(np.sqrt(np.mean(err[inl] ** 2))),
                             max_px=float(err.max()), inliers=int(inl.sum()), n=int(len(err)))


def _rows_from_work(out):
    man = json.load(open(os.path.join(out, "manifest.json")))
    cp = os.path.join(out, "clicks.json")
    clicks = json.load(open(cp)) if os.path.exists(cp) else {}
    rows = []
    for m in man:
        c = clicks.get(m["id"])
        if not c:
            continue
        rows.append(dict(m, click_L=c.get("L"), click_R=c.get("R"), skip=bool(c.get("skip"))))
    return rows


def do_fit(rows, a, save_to=None):
    res = {}
    for tag, fix in (("fixed_f", True), ("free_f", False)):
        try:
            P, info = fit_pinhole(rows, f0=a.f0, cx=a.cx, cy=a.cy, fix_f=fix, thr=a.thr)
        except Exception as e:  # noqa: BLE001
            res[tag] = dict(msg=str(e)); continue
        # what one decimetre on the table looks like in pixels, at the clicked points' mean height
        X0, _, _, _ = collect(rows)
        g = X0.mean(0)
        px_x = (project(P, g + [0.1, 0, 0]) - project(P, g)).round(1).tolist()
        px_y = (project(P, g + [0, 0.1, 0]) - project(P, g)).round(1).tolist()
        res[tag] = dict(info, px_per_10cm_x=px_x, px_per_10cm_y=px_y, P=P.tolist())
        if save_to and tag == "fixed_f":
            np.save(os.path.join(save_to, "P.npy"), P)
    if save_to:
        json.dump(res, open(os.path.join(save_to, "calib.json"), "w"), indent=1)
    return res


def fit_cmd(a):
    rows = json.load(open(a.clicked)) if a.clicked else _rows_from_work(a.out)
    os.makedirs(a.out, exist_ok=True)
    res = do_fit(rows, a, save_to=a.out)
    for tag in ("fixed_f", "free_f"):
        r = res[tag]
        if "msg" in r:
            print(tag, r["msg"]); continue
        print(f"[{tag}] f={r['f']:.0f}  rms {r['rms_px']:.2f} px  max {r['max_px']:.1f} px  inliers {r['inliers']}/{r['n']}  "
              f"finger half-thickness {1000 * r['t_finger_m']:.1f} mm  +10 cm x -> {r['px_per_10cm_x']} px  +10 cm y -> {r['px_per_10cm_y']} px")
    print("->", os.path.join(a.out, "P.npy"), "(fixed-f fit) and calib.json")


# --------------------------------------------------------------------------------------------------- serve
HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Click the finger tips</title>
<style>
body{margin:0;font-family:sans-serif;background:#222;color:#eee;display:flex}
#left{position:relative} canvas{display:block;cursor:crosshair}
#loupe{position:absolute;width:200px;height:200px;border:2px solid #0f0;border-radius:4px;pointer-events:none;display:none;background:#000}
#panel{padding:12px;width:520px;font-size:14px;line-height:1.5}
.k{color:#8f8}.w{color:#fc6}.e{color:#f66} pre{background:#111;padding:6px;white-space:pre-wrap;font-size:12px} button{margin:2px;padding:4px 8px}
</style></head><body>
<div id="left"><canvas id="c" width="960" height="720"></canvas><canvas id="loupe" width="200" height="200"></canvas></div>
<div id="panel">
<b>How to click</b>: two clicks per frame: first the tip of the <span class="e">left finger</span> as seen in the image (lowest point, centre of the pad), then the <span class="w">right finger</span>.
The next frame loads after the second click. If a tip is hidden press <b>S</b> to skip.<br>
Keys: <b>&larr; &rarr;</b> previous/next &nbsp;<b>Backspace</b> undo last click &nbsp;<b>S</b> skip &nbsp;<b>F</b> fit now &nbsp;<b>G</b> show/hide the fitted projection (green crosses = where the current calibration predicts the tips).
A 4x loupe follows the mouse.
<hr><div id="st"></div>
<div><button onclick="nav(-1)">prev</button><button onclick="nav(1)">next</button><button onclick="skip()">skip</button><button onclick="undo()">undo</button><button onclick="fit()">fit</button><button onclick="nextUnlabeled()">next unlabeled</button></div>
<hr><b>Fit</b> (refreshes every 6 frames; the fixed-f fit is the one saved as P.npy)<pre id="fit">nothing yet</pre>
</div>
<script>
let man=[],clicks={},cur=0,showProj=true,proj=null;const S=1.5;
const cv=document.getElementById('c'),ctx=cv.getContext('2d'),lp=document.getElementById('loupe'),lctx=lp.getContext('2d');
const img=new Image();
async function init(){man=await (await fetch('/api/manifest')).json();clicks=await (await fetch('/api/clicks')).json();cur=firstUnlabeled();load();fit();}
function firstUnlabeled(){for(let i=0;i<man.length;i++){const c=clicks[man[i].id];if(!c||(!c.skip&&!(c.L&&c.R)))return i;}return 0;}
function nextUnlabeled(){for(let i=cur+1;i<man.length;i++){const c=clicks[man[i].id];if(!c||(!c.skip&&!(c.L&&c.R))){cur=i;load();return;}}}
function load(){img.onload=draw;img.src='/'+man[cur].img+'?'+Date.now();status();}
function draw(){ctx.drawImage(img,0,0,cv.width,cv.height);const c=clicks[man[cur].id]||{};
 if(c.L)mark(c.L,'#f66','L');if(c.R)mark(c.R,'#fc6','R');
 if(showProj&&proj&&proj[man[cur].id]){const p=proj[man[cur].id];cross(p.L,'#0f0');cross(p.R,'#0f0');}}
function mark(p,col,t){ctx.strokeStyle=col;ctx.lineWidth=2;ctx.beginPath();ctx.arc(p[0]*S,p[1]*S,6,0,6.28);ctx.stroke();ctx.fillStyle=col;ctx.fillText(t,p[0]*S+8,p[1]*S-8);}
function cross(p,col){ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(p[0]*S-8,p[1]*S);ctx.lineTo(p[0]*S+8,p[1]*S);ctx.moveTo(p[0]*S,p[1]*S-8);ctx.lineTo(p[0]*S,p[1]*S+8);ctx.stroke();}
function status(){const m=man[cur],c=clicks[m.id]||{};let done=0,sk=0;for(const k in clicks){if(clicks[k].skip)sk++;else if(clicks[k].L&&clicks[k].R)done++;}
 document.getElementById('st').innerHTML=`frame <b>${cur+1}</b> / ${man.length} &nbsp; ${m.id} (${m.phase}) &nbsp; TCP z=${m.tcp[2].toFixed(3)} width ${(m.width*1000).toFixed(0)} mm<br>
 this frame: ${c.skip?'<span class=w>skipped</span>':(c.L?'L ok ':'L _ ')+(c.R?'R ok':'R _')} &nbsp;&nbsp; labelled <b>${done}</b>, skipped ${sk}`;}
cv.addEventListener('click',async e=>{const r=cv.getBoundingClientRect();const u=(e.clientX-r.left)/S,v=(e.clientY-r.top)/S;
 const id=man[cur].id;const c=clicks[id]||{};delete c.skip;
 if(!c.L)c.L=[+u.toFixed(1),+v.toFixed(1)];else if(!c.R)c.R=[+u.toFixed(1),+v.toFixed(1)];else{c.L=[+u.toFixed(1),+v.toFixed(1)];delete c.R;}
 clicks[id]=c;await save(id);draw();status();
 if(c.L&&c.R){let n=0;for(const k in clicks)if(clicks[k].L&&clicks[k].R)n++;if(n%6===0)fit();setTimeout(()=>{if(cur<man.length-1){cur++;load();}},250);}});
cv.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect();const x=e.clientX-r.left,y=e.clientY-r.top;
 lp.style.display='block';lp.style.left=(x+20>cv.width-200?x-220:x+20)+'px';lp.style.top=(y+20>cv.height-200?y-220:y+20)+'px';
 lctx.imageSmoothingEnabled=false;lctx.drawImage(cv,x-25,y-25,50,50,0,0,200,200);lctx.strokeStyle='#0f0';lctx.beginPath();lctx.moveTo(100,0);lctx.lineTo(100,200);lctx.moveTo(0,100);lctx.lineTo(200,100);lctx.stroke();});
cv.addEventListener('mouseleave',()=>lp.style.display='none');
async function save(id){await fetch('/api/save',{method:'POST',body:JSON.stringify({id:id,c:clicks[id]||{}})});}
function nav(d){cur=Math.max(0,Math.min(man.length-1,cur+d));load();}
async function skip(){const id=man[cur].id;clicks[id]={skip:true};await save(id);if(cur<man.length-1){cur++;load();}else{draw();status();}}
async function undo(){const id=man[cur].id;const c=clicks[id]||{};if(c.R)delete c.R;else if(c.L)delete c.L;else delete c.skip;clicks[id]=c;await save(id);draw();status();}
async function fit(){const r=await (await fetch('/api/fit')).json();proj=r.proj||null;delete r.proj;
 let s='';for(const t of ['fixed_f','free_f']){const f=r[t];if(!f)continue;s+=(t==='fixed_f'?'[f fixed] ':'[f free] ')+(f.msg?f.msg:`rms ${f.rms_px.toFixed(2)} px  max ${f.max_px.toFixed(1)} px  inliers ${f.inliers}/${f.n}  f=${f.f.toFixed(0)}  finger ${(1000*f.t_finger_m).toFixed(1)} mm\n  +10 cm x -> (${f.px_per_10cm_x}) px   +10 cm y -> (${f.px_per_10cm_y}) px`)+'\n';}
 document.getElementById('fit').textContent=s;draw();}
document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')nav(-1);else if(e.key==='ArrowRight')nav(1);else if(e.key==='Backspace'){e.preventDefault();undo();}
 else if(e.key==='s'||e.key==='S')skip();else if(e.key==='f'||e.key==='F')fit();else if(e.key==='g'||e.key==='G'){showProj=!showProj;draw();}});
init();
</script></body></html>"""


def serve(a):
    out = a.out
    man = json.load(open(os.path.join(out, "manifest.json")))
    cpath = os.path.join(out, "clicks.json")
    clicks = json.load(open(cpath)) if os.path.exists(cpath) else {}
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/":
                self._send(200, HTML.encode(), "text/html; charset=utf-8")
            elif p == "/api/manifest":
                self._send(200, json.dumps(man).encode())
            elif p == "/api/clicks":
                with lock:
                    self._send(200, json.dumps(clicks).encode())
            elif p == "/api/fit":
                with lock:
                    rows = _rows_from_work(out)
                res = do_fit(rows, a, save_to=out)
                if "P" in res.get("fixed_f", {}):
                    P = np.asarray(res["fixed_f"]["P"]); t = res["fixed_f"]["t_finger_m"]; proj = {}
                    for m in man:
                        L, R = finger_tips(m["tcp"], m["R"], m["width"], tip_offset=t)
                        proj[m["id"]] = dict(L=project(P, L).round(1).tolist(), R=project(P, R).round(1).tolist())
                    res["proj"] = proj
                self._send(200, json.dumps(res).encode())
            elif p.startswith("/frames/"):
                fn = os.path.join(out, "frames", os.path.basename(p))
                if os.path.exists(fn):
                    self._send(200, open(fn, "rb").read(), "image/jpeg")
                else:
                    self._send(404, b"")
            else:
                self._send(404, b"")

        def do_POST(self):
            if self.path == "/api/save":
                n = int(self.headers.get("Content-Length", 0)); d = json.loads(self.rfile.read(n))
                with lock:
                    if d["c"]:
                        clicks[d["id"]] = d["c"]
                    else:
                        clicks.pop(d["id"], None)
                    tmp = cpath + ".tmp"; json.dump(clicks, open(tmp, "w"), indent=0); os.replace(tmp, cpath)
                self._send(200, b"{}")
            else:
                self._send(404, b"")

    print(f"serving {len(man)} frames at http://127.0.0.1:{a.port}  clicks -> {cpath}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--demos", required=True, help="directory of demonstration .npz files")
    p.add_argument("--out", required=True); p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--step-down", type=int, default=15); p.add_argument("--step-up", type=int, default=20)
    p.add_argument("--min-width", type=float, default=0.06); p.set_defaults(fn=prepare)
    for name, fn in (("serve", serve), ("fit", fit_cmd)):
        p = sub.add_parser(name); p.add_argument("--out", required=True); p.add_argument("--port", type=int, default=8030)
        p.add_argument("--clicked", default=None, help="fit: a clicked_frames.json instead of <out>/manifest+clicks")
        p.add_argument("--f0", type=float, default=615.0, help="nominal focal length (px) of the camera")
        p.add_argument("--cx", type=float, default=320.0); p.add_argument("--cy", type=float, default=240.0)
        p.add_argument("--thr", type=float, default=5.0, help="inlier threshold (px)"); p.set_defaults(fn=fn)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
