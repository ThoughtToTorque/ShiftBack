#!/usr/bin/env python3
"""HTTP service (GPU machine): OWLv2 zero-shot detection + ShiftBack computation for the robot-side runner.

    python real/shiftback_service.py --port 8010 --calib data/real_sample/calib/P.npy --anchors data/real_sample/calib/anchors.json

POST /detect     {"image_png_b64": ..., "phrases": [...]}      -> {"detections": {phrase: {uv, box, score} | null}}
POST /shiftback  {"image_png_b64": ...}                        -> {"detections": ..., "shiftback": {...}}   (uses the anchors' phrases)
GET  /health                                                   -> {"ok": true}
"""
import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shiftback_real import OwlDetector, ShiftBack, decode_png_b64  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--dead-band-m", type=float, default=0.04)
    ap.add_argument("--log-dir", default=None, help="save every received first frame + result here")
    a = ap.parse_args()
    det = OwlDetector()
    sb = ShiftBack(a.calib, a.anchors, dead_band_m=a.dead_band_m)
    if a.log_dir:
        os.makedirs(a.log_dir, exist_ok=True)

    class H(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            self._send({"ok": True, "phrases": sb.phrases})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n).decode())
            img = decode_png_b64(req["image_png_b64"]); t0 = time.time()
            if self.path == "/detect":
                d = det.detect(img, req["phrases"]); self._send({"detections": d, "secs": time.time() - t0})
            elif self.path == "/shiftback":
                d = det.detect(img, sb.phrases); r = sb.compute(d)
                out = {"detections": d, "shiftback": r, "secs": time.time() - t0}
                if a.log_dir:
                    from PIL import Image
                    ts = time.strftime("%Y%m%d_%H%M%S"); Image.fromarray(img).save(os.path.join(a.log_dir, f"{ts}.png"))
                    json.dump(out, open(os.path.join(a.log_dir, f"{ts}.json"), "w"), indent=1)
                print(ts if a.log_dir else "", "pick", r["pick"]["delta_m"], r["pick"]["reason"], "| place", r["place"]["delta_m"], r["place"]["reason"], flush=True)
                self._send(out)
            else:
                self._send({"error": "unknown path"}, 404)

        def log_message(self, *args):
            pass

    print(f"shiftback service on :{a.port}  phrases {sb.phrases}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
