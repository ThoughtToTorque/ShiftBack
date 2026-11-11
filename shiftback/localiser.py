"""Phrase extraction from the instruction and the OWLv2 localiser.

The localiser returns one pixel per phrase on the first frame. Two read-outs are supported:
``"box"`` (the centre of the highest-scoring box, for the released zero-shot OWLv2, as on our
robot) and ``"point"`` (the peak of the per-patch class logits, for a checkpoint fine-tuned with
``localiser/finetune_owlv2.py`` as on the benchmark).
"""
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

OWLV2_ID = "google/owlv2-base-patch16-ensemble"


def phrases_from_instruction(instruction: str) -> Tuple[Optional[str], Optional[str]]:
    """``(object phrase, placement phrase)`` read from a pick-and-place instruction.

    Handles ``"pick up the X and place it in/on the Y"`` and ``"put the X on/in the Y"``;
    the placement phrase is ``None`` when the instruction has no destination clause.
    """
    s = instruction.strip().rstrip(".").lower()
    m = re.match(r"pick up the (.+?) and (?:place|put) it (in|on|into|onto) the (.+)$", s)
    if m:
        return m.group(1), m.group(3)
    m = re.match(r"(?:put|place) the (.+?) (?:on|in|into|onto|to) (?:the )?(.+)$", s)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r"(?:pick up|grasp|take) the (.+)$", s)
    if m:
        return m.group(1), None
    return None, None


def _post(proc):
    """transformers >= 5 renamed the OWLv2 post-processor."""
    return getattr(proc, "post_process_object_detection", None) or proc.post_process_grounded_object_detection


class OwlLocaliser:
    """OWLv2 (HuggingFace transformers) as a phrase -> pixel localiser.

    ``model_id`` is the released checkpoint or a directory saved by ``localiser/finetune_owlv2.py``.
    Works on CPU (a few seconds per image) or GPU.
    """

    def __init__(self, model_id: str = OWLV2_ID, device: Optional[str] = None, grid: int = 60):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.grid = grid  # 960 / 16 patches per side

    def boxes(self, img_rgb: np.ndarray, phrases: Sequence[str]) -> Dict[str, Optional[Dict]]:
        """Best box per phrase: ``{phrase: {uv, box, score} | None}`` in the image's pixel frame."""
        from PIL import Image
        torch = self.torch
        im = Image.fromarray(img_rgb)
        with torch.no_grad():
            inp = self.proc(text=[list(phrases)], images=im, return_tensors="pt").to(self.device)
            out = self.model(**inp)
            side = max(im.size)  # OWLv2 pads the image to a square before resizing
            post = _post(self.proc)(out, threshold=0.0, target_sizes=torch.tensor([[side, side]]).to(self.device))[0]
        res: Dict[str, Optional[Dict]] = {}
        for k, ph in enumerate(phrases):
            m = post["labels"] == k
            sc, bx = post["scores"][m].cpu().numpy(), post["boxes"][m].cpu().numpy()
            if len(sc) == 0:
                res[ph] = None
                continue
            i = int(np.argmax(sc))
            b = bx[i]
            res[ph] = dict(uv=[float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)],
                           box=[float(x) for x in b], score=float(sc[i]))
        return res

    def points(self, img_rgb: np.ndarray, phrases: Sequence[str], n_peaks: int = 1,
               nms_patches: int = 6, prefix: str = "a photo of a ") -> Dict[str, List[Tuple[List[float], float]]]:
        """Peak(s) of the per-patch class logits per phrase: ``{phrase: [(uv, prob), ...]}``.

        This is the read-out of the fine-tuned point localiser (the checkpoint was trained with a
        Gaussian target around the demonstrated grasp pixel). Pixels are in the frame of the
        padded square image, i.e. of ``img_rgb`` when it is square.
        """
        from PIL import Image
        torch = self.torch
        im = Image.fromarray(img_rgb)
        side = max(im.size)
        with torch.no_grad():
            inp = self.proc(text=[[prefix + p] for p in phrases], images=[im] * len(phrases),
                            return_tensors="pt").to(self.device)
            lg = self.model(**inp).logits[:, :, 0].float().cpu().numpy()
        res = {}
        G = self.grid
        for ph, l in zip(phrases, lg):
            g = l.reshape(G, G).copy()
            peaks = []
            for _ in range(n_peaks):
                j = int(g.argmax())
                y, x = divmod(j, G)
                peaks.append(([float((x + 0.5) / G * side), float((y + 0.5) / G * side)],
                              float(1 / (1 + np.exp(-g[y, x])))))
                g[max(0, y - nms_patches):y + nms_patches + 1, max(0, x - nms_patches):x + nms_patches + 1] = -1e9
            res[ph] = peaks
        return res

    def locate(self, img_rgb: np.ndarray, phrases: Sequence[str], mode: str = "box") -> Dict[str, Optional[Dict]]:
        """One pixel per phrase, ``{phrase: {uv, score} | None}``, by either read-out."""
        if mode == "box":
            return self.boxes(img_rgb, phrases)
        pts = self.points(img_rgb, phrases)
        return {ph: dict(uv=p[0][0], score=p[0][1]) for ph, p in pts.items()}


__all__ = ["OWLV2_ID", "phrases_from_instruction", "OwlLocaliser"]
