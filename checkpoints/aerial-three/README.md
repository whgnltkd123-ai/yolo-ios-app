# Balanced flight YOLO26s

`best.pt` is the fixed final (LAST) checkpoint from balanced-flight recovery run 364227993, retained under the existing build input name. SHA256: c7df126a039c1edde8d6b3e0c78be99f43e6770913eb1ff201976e2cf2505d26.

Public 270: confirmed drone TP63 FP5/98, bird TP63 FP36/102, airplane TP38 FP0/90. Background71 includes one confirmed drone false positive. Full accuracy gate FAILED. User requested application with disclosed limitations. Public160-frame replacement video evaluation completed; private292 unavailable and not evaluated.

Native display: suspicion >=0.30, confirmed >=0.60, same-anchor margin strictly >0.20; another aerial class >=0.30 or invalid/missing drone/bird scores => unknown. Scores are not calibrated probabilities. No tracking. Below detector threshold produces no fabricated box.

See aerial-validation.json. 300m, iPhone15 FPS and device accuracy remain unverified. CoreML conversion must verify checkpoint hash and 3-class same-anchor output parity before packaging.
