# Original-author ByteTrack integration
Base: build24 branch fix/small-target-continuity, b8b9993.
Detector: unchanged aerial3 YOLO26s c7df126a039c1edde8d6b3e0c78be99f43e6770913eb1ff201976e2cf2505d26 in all modes.
Engine source: https://github.com/FoundationVision/ByteTrack/tree/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/deploy/ncnn/cpp
This is the original authors' official C++ implementation, NOT Ultralytics' Python implementation, NOT a Swift reimplementation. Upstream MIT license is preserved under Sources/OfficialByteTrack/upstream/LICENSE.

Local changes to upstream:
- STrack.h uses OpenCV core/types.hpp instead of the all-modules umbrella; adds standard C++ headers.
- STrack carries detection_index; BYTETracker initializes it from Object.label and update/re_activate propagate it. It does not participate in assignment, costs, scores or filtering.
- All Kalman, assignment, lifecycle and original .5 association/.6 birth/.8 match thresholds unchanged.
The app's 30/60 classification policy is separate: a new .3 detector proposal cannot initialize an upstream ID. Weak observations may rescue an established ID. This behavior must not be described as fixed long-range recall.
- Official code does not guarantee ID retention for nonoverlapping tiny detections or ambiguous crossings. Previous custom-heuristic assertions were replaced with tests for documented upstream operation, not a claim those field failures are solved.

The C ABI bridge forwards at most 50 bounding boxes, with no camera frame copy or secondary neural inference. Source indices eliminate a second approximate class-to-track matching pass.
GMC remains the existing bounded translation estimator. Input coordinates subtract cumulative translation before official tracking; output adds it back. This is an app coordinate adapter, not official BoT-SORT GMC.
Mode1 leaves the tracker unallocated. VideoCapture, model loader, preview, decoder, ANE configuration and alarm policy are unchanged.
Missing observations are shown briefly as last observed position, marked '관측 끊김', score0 and no detectionIndex. They never count as observed detection or renew an alarm.

Pinned Eigen3 and OpenCV header dependencies are prepared before build; only needed inline OpenCV geometry is used, no OpenCV image/inference runtime. Build must prove this does not require unresolved runtime symbols.
Validation pending: native C++ smoke, iOS Simulator decisions/IDs, device build. iPhone15 FPS/p95 latency/thermal/field IDs and camera motion performance are NOT measured.
Deployment gate: same footage baseline vs new in modes1/2/3, report actual camera/AI FPS, p95 latency, ID switches and false merges. Do not assert unchanged speed from a successful build.

Platform exception: the upstream rectangular-cost error branch used desktop system(pause)/exit. It now throws invalid_argument; the C ABI catches exceptions and Swift resets the failed engine. Normal assignment is unchanged. Unassigned detector candidates >=.30 remain visible without an ID; they are not counted as tracker births. Native macOS smoke passed at 5d84894; iOS verification remains pending.
