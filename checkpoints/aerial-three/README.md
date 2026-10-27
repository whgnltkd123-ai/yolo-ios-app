# Three-class build checkpoint

This is the latest completed 15-epoch reinforcement from the original app checkpoint, Kaggle run362771654. Class order is drone, bird, airplane. It does not include the separate 50-epoch MBAOD job still running when the user requested this installation.

- Bundled SHA256: `f1701e53e9952b9cfc3726fb118d3f718de2908973a5021dff90731db1b23b0e`.
- Original parent SHA256: `024b3ba7a552a9b52a88cb6de05cf6c2927b8db242ae2d792c215d602e450371`.
- Sources: 15,206 frozen Seraphim/COCO/MIT replay images, 646 reviewed LAT RGB bird frames and 936 USC MCL RGB drone frames; 16,788 unique images. Weighted repeats are not additional unique images.
- Original and earlier checkpoints remain in Git history and verified workspace backups. No unrelated weights were merged. Personal videos were not used for training.

The preserved legacy diagnostic failed the planned accuracy gates: drone10TP/33FP/98GT at.70; bird0TP/0FP/102GT at.60; airplane21TP/9FP/90GT at.60. Background71 and private292 had no high-score drone detections. The user requested installing the latest completed model with the new decision policy. No accuracy improvement is asserted.

The app uses suspicion40, confirmation60 for both bird and drone, and a strictly greater than20 percentage-point difference between the same detection's two class scores. Equal20 or missing scores remains unidentified. Consecutive observations are required before confirmation. The legacy diagnostic is preserved separately and does not measure this new app policy. Model scores are not calibrated probabilities.

The build verifies checkpoint SHA and class order, exports Core ML at640 with same-anchor class scores, compares numeric output on three deterministic public frames, builds the native app and runs simulator tracking/decision checks. This is not physical-device or accuracy validation. The three conversion inputs are full public CC0 frames, not GT crops or personal videos.

LAT material retains CC BY-NC-SA4.0 terms. This packaging does not establish permission for commercial distribution. Static-frame training and the experimental motion veto do not demonstrate temporal wingbeat recognition or independent field performance.
