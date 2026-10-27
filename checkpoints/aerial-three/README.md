# Three-class build checkpoint

This is the existing app model after 50 epochs of additional training in Kaggle run362665966. Class order is drone, bird, airplane. It is not a fresh COCO model or a merge of unrelated weights.

- Bundled SHA256: `96c48c83c0ac6210187d4cfadf05dfb0ef337de91e8a368cbde81f82bbb07b03`.
- Original parent SHA256: `024b3ba7a552a9b52a88cb6de05cf6c2927b8db242ae2d792c215d602e450371`.
- Frozen sources: Seraphim, COCO and MIT flight frames. LAT and personal videos were not used for training.
- Local rollback: the original checkpoint remains in Git parent9852a2a and the verified workspace backup.

The fixed final diagnostic did not meet the planned accuracy gates: drone9TP/28FP/98GT at.70; bird0TP/0FP/102GT at.60; airplane13TP/10FP/90GT at.60. Background71 had no high-score detections. The user explicitly requested applying this reinforcement after receiving these results. This does not establish a performance improvement.

The build checks checkpoint SHA and class order, exports Core ML at640, compares numeric output on three deterministic public frames, builds the native app and runs the existing simulator tracker checks. The conversion report is a build artifact. It is not an accuracy or physical-device test. Runtime thresholds remain unchanged.

The fixed conversion inputs under tools/coreml-check-inputs are the lexicographically first public validation image for each annotated class from the existing CC0 video dataset. They are full frames, not GT crops, and contain no personal videos. The original final evaluation is retained; these inputs only check conversion.
