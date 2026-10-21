# Three-class build checkpoint

`best.pt` is the user's YOLO26n checkpoint after ten epochs on the
Seraphim/COCO-derived dataset. Class order: drone, bird, airplane.
SHA256: `024b3ba7a552a9b52a88cb6de05cf6c2927b8db242ae2d792c215d602e450371`.
The accompanying validation report belongs to this exact checkpoint.

The existing validation set gives aggregate mAP50 0.757838 and mAP50–95
0.548122. Per-class mAP50–95: drone 0.690234, bird 0.276672,
airplane 0.677461. These are detection metrics, not probabilities of a
correct real-world identification. Seraphim's test split was used for
validation; no independent field test was performed.

An additional five-epoch public-video fine-tune reduced all three
per-class scores on this validation set, so it is not bundled.
Video frames train appearance, not temporal wing motion.

The manual `yolo26-drone-alert` Codemagic workflow verifies this file's
hash and class order before exporting Core ML. Core ML conversion,
quantization effects, iPhone camera performance and unsigned IPA build
are not yet verified.
