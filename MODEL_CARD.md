# Model card: TruthLens

**Intended use:** triage support for human reviewers. Not a truth verdict, not for automated takedowns or legal evidence.

| Module | Method | Training data | Known limits |
|---|---|---|---|
| Image | EfficientNet-B0 fine-tuned, compression/blur/resize augmentation, temperature scaling | Subset of a public real-vs-fake faces dataset (fill in after training) | Faces only; weak on unseen generators; report cross-generator drop |
| Lip-sync | Mouth motion (MediaPipe or OpenCV Haar fallback) correlated with audio RMS envelope | None | Talking-head clips only; fails with occlusion, side profiles, music over speech |
| Audio | Frozen wav2vec2 embeddings + logistic regression | Public spoof subset (fill in) | English-heavy data; Hindi/Telugu performance must be measured separately |
| Fusion | Weighted logit average; Uncertain on any Real-vs-Fake disagreement | n/a | Weights are hand-set, not learned |
| Risk | Synthetic confidence + conflict + provenance + caption keywords | n/a | Heuristic weights; missing EXIF is common for innocent re-shares |

**Abstention:** probabilities in 0.35 to 0.65 are reported as Uncertain and routed to the review queue.
**Results:** fill in `reports/robustness.json`, lip-sync detection rate, coverage-vs-accuracy and files/min before submitting.
Numbers on small public subsets do not predict real-world performance. No detector is future-proof; use drift monitoring and the review loop.
