---
pretty_name: MedStyleAudit Experiments
license: other
task_categories:
  - image-classification
---

# MedStyleAudit final experiment artifacts

This private dataset repository stores derived artifacts for **Auditing
Hospital-Associated Context Sensitivity with a Pixel-Identical Label-Defining
Region**. It contains protocol records, descriptors, matched-triplet ledgers,
random-initialized model checkpoints, predictions, audit records, aggregate
tables, figures, and execution logs.

It does **not** redistribute raw Camelyon17-WILDS patches, original CAMELYON17
WSIs, XML annotations, or unchanged source dataset shards. Obtain the original
data from its official source under its applicable terms. Reproduction code is
maintained at <https://github.com/PeiYuanHao/MedStyleAudit-github>.

HCS summarizes excess absolute normalized-logit instability under matched
cross-hospital versus within-hospital peripheral-context transplantation. HCE
summarizes the corresponding signed cross-minus-within shift. Both are
sensitivity contrasts, not causal identification of scanner, stain, hospital,
or shortcut mechanisms.

The immutable resolved protocol and its SHA256 are under `protocol/`.
`MANIFEST.json` records every artifact path, checksum, byte size, experiment,
backbone, seed, split, Git commit, and protocol hash. No experimental conclusion
is claimed by this card before the final suite completes.

Download and checksum-verify all artifacts with:

```bash
export MEDSTYLE_HF_REPO="PeiyuanHao/MedStyleAudit-Experiments"
# HF_TOKEN must already be present in the environment.
python scripts/hf_download_artifacts.py /path/to/artifacts
```
