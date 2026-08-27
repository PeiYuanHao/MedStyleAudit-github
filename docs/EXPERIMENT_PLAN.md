# Experiment plan

1. **P0 feasibility:** metadata integrity, exact slide mapping, physical cluster
   recovery, lesion-alignment status, tissue-mask stability, descriptors,
   matching coverage, attrition, and balance.
2. **P1 core (implemented):** one-seed ResNet-50 path, configured ten-seed
   ResNet-50/DenseNet-121 execution, primary HCS/HCE, exact final-tensor ROI QA,
   source-buffer/seam robustness, directed pairs, and cross-run aggregation.
3. **P1b validation (partial):** context-randomized quota preparation and the
   planted-cue generator exist, but their training/audit pipelines are
   `not_implemented`. Lesion mapping is `unavailable` until exact WSI/XML assets
   and coordinate alignment validate. Ladder levels 5/6 are not implemented.
4. **P2 secondary:** full ladder, HED augmentation, GroupDRO,
   context-consistency mitigation, and exploratory HCS/OOD association.

The feasibility gate is deliberately not converted into an automatic PASS by
an arbitrary coverage cutoff. Review the saved coverage and balance tables and
lock the decision before any hospital-2 predictions are viewed.

No pair-weighted global HCS/HCE is available unless its complete configured
directed-pair set is present. Missing seeds/backbones are written to
`aggregate/<split>/missing_runs.csv` rather than dropped.
