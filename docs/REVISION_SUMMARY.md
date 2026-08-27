# Implementation summary

The primary pipeline is config-driven from metadata integrity through
descriptors, balanced matching, coverage, exact counterfactual construction,
ERM training, HCS/HCE, robustness, and multi-seed/multi-backbone aggregation.
Pair-weighted summaries now require the complete prespecified directed-pair set,
and ROI identity is checked on the normalized classifier input tensor.

Phase 0 now always writes `patch_mapping.csv`; missing or ambiguous WSI/XML,
coordinates, orientation, or pyramid metadata remain explicit. Controls and P2
placeholders report `partial`, `unavailable`, or `not_implemented` rather than
appearing successful. The context-randomized and planted-shortcut pipelines are
not implemented end-to-end, and mitigation, GroupDRO, HED, and ladder levels
5/6 remain secondary work.

The repository does not include datasets, model weights, or empirical results.
Large-dataset network downloads are disabled in code.
