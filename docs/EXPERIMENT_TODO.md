# Final execution status

The required protocol and execution paths are frozen and reduced to the final
paper scope: ResNet-50 only, seeds `11/42/101`, Hospital 1 (OOD validation) and
Hospital 2 (final OOD test). Remaining work is running the reduced suite on the
server, not adding methodology.

Lesion-aware analysis runs only after validated WSI/XML alignment; otherwise its
status is `unavailable`. Removed experiments (DenseNet-121, ten-seed runs, planted
shortcut, context-randomized training, the identification ladder, GroupDRO, HED
augmentation, context-consistency mitigation, and appearance/second-dataset work)
are legacy/optional and do not appear in the final runner.
