# Experiment plan

1. **P0 feasibility:** metadata integrity, exact slide mapping, physical cluster
   recovery, lesion-alignment status, tissue-mask stability, descriptors,
   matching coverage, attrition, and balance.
2. **P1 core:** one-seed ResNet-50 end-to-end smoke test, all ten ResNet-50 and
   DenseNet-121 seeds, primary HCS/HCE, ROI-only control, planted shortcut,
   source-buffer and seam robustness, and directed hospital pairs.
3. **P1b validation:** context-randomized training, periphery-only hospital
   classifier, full/ROI/periphery tumor diagnostics, and key ladder levels.
4. **P2 secondary:** full ladder, HED augmentation, GroupDRO,
   context-consistency mitigation, and exploratory HCS/OOD association.

The feasibility gate is deliberately not converted into an automatic PASS by
an arbitrary coverage cutoff. Review the saved coverage and balance tables and
lock the decision before any hospital-2 predictions are viewed.
