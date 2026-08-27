# Data pipeline and schemas

`descriptors.csv` contains canonical metadata (`source_id`, `split`, hospital,
slide, patient/slide physical identity, label) plus mask geometry. Standardizing
means/scales are fitted only on training rows and saved separately.

`triplets.csv` has one row per donor pair and retains source/within/cross
physical identities, distances, cost, and donor multiplicity index. Removing a
construction-invalid pair removes both arms. `source_ledger.csv` retains every
candidate directed comparison and its exclusion reason.

`predictions.csv` stores original, within, and cross logits on the same row.
`source_metrics.csv` collapses the K rows only after normalization by ID-val
original-logit median and IQR. Directed and global HCS/HCE tables are derived
from that record, never from rounded paper values.
