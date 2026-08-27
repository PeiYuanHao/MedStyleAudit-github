import pandas as pd

from medstyleaudit.matching.candidate_bank import CandidateBank
from medstyleaudit.matching.triplet_matcher import BalancedTripletMatcher


def _frame():
    rows = []
    index = 0
    for hospital in [0, 1]:
        for label in [0, 1]:
            for offset in range(5):
                rows.append({"source_id": index, "split": "train", "hospital_id": hospital, "label": label, "slide_id": f"s{hospital}-{offset}", "patient_id": None, "physical_id": f"slide:s{hospital}-{offset}", "f1": float(offset), "f2": float(label)})
                index += 1
    return pd.DataFrame(rows)


def test_balanced_match_enforces_constraints_and_is_deterministic():
    frame = _frame(); config = {"donors_per_source": 2, "lambda_balance": 1, "lambda_pair": .1, "tau_distance": 10, "tau_balance": 2, "candidate_pool_size": 10, "donor_reuse_cap": 99}
    bank = CandidateBank.build(frame, ["f1", "f2"])
    source = bank.frame.iloc[0]
    first, reason = BalancedTripletMatcher(bank, config).match_source(source, 1, "train")
    second, _ = BalancedTripletMatcher(bank, config).match_source(source, 1, "train")
    assert reason is None and first == second and len(first) == 2
    assert all(row["source_physical_id"] not in {row["within_physical_id"], row["cross_physical_id"]} for row in first)
    assert all(frame.set_index("source_id").loc[row["within_donor"], "label"] == source.label for row in first)


def test_match_fails_as_a_pair_when_k_unavailable():
    frame = _frame(); bank = CandidateBank.build(frame, ["f1", "f2"])
    selected, reason = BalancedTripletMatcher(bank, {"donors_per_source": 99}).match_source(bank.frame.iloc[0], 1, "train")
    assert selected == [] and reason == "insufficient_balanced_pairs"
