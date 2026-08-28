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


def test_indexed_nearest_pool_matches_brute_force_definition():
    frame = _frame(); bank = CandidateBank.build(frame, ["f1", "f2"])
    matcher = BalancedTripletMatcher(bank, {"candidate_pool_size": 4})
    source = bank.frame.iloc[0]
    vector = source[["__z_f1", "__z_f2"]].to_numpy(float)
    excluded = {str(source["physical_id"])}
    brute = matcher._nearest(vector, bank.candidates(hospital=1, label=int(source.label), excluded_physical_ids=excluded, split="train"))
    indexed = matcher._nearest_indexed(vector, hospital=1, label_value=int(source.label), split="train", excluded_physical_ids=excluded)
    assert indexed["source_id"].tolist() == brute["source_id"].tolist()
    assert indexed["__distance"].tolist() == brute["__distance"].tolist()


def test_batched_queries_preserve_sequential_matching_results():
    frame = _frame(); bank = CandidateBank.build(frame, ["f1", "f2"])
    config = {"donors_per_source": 2, "candidate_pool_size": 8, "donor_reuse_cap": 99, "donor_slide_reuse_cap": 99, "tau_distance": 99, "tau_balance": 99}
    sources = bank.frame.iloc[:4]
    batched, _ = BalancedTripletMatcher(bank, config).match(sources, {"train": [0, 1]})
    sequential_matcher = BalancedTripletMatcher(bank, config); sequential = []
    for _, source in sources.iterrows():
        target = 1 if int(source.hospital_id) == 0 else 0
        selected, reason = sequential_matcher.match_source(source, target, "train")
        assert reason is None
        sequential.extend(selected)
    columns = ["source_id", "target_hospital", "within_donor", "cross_donor", "donor_index"]
    assert batched[columns].to_dict("records") == pd.DataFrame(sequential)[columns].to_dict("records")


def test_infeasible_global_reuse_capacity_fails_before_matching():
    frame = _frame(); bank = CandidateBank.build(frame, ["f1", "f2"])
    config = {"donors_per_source": 2, "donor_reuse_cap": 99, "donor_slide_reuse_cap": 1}
    matcher = BalancedTripletMatcher(bank, config)
    capacity = matcher.reuse_capacity_report(bank.frame, {"train": [0, 1]})
    assert not bool(capacity.loc[capacity["scope"] == "global", "feasible"].iloc[0])
    try:
        matcher.match(bank.frame, {"train": [0, 1]})
    except ValueError as error:
        assert "reuse caps are infeasible" in str(error)
    else:
        raise AssertionError("infeasible matching must stop before source iteration")


def test_active_index_removes_saturated_donors_without_changing_base_bank():
    frame = _frame(); bank = CandidateBank.build(frame, ["f1", "f2"])
    key_frame = bank.candidates(hospital=0, label=0, split="train", excluded_physical_ids=set())
    saturated = set(key_frame["source_id"].iloc[:2])
    bank.refresh_active(saturated, set())
    source = bank.frame.iloc[0]
    positions = bank.query_group(0, 0, "train", source[["__z_f1", "__z_f2"]].to_numpy(float), 10)
    assert saturated.isdisjoint(set(bank.frame.iloc[positions]["source_id"]))
    assert saturated.issubset(set(bank.candidates(hospital=0, label=0, split="train", excluded_physical_ids=set())["source_id"]))
