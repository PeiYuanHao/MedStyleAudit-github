import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist
from numba import njit
from medstyleaudit.matching.distance import euclidean_to

from medstyleaudit.matching.candidate_bank import CandidateBank
from medstyleaudit.matching.triplet_matcher import BalancedTripletMatcher
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.io import read_table


CONFIG = Path(
    "/root/autodl-tmp/h2_rescue_c010_pool128.FROZEN.yaml"
)
PRIOR = Path(
    "/root/autodl-tmp/h1_prior_triplets_only.parquet"
)

# 足够做速度和逐结果一致性验证，又不会跑太久
N = 200


print("=" * 76, flush=True)
print("ARRAY FAST MATCHER — EXACT EQUIVALENCE BENCHMARK", flush=True)
print("=" * 76, flush=True)


cfg = load_config(CONFIG)
settings = dict(cfg["matching"])

assert settings["candidate_pool_size"] == 128
assert settings["feature_calipers"] == {"compactness": 0.1}


print("Loading descriptors...", flush=True)

frame = read_table(
    cfg["data"]["descriptors"]
)

print("descriptor rows =", len(frame), flush=True)


print("Building CandidateBank...", flush=True)

bank = CandidateBank.build(
    frame,
    settings["descriptor_columns"],
)

print("CandidateBank = READY", flush=True)


source_rows = (
    bank.frame[
        bank.frame["split"] == "test"
    ]
    .sort_values(
        ["split", "hospital_id", "source_id"]
    )
    .head(N)
)


prior = read_table(PRIOR)

print("benchmark sources =", len(source_rows), flush=True)
print("prior triplets     =", len(prior), flush=True)


# ============================================================
# ORIGINAL MATCHER INITIALIZER
# ============================================================

def init_original():
    m = BalancedTripletMatcher(
        bank,
        settings,
    )

    for row in prior.itertuples(index=False):
        for donor, slide in (
            (row.within_donor, row.within_slide),
            (row.cross_donor, row.cross_slide),
        ):
            m.donor_reuse[donor] += 1
            m.donor_slide_reuse[slide] += 1

    donor_cap = int(
        settings["donor_reuse_cap"]
    )

    slide_cap = int(
        settings.get(
            "donor_slide_reuse_cap",
            2**31 - 1,
        )
    )

    m.saturated_donor_ids = {
        donor
        for donor, count in m.donor_reuse.items()
        if count >= donor_cap
    }

    m.saturated_slide_ids = {
        slide
        for slide, count in m.donor_slide_reuse.items()
        if count >= slide_cap
    }

    return m


# ============================================================
# FAST ARRAY MATCHER
#
# Important:
# - source ordering unchanged
# - target ordering unchanged
# - KD-tree queries unchanged
# - candidate pool unchanged
# - distance formula unchanged
# - pair cost unchanged
# - exact lexicographic tie-breaking unchanged
# - donor reuse semantics unchanged
# ============================================================


@njit(cache=True, nogil=True)
def greedy_top_k_pairs(
    valid,
    cost,
    within_rank,
    cross_rank,
    within_slide,
    cross_slide,
    slide_reuse,
    slide_cap,
    k_required,
    required_slide_diversity,
):
    """
    Exact ordered-greedy equivalent without globally sorting every pair.

    Original ordering:
        cost
        within source_id string
        cross source_id string

    At each step choose the smallest still-eligible pair.
    """

    nw, nc = valid.shape

    selected_w = np.full(k_required, -1, dtype=np.int64)
    selected_c = np.full(k_required, -1, dtype=np.int64)

    used_w = np.zeros(nw, dtype=np.uint8)
    used_c = np.zeros(nc, dtype=np.uint8)

    n_selected = 0

    while n_selected < k_required:

        best_w = -1
        best_c = -1
        best_cost = 0.0
        best_wr = 0
        best_cr = 0

        for i in range(nw):

            if used_w[i]:
                continue

            for j in range(nc):

                if used_c[j] or not valid[i, j]:
                    continue

                ws = within_slide[i]
                cs = cross_slide[j]

                # Preserve original diversity rule.
                if n_selected < required_slide_diversity:

                    seen_ws = False
                    seen_cs = False

                    for q in range(n_selected):

                        if within_slide[selected_w[q]] == ws:
                            seen_ws = True

                        if cross_slide[selected_c[q]] == cs:
                            seen_cs = True

                    if seen_ws or seen_cs:
                        continue

                # Preserve within-source slide-cap accounting.
                w_added = 0
                c_added = 0

                for q in range(n_selected):

                    if within_slide[selected_w[q]] == ws:
                        w_added += 1

                    if cross_slide[selected_c[q]] == cs:
                        c_added += 1

                if slide_reuse[ws] + w_added >= slide_cap:
                    continue

                if slide_reuse[cs] + c_added >= slide_cap:
                    continue

                cc = cost[i, j]
                wr = within_rank[i]
                cr = cross_rank[j]

                if best_w < 0:
                    best_w = i
                    best_c = j
                    best_cost = cc
                    best_wr = wr
                    best_cr = cr

                elif cc < best_cost:
                    best_w = i
                    best_c = j
                    best_cost = cc
                    best_wr = wr
                    best_cr = cr

                elif cc == best_cost:

                    if (
                        wr < best_wr
                        or (
                            wr == best_wr
                            and cr < best_cr
                        )
                    ):
                        best_w = i
                        best_c = j
                        best_cost = cc
                        best_wr = wr
                        best_cr = cr

        if best_w < 0:
            break

        selected_w[n_selected] = best_w
        selected_c[n_selected] = best_c

        used_w[best_w] = 1
        used_c[best_c] = 1

        n_selected += 1

    return (
        selected_w[:n_selected],
        selected_c[:n_selected],
    )


class ArrayFastMatcher(BalancedTripletMatcher):

    def __post_init_arrays(self):

        f = self.bank.frame

        self._sid = f["source_id"].to_numpy(copy=False)
        self._sid_str = (
            f["source_id"]
            .astype(str)
            .to_numpy()
        )

        # Exact lexical rank corresponding to original astype(str) tie-break.
        unique_ids = np.unique(self._sid_str)

        lexical = {
            value: rank
            for rank, value in enumerate(unique_ids)
        }

        self._sid_rank = np.fromiter(
            (
                lexical[value]
                for value in self._sid_str
            ),
            dtype=np.int64,
            count=len(self._sid_str),
        )

        self._physical = (
            f["physical_id"]
            .astype(str)
            .to_numpy()
        )

        self._slide = f["slide_id"].to_numpy(copy=False)

        self._hospital = (
            f["hospital_id"]
            .to_numpy(copy=False)
        )

        self._label = (
            f["label"]
            .to_numpy(copy=False)
        )

        self._split = (
            f["split"]
            .astype(str)
            .to_numpy()
        )

        self._z = self.bank._standardized

        # feature arrays used by calipers
        self._feature = {}

        for name in self.config.get(
            "feature_calipers", {}
        ):
            self._feature[name] = (
                f[name]
                .to_numpy(dtype=np.float64)
            )

        # source_id -> global CandidateBank row
        self._sid_to_pos = {
            sid: i
            for i, sid in enumerate(self._sid)
        }

        # slide ids encoded once
        slide_codes, slide_values = pd.factorize(
            f["slide_id"],
            sort=False,
        )

        self._slide_code = slide_codes.astype(
            np.int64,
            copy=False,
        )

        self._slide_values = slide_values.to_numpy()

        self._slide_to_code = {
            slide: i
            for i, slide in enumerate(
                self._slide_values
            )
        }

        self._donor_reuse_array = np.zeros(
            len(f),
            dtype=np.int32,
        )

        self._slide_reuse_array = np.zeros(
            len(self._slide_values),
            dtype=np.int32,
        )


    def prepare_arrays(self):
        self.__post_init_arrays()


    def prime_prior_arrays(self):

        # counters already populated by caller;
        # convert exact current counter state to arrays
        for donor, count in self.donor_reuse.items():
            pos = self._sid_to_pos.get(donor)
            if pos is not None:
                self._donor_reuse_array[pos] = count

        for slide, count in self.donor_slide_reuse.items():
            code = self._slide_to_code.get(slide)
            if code is not None:
                self._slide_reuse_array[code] = count


    def _eligible_positions(
        self,
        positions,
        source_vector,
        source_id,
        source_physical,
    ):
        """
        Pure NumPy equivalent of original eligible().
        """

        positions = np.asarray(
            positions,
            dtype=np.int64,
        )

        if positions.size == 0:
            return (
                positions,
                np.empty(0, dtype=np.float64),
            )

        slide_codes = self._slide_code[positions]

        mask = (
            self._physical[positions]
            != source_physical
        )

        mask &= (
            self._sid[positions]
            != source_id
        )

        donor_cap = int(
            self.config.get(
                "donor_reuse_cap",
                2**31 - 1,
            )
        )

        slide_cap = int(
            self.config.get(
                "donor_slide_reuse_cap",
                2**31 - 1,
            )
        )

        mask &= (
            self._donor_reuse_array[positions]
            < donor_cap
        )

        mask &= (
            self._slide_reuse_array[slide_codes]
            < slide_cap
        )

        positions = positions[mask]

        if positions.size == 0:
            return (
                positions,
                np.empty(0, dtype=np.float64),
            )

        candidate_z = np.asfortranarray(
            self._z[positions]
        )

        distances = euclidean_to(
            source_vector,
            candidate_z,
        )

        # Exact original sorting:
        # distance primary, source_id secondary.
        order = np.lexsort(
            (
                self._sid_str[positions],
                distances,
            )
        )

        return (
            positions[order],
            distances[order],
        )


    def _nearest_positions(
        self,
        source_vector,
        *,
        hospital,
        label_value,
        split,
        source_id,
        source_physical,
        initial_positions=None,
    ):

        total = self.bank.group_size(
            hospital,
            label_value,
            split,
        )

        if total == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )

        pool = int(
            self.config.get(
                "candidate_pool_size",
                64,
            )
        )

        if initial_positions is None:
            probe = min(
                total,
                max(pool * 2, 128),
            )
        else:
            probe = len(initial_positions)

        first = True

        while True:

            if (
                first
                and initial_positions is not None
            ):
                positions = np.asarray(
                    initial_positions,
                    dtype=np.int64,
                )
            else:
                positions = self.bank.query_group(
                    hospital,
                    label_value,
                    split,
                    source_vector,
                    probe,
                )

            first = False

            pos, dist = self._eligible_positions(
                positions,
                source_vector,
                source_id,
                source_physical,
            )

            if len(pos) >= pool:

                cutoff = float(
                    dist[pool - 1]
                )

                if probe < total:

                    tolerance = max(
                        1e-12,
                        abs(cutoff) * 1e-12,
                    )

                    boundary = float(
                        dist.max()
                    )

                    if (
                        cutoff + tolerance
                        >= boundary
                    ):

                        tied = (
                            self.bank.query_group_radius(
                                hospital,
                                label_value,
                                split,
                                source_vector,
                                cutoff + tolerance,
                            )
                        )

                        pos, dist = (
                            self._eligible_positions(
                                tied,
                                source_vector,
                                source_id,
                                source_physical,
                            )
                        )

                return (
                    pos[:pool],
                    dist[:pool],
                )

            if probe >= total:
                return (
                    pos[:pool],
                    dist[:pool],
                )

            probe = min(
                total,
                probe * 2,
            )


    def match_source(
        self,
        source,
        target_hospital,
        donor_split=None,
        candidate_prefetch=None,
    ):

        settings = self.config

        k_required = int(
            settings.get(
                "donors_per_source",
                1,
            )
        )

        source_id = source["source_id"]

        source_pos = self._sid_to_pos[
            source_id
        ]

        source_vector = self._z[
            source_pos
        ]

        source_physical = str(
            source["physical_id"]
        )

        label_value = int(
            source["label"]
        )

        source_hospital = int(
            source["hospital_id"]
        )

        source_split = str(
            source["split"]
        )

        prefetch = (
            candidate_prefetch or {}
        )

        within_key = (
            source_id,
            source_hospital,
            label_value,
            source_split,
        )

        cross_key = (
            source_id,
            int(target_hospital),
            label_value,
            str(donor_split),
        )

        within_pos, within_dist = (
            self._nearest_positions(
                source_vector,
                hospital=source_hospital,
                label_value=label_value,
                split=source_split,
                source_id=source_id,
                source_physical=source_physical,
                initial_positions=prefetch.get(
                    within_key
                ),
            )
        )

        cross_pos, cross_dist = (
            self._nearest_positions(
                source_vector,
                hospital=int(target_hospital),
                label_value=label_value,
                split=str(donor_split),
                source_id=source_id,
                source_physical=source_physical,
                initial_positions=prefetch.get(
                    cross_key
                ),
            )
        )

        if within_pos.size == 0:
            return [], "no_within_candidate"

        if cross_pos.size == 0:
            return [], "no_cross_candidate"

        tau_d = float(
            settings.get(
                "tau_distance",
                np.inf,
            )
        )

        wm = within_dist <= tau_d
        cm = cross_dist <= tau_d

        within_pos = within_pos[wm]
        within_dist = within_dist[wm]

        cross_pos = cross_pos[cm]
        cross_dist = cross_dist[cm]

        if (
            within_pos.size == 0
            or cross_pos.size == 0
        ):
            return [], "distance_threshold"

        required_slide_diversity = int(
            settings.get(
                "min_donor_slide_diversity",
                1,
            )
        )

        if (
            required_slide_diversity < 1
            or required_slide_diversity
            > k_required
        ):
            raise ValueError(
                "min_donor_slide_diversity "
                "must be between 1 and "
                "donors_per_source"
            )

        if (
            len(
                pd.unique(
                    self._slide[
                        within_pos
                    ]
                )
            )
            < required_slide_diversity
            or
            len(
                pd.unique(
                    self._slide[
                        cross_pos
                    ]
                )
            )
            < required_slide_diversity
        ):
            return (
                [],
                "insufficient_donor_slide_diversity",
            )

        # ====================================================
        # Vectorized pair construction
        # ====================================================

        imbalance = np.abs(
            within_dist[:, None]
            - cross_dist[None, :]
        )

        valid = (
            imbalance
            <= float(
                settings.get(
                    "tau_balance",
                    np.inf,
                )
            )
        )

        valid &= (
            self._physical[
                within_pos
            ][:, None]
            !=
            self._physical[
                cross_pos
            ][None, :]
        )

        for name, limit in settings.get(
            "feature_calipers", {}
        ).items():

            wv = self._feature[
                name
            ][within_pos]

            cv = self._feature[
                name
            ][cross_pos]

            valid &= (
                np.abs(
                    wv[:, None]
                    - cv[None, :]
                )
                <= float(limit)
            )

        if not valid.any():
            return (
                [],
                "insufficient_balanced_pairs",
            )

        within_z = np.asfortranarray(
            self._z[within_pos]
        )

        cross_z = np.asfortranarray(
            self._z[cross_pos]
        )

        pair_distance = cdist(
            within_z,
            cross_z,
            metric="euclidean",
        )

        cost = (
            within_dist[:, None]
            + cross_dist[None, :]
            + float(
                settings.get(
                    "lambda_balance",
                    0.0,
                )
            ) * imbalance
            + float(
                settings.get(
                    "lambda_pair",
                    0.0,
                )
            ) * pair_distance
        )

        within_rank = self._sid_rank[
            within_pos
        ]

        cross_rank = self._sid_rank[
            cross_pos
        ]

        within_slide_codes = self._slide_code[
            within_pos
        ].astype(np.int64, copy=False)

        cross_slide_codes = self._slide_code[
            cross_pos
        ].astype(np.int64, copy=False)

        slide_cap = int(
            settings.get(
                "donor_slide_reuse_cap",
                2**31 - 1,
            )
        )

        selected_w, selected_c = greedy_top_k_pairs(
            valid,
            cost,
            within_rank,
            cross_rank,
            within_slide_codes,
            cross_slide_codes,
            self._slide_reuse_array,
            slide_cap,
            k_required,
            required_slide_diversity,
        )

        if len(selected_w) != k_required:
            return (
                [],
                "insufficient_balanced_pairs",
            )

        selected = []

        for donor_index in range(k_required):

            wlocal = int(
                selected_w[donor_index]
            )

            clocal = int(
                selected_c[donor_index]
            )

            wpos = int(
                within_pos[wlocal]
            )

            cpos = int(
                cross_pos[clocal]
            )

            selected.append({
                "source_id":
                    source["source_id"],

                "source_split":
                    source["split"],

                "source_hospital":
                    int(source["hospital_id"]),

                "target_hospital":
                    int(target_hospital),

                "source_slide":
                    source["slide_id"],

                "source_patient":
                    source.get(
                        "patient_id",
                        pd.NA,
                    ),

                "source_physical_id":
                    source["physical_id"],

                "label":
                    int(source["label"]),

                "within_donor":
                    self._sid[wpos],

                "within_slide":
                    self._slide[wpos],

                "within_physical_id":
                    self._physical[wpos],

                "cross_donor":
                    self._sid[cpos],

                "cross_slide":
                    self._slide[cpos],

                "cross_physical_id":
                    self._physical[cpos],

                "matching_distance_within":
                    float(within_dist[wlocal]),

                "matching_distance_cross":
                    float(cross_dist[clocal]),

                "matching_distance_pair":
                    float(
                        pair_distance[
                            wlocal,
                            clocal,
                        ]
                    ),

                "matching_cost":
                    float(
                        cost[
                            wlocal,
                            clocal,
                        ]
                    ),

                "donor_index":
                    donor_index,
            })

        if len(selected) != k_required:
            return (
                [],
                "insufficient_balanced_pairs",
            )

        # ====================================================
        # Exact global reuse commit
        # ====================================================

        donor_cap = int(
            settings.get(
                "donor_reuse_cap",
                2**31 - 1,
            )
        )

        for record in selected:

            for donor, slide in (
                (
                    record["within_donor"],
                    record["within_slide"],
                ),
                (
                    record["cross_donor"],
                    record["cross_slide"],
                ),
            ):

                pos = self._sid_to_pos[
                    donor
                ]

                scode = self._slide_to_code[
                    slide
                ]

                self.donor_reuse[
                    donor
                ] += 1

                self.donor_slide_reuse[
                    slide
                ] += 1

                self._donor_reuse_array[
                    pos
                ] += 1

                self._slide_reuse_array[
                    scode
                ] += 1

                if (
                    self.donor_reuse[
                        donor
                    ]
                    >= donor_cap
                ):
                    self.saturated_donor_ids.add(
                        donor
                    )

                if (
                    self.donor_slide_reuse[
                        slide
                    ]
                    >= slide_cap
                ):
                    self.saturated_slide_ids.add(
                        slide
                    )

        return selected, None


# ============================================================
# FAST MATCHER INITIALIZER
# ============================================================

def init_fast():

    m = ArrayFastMatcher(
        bank,
        settings,
    )

    m.prepare_arrays()

    for row in prior.itertuples(index=False):

        for donor, slide in (
            (
                row.within_donor,
                row.within_slide,
            ),
            (
                row.cross_donor,
                row.cross_slide,
            ),
        ):

            m.donor_reuse[
                donor
            ] += 1

            m.donor_slide_reuse[
                slide
            ] += 1

    donor_cap = int(
        settings["donor_reuse_cap"]
    )

    slide_cap = int(
        settings.get(
            "donor_slide_reuse_cap",
            2**31 - 1,
        )
    )

    m.saturated_donor_ids = {
        donor
        for donor, count
        in m.donor_reuse.items()
        if count >= donor_cap
    }

    m.saturated_slide_ids = {
        slide
        for slide, count
        in m.donor_slide_reuse.items()
        if count >= slide_cap
    }

    m.prime_prior_arrays()

    return m


targets = settings[
    "target_hospitals"
]


# ============================================================
# ORIGINAL
# ============================================================

print()
print("=" * 76, flush=True)
print("1/2 ORIGINAL MATCHER", flush=True)
print("=" * 76, flush=True)

bank.refresh_active(
    set(),
    set(),
)

m1 = init_original()

t0 = time.perf_counter()

trip1, ledger1 = m1.match(
    source_rows,
    targets,
    show_progress=True,
)

original_seconds = (
    time.perf_counter() - t0
)

original_sps = (
    len(source_rows)
    / original_seconds
)

print(
    f"ORIGINAL: "
    f"{original_seconds:.3f}s | "
    f"{original_sps:.3f} source/s",
    flush=True,
)


# ============================================================
# FAST ARRAY
# ============================================================

print()
print("=" * 76, flush=True)
print("2/2 ARRAY FAST MATCHER", flush=True)
print("=" * 76, flush=True)

bank.refresh_active(
    set(),
    set(),
)

m2 = init_fast()

t0 = time.perf_counter()

trip2, ledger2 = m2.match(
    source_rows,
    targets,
    show_progress=True,
)

fast_seconds = (
    time.perf_counter() - t0
)

fast_sps = (
    len(source_rows)
    / fast_seconds
)

print(
    f"ARRAY FAST: "
    f"{fast_seconds:.3f}s | "
    f"{fast_sps:.3f} source/s",
    flush=True,
)


# ============================================================
# EXACT EQUIVALENCE
# ============================================================

print()
print("=" * 76, flush=True)
print("EXACT EQUIVALENCE CHECK", flush=True)
print("=" * 76, flush=True)

strict_exact = True
structural_exact = True
numeric_close = True
error = None

identity_columns = [
    "triplet_id",
    "source_id",
    "source_split",
    "source_hospital",
    "target_hospital",
    "source_slide",
    "source_patient",
    "source_physical_id",
    "label",
    "within_donor",
    "within_slide",
    "within_physical_id",
    "cross_donor",
    "cross_slide",
    "cross_physical_id",
    "donor_index",
]

numeric_columns = [
    "matching_distance_within",
    "matching_distance_cross",
    "matching_distance_pair",
    "matching_cost",
]

try:

    pd.testing.assert_frame_equal(
        ledger1.reset_index(drop=True),
        ledger2.reset_index(drop=True),
        check_exact=True,
    )

    pd.testing.assert_frame_equal(
        trip1[identity_columns].reset_index(drop=True),
        trip2[identity_columns].reset_index(drop=True),
        check_exact=True,
    )

    assert (
        Counter(m1.donor_reuse)
        ==
        Counter(m2.donor_reuse)
    )

    assert (
        Counter(m1.donor_slide_reuse)
        ==
        Counter(m2.donor_slide_reuse)
    )

except Exception as exc:

    structural_exact = False
    error = "STRUCTURAL: " + repr(exc)


max_abs_diff = 0.0

if structural_exact:

    for column in numeric_columns:

        a = trip1[column].to_numpy(dtype=np.float64)
        b = trip2[column].to_numpy(dtype=np.float64)

        if len(a):

            diff = float(
                np.max(
                    np.abs(a - b)
                )
            )

            max_abs_diff = max(
                max_abs_diff,
                diff,
            )

            if not np.allclose(
                a,
                b,
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            ):
                numeric_close = False


try:

    pd.testing.assert_frame_equal(
        trip1.reset_index(drop=True),
        trip2.reset_index(drop=True),
        check_exact=True,
    )

except Exception:

    strict_exact = False


exact = (
    structural_exact
    and numeric_close
)

speedup = (
    original_seconds
    / fast_seconds
)

projected_minutes = (
    85054
    / fast_sps
    / 60
)

print()
print("=" * 76)
print("ARRAY FAST BENCHMARK RESULT")
print("=" * 76)

print(
    f"sources                  = "
    f"{len(source_rows)}"
)

print(
    f"original_seconds         = "
    f"{original_seconds:.3f}"
)

print(
    f"fast_seconds             = "
    f"{fast_seconds:.3f}"
)

print(
    f"original_source_per_sec  = "
    f"{original_sps:.3f}"
)

print(
    f"fast_source_per_sec      = "
    f"{fast_sps:.3f}"
)

print(
    f"speedup                  = "
    f"{speedup:.3f}x"
)

print(
    f"projected_matching_min   = "
    f"{projected_minutes:.2f}"
)

print(
    "STRUCTURAL_EXACT         =",
    "PASS" if structural_exact else "FAIL",
)

print(
    "NUMERIC_1E-12            =",
    "PASS" if numeric_close else "FAIL",
)

print(
    "STRICT_BITWISE           =",
    "PASS" if strict_exact else "FAIL",
)

print(
    f"MAX_NUMERIC_ABS_DIFF      = "
    f"{max_abs_diff:.3e}"
)

print(
    "EXACT_EQUIVALENCE        =",
    "PASS" if exact else "FAIL",
)

if error:
    print(
        "EQUIVALENCE_ERROR       =",
        error,
    )

if exact and fast_sps >= 30:
    decision = "EXCELLENT_USE_ARRAY_FAST"

elif exact and fast_sps >= 23.63:
    decision = "UNDER_1H_USE_ARRAY_FAST"

elif exact and fast_sps >= 18:
    decision = "FAST_BUT_OVER_1H"

elif exact:
    decision = "NOT_FAST_ENOUGH"

else:
    decision = "DO_NOT_USE"

print(
    "DECISION                 =",
    decision,
)

print("=" * 76)
