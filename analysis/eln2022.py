"""ELN 2022 risk (Döhner et al., Blood 2022), approximated from Beat AML calls.

    python -m analysis.eln2022        # prints the 2017 -> 2022 shift table

Inputs: consensus fusions, karyotype string, NPM1 / FLT3-ITD (clinical), CEBPA
biallelic (proxy for in-frame bZIP), and WES calls for TP53 and the
myelodysplasia-related (MR) genes.
Order of rules:
  1. Favorable fusions: RUNX1::RUNX1T1, CBFB::MYH11              -> Favorable
  2. Adverse cytogenetics (fusions or karyotype, see below)      -> Adverse
  3. TP53 mutation                                               -> Adverse
  4. NPM1 mutation:  without FLT3-ITD -> Favorable, with ITD -> Intermediate
     (MR mutations do not override NPM1 in ELN 2022)
  5. MR gene mutation (ASXL1 BCOR EZH2 RUNX1 SF3B1 SRSF2 STAG2 U2AF1 ZRSR2) -> Adverse
  6. CEBPA biallelic                                             -> Favorable
  7. everything else (incl. NPM1-wt FLT3-ITD at any allelic ratio) -> Intermediate
APL (PML::RARA) is outside ELN and keeps NaN. Patients with no karyotype, no fusion
call and no ELN 2017 call are NaN. Limitations: karyotype parsing is approximate;
patients without WES are treated as MR/TP53 negative (flagged in `wes_missing`).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

MR_GENES = ["ASXL1", "BCOR", "EZH2", "RUNX1", "SF3B1", "SRSF2", "STAG2", "U2AF1", "ZRSR2"]
ADVERSE_FUSIONS = ["KMT2A_re", "GATA2-MECOM", "DEK-NUP214", "BCR-ABL1"]   # MLLT3-KMT2A t(9;11) is intermediate
ADVERSE_KARYO = [r"(^|,)-5(,|\[|$)", r"del\(5\)\(q", r"(^|,)-7(,|\[|$)", r"(^|,)-17(,|\[|$)", r"del\(17\)\(p",
                 r"i\(17\)\(q", r"inv\(3\)", r"t\(3;3\)", r"t\(6;9\)", r"t\(9;22\)", r"t\(8;16\)", r"t\(3q26"]
SKIP = {"sl", "sdl", "sdl1", "sdl2", "idem", ""}


def karyotype_flags(k) -> tuple[bool, bool]:
    """(adverse cytogenetics, parsed). Complex = >= 3 distinct abnormalities across clones;
    monosomal = >= 2 autosomal monosomies, or 1 plus a structural abnormality."""
    if not isinstance(k, str) or not re.search(r"\d", k):
        return False, False
    k = k.replace(" ", "")
    abn = set()
    for clone in k.split("/"):
        items = re.sub(r"\[\d+\]", "", clone).split(",")
        for it in items[2:]:
            if it not in SKIP and not re.fullmatch(r"\d+~?\d*|X+Y*|XY|XX", it):
                abn.add(it)
    pattern = any(re.search(p, k) for p in ADVERSE_KARYO)
    monos = [a for a in abn if re.fullmatch(r"-\d+", a)]
    struct = [a for a in abn if "(" in a]
    complex_ = len(abn) >= 3
    monosomal = len(monos) >= 2 or (len(monos) == 1 and len(struct) >= 1)
    return pattern or complex_ or monosomal, True


def eln2022(clin: pd.DataFrame, muts: pd.DataFrame) -> pd.DataFrame:
    fus = clin.consensusAMLFusions.fillna("")
    has = lambda f: fus.str.contains(f, regex=False)
    fav_fus = has("RUNX1-RUNX1T1") | has("CBFB-MYH11")
    adv_fus = pd.concat([has(f) for f in ADVERSE_FUSIONS], axis=1).any(axis=1) & ~has("MLLT3-KMT2A")
    kf = clin.karyotype.apply(karyotype_flags)
    adv_kar = kf.str[0]
    parsed = kf.str[1]
    pos = lambda g: muts[g].fillna(0).gt(0) if g in muts else pd.Series(False, index=clin.index)
    npm1, itd, tp53 = pos("NPM1"), pos("FLT3_ITD"), pos("TP53")
    mr = pd.concat([pos(g) for g in MR_GENES], axis=1).any(axis=1)
    cebpa = clin.CEBPA_Biallelic.eq("bi")

    risk = pd.Series("Intermediate", index=clin.index)
    risk[cebpa] = "Favorable"
    risk[mr] = "Adverse"
    risk[npm1 & ~itd] = "Favorable"
    risk[npm1 & itd] = "Intermediate"
    risk[tp53] = "Adverse"
    risk[adv_fus | adv_kar] = "Adverse"
    risk[fav_fus] = "Favorable"
    known = parsed | fus.ne("") | clin.ELN2017.notna()
    risk[~known | has("PML-RARA")] = np.nan
    return pd.DataFrame({"ELN2022": risk, "wes_missing": muts.reindex(clin.index)[MR_GENES[0]].isna()
                         if MR_GENES[0] in muts else True, "adverse_cyto": adv_fus | adv_kar})


if __name__ == "__main__":
    from analysis.part2_gaps import rna560
    D = rna560()
    e = eln2022(D["clin"], D["muts"])
    print(pd.crosstab(D["clin"].ELN2017.fillna("missing"), e.ELN2022.fillna("missing"), margins=True))
    print("WES missing:", int(e.wes_missing.sum()))
