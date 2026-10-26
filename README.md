# Deep Representation Learning and Unsupervised Deconvolution for AML Functional Genomics

02-518/02-718 Computational Medicine, Fall 2026
Arrio Gonsalves · Shreyan Nalwad · Vania Halim

Replaces the hand-curated 30-gene PC1 maturation score from Bottomly et al. (2022)
with data-driven latent features (β-VAE, NMF) and tests whether they predict ex vivo
drug AUC better, recover known biology, and define subtypes with distinct survival.

---

## Status

Schema is **verified** — `COLS` in `src/load.py` is filled in and `_CONFIRMED = True`.
Two of the originally guessed column names were wrong; see
[`reports/schema_notes.md`](reports/schema_notes.md) for those corrections, the
file inventory, and twelve data traps worth knowing before you model anything.

The number that matters: **the modeling set is n=429**, not the 942 specimens /
805 patients in the release. That is the expression ∩ drug-screen intersection
after AUC QC (Waves1+2 266 / Waves3+4 163). The survival aim should *not* use
that intersection — it has more power on the 560-specimen RNA-seq set (n=312
after the paper's de novo + complete-outcome filters).

```bash
python -m src.load        # prints every join count
```

Two things the proposal needs that are **not** in the Beat AML release:

- **van Galen et al. 2019 Table S3** — the top-30 marker genes that *define* the
  PC1 baseline. Needed only if you refit PC1 per CV fold. The published PC1
  scores for all 560 specimens are in `beataml2_manuscript_vg_cts.xlsx`, so the
  baseline is usable as-is on the paper's Waves1+2 → Waves3+4 split.
- **TCGA-LAML** — the independent survival-validation cohort. GDC open tier.

---

## You do not need dbGaP

dbGaP **phs001657** is controlled-access and contains *raw sequencing reads*. Everything
this project uses — gene-level expression, drug AUC, mutation calls, clinical and
survival, WGCNA modules — is in the public CC-BY-4.0 release at
[biodev/beataml2.0_data](https://github.com/biodev/beataml2.0_data).

The `_dbgap` suffix in those filenames only means sample IDs are dbGaP-registered.
The files themselves are open. A Data Access Request needs an eRA Commons account,
a CMU signing official, and NIH Data Access Committee review — weeks to months, for
data the proposal never touches.

Apply only if you later decide you need raw reads, which this design does not.

TCGA-LAML, used for survival validation, is likewise open — the expression and
clinical/survival files are in the GDC open tier, no application required.

---

## Setup

```bash
git clone https://github.com/Shreyan-A0I/BeatAML.git
cd BeatAML
```

Environment — either path works, pick one:

```bash
# conda (environment.yml, pins Python 3.11)
conda env create -f environment.yml && conda activate beataml

# or plain venv (requirements.txt, works on Python 3.13)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Data — `data/` is gitignored, so fetch it:

```bash
bash scripts/get_data.sh                      # fresh machine: ~430 MB into data/raw/
bash scripts/link_data.sh ../beataml2.0_data  # or symlink an existing clone, no copy

Rscript scripts/export_wgcna.R                # .RData -> data/interim/*.csv
python scripts/inspect_columns.py > reports/schema.txt
python -m src.load                            # sanity check: prints join counts
```

### The Git LFS trap

`biodev/beataml2.0_data` stores its files with Git LFS. **A plain `git clone` without
git-lfs installed gives you ~130-byte pointer files, not data.** If you already cloned
it and the expression file is tiny, that is what happened:

```bash
git lfs install
cd <your-clone> && git lfs pull
```

`scripts/get_data.sh` checks for this and fails loudly rather than leaving you with
pointers.

### Fill in the schema first

**Already done** — kept here because the guard is the point. The column names are
not documented anywhere online, so `scripts/inspect_columns.py` prints every file's
headers and the distinct values of label-like columns. Its output is committed at
`reports/schema.txt`; the `COLS` mapping in `src/load.py` is filled in from it and
`_CONFIRMED` is flipped. The guard caught two wrong guesses — `specimenType` is the
tissue, not the timepoint, and the AUC table cannot be joined on
`dbgap_dnaseq_sample` because it is blank for 2,652 rows. Both would have failed
silently. If you add a file, re-run the script and re-verify rather than guessing.

---

## Layout

```
data/raw/          fetched by get_data.sh — gitignored
data/interim/      intermediate matrices — gitignored
data/processed/    model-ready matrices — gitignored
scripts/           data fetch/link, schema inspection, WGCNA export
src/               load.py done; features (VAE/NMF/PC1), models, evaluation to come
notebooks/         exploration (outputs stripped — see below)
results/           figures and tables — gitignored
reports/           schema.txt, proposal, write-up
```

**Data is never committed.** The expression matrix is 281 MB and GitHub hard-rejects
anything over 100 MB. `get_data.sh` writes `data/raw/MANIFEST.sha256` — commit that
small file so all three of you can verify identical data.

---

## Three traps specific to this dataset

1. **Patient ≠ specimen.** 805 patients, 942 specimens. Expression is keyed to
   specimen, survival to patient. Splitting on specimens leaks — use
   `patient_groups()` with `GroupKFold`. The proposal's one-specimen-per-patient
   filter mostly sidesteps this; keep the grouping anyway.
2. **Per-drug n, not 942.** The AUC matrix is sparse and counts vary a lot per
   inhibitor. Call `drug_sample_counts()` and set a minimum n before looking at
   any result.
3. **Expression and drug screens only partially overlap.** `build_modeling_frame()`
   prints the three counts. Put those numbers in the report — reviewers will ask.

---

## Working as three people

```bash
nbstripout --install     # once per clone; keeps notebook outputs out of diffs
```

Notebooks are the main source of merge conflicts. Strip outputs, keep exploration in
`notebooks/` with one notebook per person, and move anything shared into `src/`.

Branch per workstream (`vae`, `nmf`, `baseline-pc1`, `survival`), PR into `main`.

---

## Working away from your computer

**GitHub Codespaces** is the simplest option. Open the repo on github.com → Code →
Codespaces → create, then set the environment up as above. (There is no
`.devcontainer/` yet, so this is not yet automatic — worth adding if the three of us
end up living in Codespaces.) Free personal accounts get 120 core-hours and 15 GB
per month; this project fits comfortably. Re-run `scripts/get_data.sh` inside the
Codespace, since `data/` is gitignored.

Compute is not the constraint here — ~800 samples × 20k genes trains a β-VAE in
minutes on CPU. You do not need a GPU for this project.

Other options, roughly in order of hassle:

- **CMU resources.** You likely have PSC Bridges-2 access through CMU. Worth it only
  if you end up wanting GPUs, which this design does not require.
- **Google Colab.** Free GPU, but session timeouts and awkward with a repo workflow.
  Fine for a one-off experiment, poor as the main environment.
- **SSH to your own machine.** Tailscale is the painless way to reach it without port
  forwarding or a static IP. Best if you want your own hardware and storage.

---

## Reproducing

```bash
bash scripts/get_data.sh           # or: bash scripts/link_data.sh ../beataml2.0_data
Rscript scripts/export_wgcna.R
python scripts/inspect_columns.py > reports/schema.txt
python -m src.load                 # prints join counts
```

`scripts/get_manuscript_code.sh` clones the paper's own R analysis code. It is not
vendored here because the upstream repo ships no LICENSE file.
