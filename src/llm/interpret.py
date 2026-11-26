"""Ask a local LLM (Ollama, e.g. Gemma) to interpret our FIMM replication results and propose
validation steps, then CHECK what it says against the real numbers.

    python -m src.validation.fimm_pear1          # make the results first
    python -m src.llm.interpret                  # talk to Ollama
    python -m src.llm.interpret --dry-run        # just write the prompt, no model call
    python -m src.llm.interpret --model gemma4 --critic

Why the checker matters: LLMs sound confident even when a number is made up. Every numbered
fact the model is allowed to use is given an id (E1, E2, ...). After it answers we
  1. flag any evidence id it cites that does not exist,
  2. flag any number that does not appear (at the model's own rounding) in the evidence it cited for that statement,
  3. flag numbers that cite nothing, and findings that cite no evidence at all.
The model never sees or changes the checker. Flagged lines are marked in the report.

Needs only the standard library. Ollama must be running on the same machine (ollama serve).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLES, OUT = ROOT / "results" / "tables", ROOT / "results" / "llm"
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not DEFAULT_HOST.startswith("http"):
    DEFAULT_HOST = "http://" + DEFAULT_HOST

# Facts from the team's own results write-up (Nov 21). Edit here if those numbers change.
TEAM_FACTS = [
    "Survival: PEAR1 predicts overall survival in patients under 45, adjusted for age, cytogenetic risk, HSC-like state and its own gene module: pooled Beat AML + TCGA hazard ratio 2.67 per SD (95% CI 1.50 to 4.74), 39 deaths. In all ages pooled the hazard ratio is 1.01 (0.80 to 1.26).",
    "Beat AML alone, under 45: hazard ratio 1.90 (0.79 to 4.59), 21 deaths, p = 0.15. TCGA alone, under 45: 3.54 (1.32 to 9.51), 18 deaths. PEAR1 and its gene module (module 3) correlate at about 0.8.",
    "Beat AML drug screen (122 drugs, ~87 young patients per drug), adjusted for HSC-like and age: 53 drugs have q < 0.10 in the young, mostly toward resistance. No drug shows a significant PEAR1 x young interaction, so the drug pattern is not age-specific while the survival effect is.",
    "In Beat AML, trametinib sensitivity of PEAR1-high tumors survives adjustment for RAS-pathway mutations and the monocytic state (effect -0.25, 95% CI -0.49 to -0.01).",
    "Genes correlated with PEAR1 are enriched for TNF-alpha/NF-kB, interferon and inflammatory response signatures (NES 3.2) and depleted for oxidative phosphorylation and fatty-acid metabolism (q < 0.01).",
    "Ex vivo cytarabine AUC did not predict clinical remission (AUROC 0.51), so all drug findings are hypothesis-generating, not treatment advice.",
]
LIMITS = [
    "FIMM measures drug response as sDSS (normalised to healthy controls, higher = more sensitive); Beat AML uses AUC. We use minus sDSS so higher = more resistant in both.",
    "Stemness is adjusted with LSC17 in both cohorts because the van Galen HSC-like signature is not available for FIMM, so this is not the identical adjustment used in the earlier Beat AML analysis.",
    "FIMM is a Helsinki cohort that mixes diagnosis and relapsed/refractory samples and two culture media (CM and MCM); both are covariates. Beat AML is mostly first-timepoint samples.",
]


# ----------------------------------------------------------------------------- evidence
def build_evidence() -> list[tuple[str, str]]:
    import pandas as pd
    s = json.loads((TABLES / "fimm_pear1_summary.json").read_text())
    t = pd.read_csv(TABLES / "fimm_pear1_drugs.csv")
    ev: list[str] = []
    c = s["cohorts"]
    ev.append(f"Cohorts: Beat AML {c['beat_aml_patients']} patients; FIMM {c['fimm_patients']} patients "
              f"({c['fimm_diagnosis']} at diagnosis, {c['fimm_relapse_or_refractory']} relapsed/refractory; "
              f"only {c['fimm_age_under_45']} are under 45). PEAR1 correlates with the stemness score LSC17 at "
              f"{c['corr_pear1_lsc17_beat']:.2f} in Beat AML and {c['corr_pear1_lsc17_fimm']:.2f} in FIMM.")
    ev.append("Convention: effect = change in z-scored drug resistance per 1 SD of PEAR1, adjusted for stemness (LSC17), age and "
              "sample timepoint. Positive effect = PEAR1-high tumors are more resistant; negative = more sensitive.")
    for h in s["headline_drugs"]:
        if "beta_fimm" not in h:
            continue
        pred = "resistant" if h["predicted_sign"] > 0 else "sensitive"
        if h["replicated_p_lt_0.05"] and h["p_fimm_two_sided"] < 0.05:
            verdict = "replicated"
        elif h["replicated_p_lt_0.05"]:
            verdict = "same direction; significant only with the one-sided test, so borderline"
        else:
            verdict = "same direction but not significant" if h["same_direction_as_predicted"] else "opposite direction"
        ev.append(f"{h['drug']} (prediction stated beforehand: PEAR1-high {pred}): Beat AML effect {h['beta_beat']:+.2f} (n={h['n_beat']}); "
                  f"FIMM effect {h['beta_fimm']:+.2f} with standard error {h['se_fimm']:.2f} (n={h['n_fimm']}); "
                  f"two-sided p = {h['p_fimm_two_sided']:.3f}, one-sided p in predicted direction = {h['p_fimm_one_sided_predicted_direction']:.3f}. Result: {verdict}.")
    a = s["agreement_primary"]
    pct = 100 * a["same_sign_in_fimm"] / max(a["n_beat_significant_q10"], 1)
    ev.append(f"Across {a['n_shared_drugs']} drugs tested in both cohorts, the PEAR1 effect sizes correlate: Spearman {a['spearman_effects']:+.2f} (p = {a['spearman_p']:.3f}).")
    ev.append(f"Of {a['n_beat_significant_q10']} drugs with q < 0.10 in Beat AML, {a['same_sign_in_fimm']} ({pct:.0f}%) point the same direction in FIMM (sign test p = {a['sign_test_p']:.3f}).")
    b = s["agreement_pear1_only"]
    ev.append(f"Without any adjustment the agreement is weaker: Spearman {b['spearman_effects']:+.2f} (p = {b['spearman_p']:.3f}); {b['same_sign_in_fimm']} of {b['n_beat_significant_q10']} significant drugs agree in direction.")
    sig = t[t["q_beat"] < 0.10].sort_values("beta_beat")
    for _, r in sig.iterrows():
        mech = r["mechanism"] if isinstance(r["mechanism"], str) else "unknown mechanism"
        ev.append(f"Drug {r['drug_fimm']} ({mech[:70]}): Beat AML effect {r['beta_beat']:+.2f} (q = {r['q_beat']:.3f}), FIMM effect {r['beta_fimm']:+.2f} (p = {r['p_fimm']:.3f}).")
    ev += ["Team result: " + f for f in TEAM_FACTS]
    ev += ["Limitation: " + f for f in LIMITS]
    return [(f"E{i}", text) for i, text in enumerate(ev, 1)]


SYSTEM = """You are a careful biomedical data scientist helping a student team interpret results from an acute myeloid leukemia (AML) study.
Rules:
1. Use ONLY the numbered evidence for facts and numbers. Cite evidence ids like [E3].
2. Never invent numbers, p-values, drug names or studies. If something is not in the evidence, say it is unknown.
3. General biology ideas are allowed ONLY inside "hypotheses", and each must be marked speculative.
4. These are ex vivo and retrospective results. Do not give treatment advice.
5. Reply with ONE JSON object and nothing else."""

SCHEMA = """{
  "plain_summary": "under 120 words, simple language, cite ids like [E3] right after every number",
  "findings": [{"statement": "...", "evidence": ["E1"], "strength": "strong|moderate|weak"}],
  "hypotheses": [{"statement": "...", "evidence": ["E1"], "speculative": true}],
  "validation_steps": [{"step": "...", "tests": "which hypothesis or finding", "data_needed": "...", "difficulty": "easy|medium|hard"}],
  "caveats": ["..."]
}"""

TASK = """Task:
A) Summarise what the FIMM check shows about the PEAR1 drug pattern: what replicated, what did not, how strong it is.
B) Give 3 to 6 findings, each citing evidence ids.
C) Give 2 to 4 mechanistic hypotheses that could explain the pattern (including why venetoclax and trametinib might behave as they do).
D) Propose 4 to 6 validation steps, ranked from best value for effort. Prefer steps doable with: Beat AML, FIMM, TCGA-LAML, other public AML data, or a basic lab experiment.
E) List the main caveats.
Return JSON in exactly this shape:
"""

CRITIC_SYSTEM = """You are a skeptical statistical reviewer. You are given evidence and a draft interpretation. List the strongest reasons the conclusions could be wrong or overstated. Use only the evidence ids given; do not invent numbers. Reply with ONE JSON object: {"objections": [{"point": "...", "evidence": ["E1"], "how_to_check": "..."}]}"""


def build_prompt(evidence: list[tuple[str, str]]) -> str:
    lines = "\n".join(f"[{i}] {t}" for i, t in evidence)
    return f"Evidence:\n{lines}\n\n{TASK}{SCHEMA}"


# ----------------------------------------------------------------------------- Ollama
def _post(host: str, path: str, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(host.rstrip("/") + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def installed_models(host: str) -> list[str]:
    with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=10) as r:
        return [m["name"] for m in json.loads(r.read()).get("models", [])]


def chat(host: str, model: str, system: str, user: str, temperature: float = 0.2, timeout: int = 900) -> str:
    payload = {"model": model, "stream": False, "format": "json",
               "options": {"temperature": temperature, "num_ctx": 8192},
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    try:
        return _post(host, "/api/chat", payload, timeout)["message"]["content"]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            try:
                have = ", ".join(installed_models(host)) or "none"
            except Exception:
                have = "could not list"
            sys.exit(f"Ollama does not have a model called '{model}'. Installed: {have}\n"
                     f"Use --model NAME (see `ollama list`).")
        sys.exit(f"Ollama returned HTTP {e.code}: {e.read()[:300]!r}")
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        sys.exit(f"Could not reach Ollama at {host} ({e}).\nStart it with `ollama serve` (or open the Ollama app) and try again.")


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def ask_json(host, model, system, user, retries=1) -> dict:
    msg = user
    for attempt in range(retries + 1):
        raw = chat(host, model, system, msg)
        try:
            return parse_json(raw)
        except json.JSONDecodeError:
            msg = user + "\n\nYour last reply was not valid JSON. Reply again with ONLY the JSON object."
    sys.exit("The model did not return valid JSON after a retry. Try a larger model or --model.")


# ----------------------------------------------------------------------------- checker
NUM = re.compile(r"(?<![A-Za-z0-9.])[-+]?\d+(?:\.\d+)?")
ID = re.compile(r"\bE\d+\b")


def numbers_in(text: str) -> list[float]:
    return [abs(float(x)) for x in NUM.findall(ID.sub(" ", text))]


def matches(tok: str, pool: list[float]) -> bool:
    """True if the model's number equals some number in `pool` once rounded to the model's own precision."""
    x = abs(float(tok))
    d = len(tok.split(".")[1]) if "." in tok else 0
    return any(abs(round(n, d) - x) < 1e-9 for n in pool)


def check_text(text: str, by_id: dict[str, list[float]], fallback: tuple[str, ...]) -> dict:
    """Problems in one string. A number must match the evidence the string (or its item) cites."""
    cited = ID.findall(text)
    out = {"unknown_ids": sorted({i for i in cited if i not in by_id}), "unsupported_numbers": [], "uncited_numbers": []}
    ids = [i for i in cited if i in by_id] or [i for i in fallback if i in by_id]
    pool = [n for i in ids for n in by_id[i]]
    for tok in NUM.findall(ID.sub(" ", text)):
        if "." not in tok and abs(float(tok)) <= 5:   # small counts like "two cohorts" are not checked
            continue
        if not ids:
            out["uncited_numbers"].append(tok)
        elif not matches(tok, pool):
            out["unsupported_numbers"].append(tok)
    return out


def check_report(report: dict, evidence) -> dict:
    by_id = {i: numbers_in(t) for i, t in evidence}
    flags, stats = {}, {"checked": 0, "unsupported": 0, "uncited": 0}

    def visit(path, obj, fallback=()):
        if isinstance(obj, str):
            p = check_text(obj, by_id, fallback)
            stats["checked"] += len([t for t in NUM.findall(ID.sub(" ", obj)) if "." in t or abs(float(t)) > 5])
            stats["unsupported"] += len(p["unsupported_numbers"])
            stats["uncited"] += len(p["uncited_numbers"])
            if any(p.values()):
                flags[path] = {k: v for k, v in p.items() if v}
        elif isinstance(obj, dict):
            ev = obj.get("evidence")
            fb = tuple(x for x in ev if isinstance(x, str)) if isinstance(ev, list) else fallback
            for k, v in obj.items():
                visit(f"{path}.{k}" if path else k, v, fb)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                visit(f"{path}[{i}]", v, fallback)

    visit("", report)
    for i, f in enumerate(report.get("findings", [])):
        if not f.get("evidence"):
            flags.setdefault(f"findings[{i}]", {})["no_evidence_cited"] = True
    return {"flags": flags, "numbers_checked": stats["checked"], "numbers_unsupported": stats["unsupported"],
            "numbers_uncited": stats["uncited"]}


# ----------------------------------------------------------------------------- report
def render_md(report, critic, check, evidence, model) -> str:
    fl = check["flags"]

    def mark(path):
        return " ⚠" if any(k == path or k.startswith(path + ".") for k in fl) else ""

    L = [f"# LLM interpretation of the FIMM replication ({model})", "",
         f"Automatic check of the numbers in the model's text: {check['numbers_checked']} checked, "
         f"{check['numbers_unsupported']} do not match the evidence they cite, {check['numbers_uncited']} cite no evidence at all; "
         f"{len(fl)} item(s) flagged ⚠ in total (also unknown evidence ids or findings with no evidence).",
         "The model's text is NOT verified science; the check only catches invented numbers and ids.", "",
         "## Plain summary", report.get("plain_summary", "")]
    for title, key in (("Findings", "findings"), ("Hypotheses (speculative)", "hypotheses")):
        L += ["", f"## {title}"]
        for i, f in enumerate(report.get(key, [])):
            tag = f" [{f.get('strength')}]" if f.get("strength") else ""
            L.append(f"- {f.get('statement', '')}{tag} — evidence: {', '.join(f.get('evidence', [])) or 'none'}{mark(f'{key}[{i}]')}")
    L += ["", "## Suggested validation steps"]
    for i, v in enumerate(report.get("validation_steps", []), 1):
        L.append(f"{i}. **{v.get('step', '')}** (tests: {v.get('tests', '?')}; data: {v.get('data_needed', '?')}; {v.get('difficulty', '?')}){mark(f'validation_steps[{i - 1}]')}")
    L += ["", "## Caveats"] + [f"- {c}{mark(f'caveats[{i}]')}" for i, c in enumerate(report.get("caveats", []))]
    if critic:
        L += ["", "## Skeptical reviewer (second model pass)"]
        for o in critic.get("objections", []):
            L.append(f"- {o.get('point', '')} (evidence: {', '.join(o.get('evidence', [])) or 'none'}) — check: {o.get('how_to_check', '')}")
    if fl:
        L += ["", "## Flagged items"] + [f"- `{p}`: {json.dumps(v)}" for p, v in fl.items()]
    L += ["", "## Evidence the model was given"] + [f"- [{i}] {t}" for i, t in evidence]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--dry-run", action="store_true", help="write the prompt and stop")
    ap.add_argument("--critic", action="store_true", help="add a skeptical-reviewer pass")
    ap.add_argument("--list-models", action="store_true")
    a = ap.parse_args()

    if a.list_models:
        print("\n".join(installed_models(a.host)))
        return
    if not (TABLES / "fimm_pear1_summary.json").exists():
        sys.exit("Run `python -m src.validation.fimm_pear1` first (it writes the results this reads).")

    OUT.mkdir(parents=True, exist_ok=True)
    evidence = build_evidence()
    prompt = build_prompt(evidence)
    (OUT / "prompt.txt").write_text(f"SYSTEM:\n{SYSTEM}\n\nUSER:\n{prompt}\n")
    print(f"evidence items: {len(evidence)} | prompt saved to {OUT / 'prompt.txt'}")
    if a.dry_run:
        return

    print(f"asking {a.model} at {a.host} (can take a few minutes) ...")
    report = ask_json(a.host, a.model, SYSTEM, prompt)
    critic = None
    if a.critic:
        print("asking for a skeptical review ...")
        critic = ask_json(a.host, a.model, CRITIC_SYSTEM,
                          "Evidence:\n" + "\n".join(f"[{i}] {t}" for i, t in evidence)
                          + "\n\nDraft interpretation:\n" + json.dumps(report, indent=1))
    check = check_report({**report, **({"critic": critic} if critic else {})}, evidence)
    (OUT / "interpretation.json").write_text(json.dumps({"model": a.model, "report": report, "critic": critic, "check": check}, indent=2))
    (OUT / "interpretation.md").write_text(render_md(report, critic, check, evidence, a.model))
    print(f"numbers checked {check['numbers_checked']}: unsupported {check['numbers_unsupported']}, uncited {check['numbers_uncited']}; flagged items {len(check['flags'])}")
    print(f"wrote {OUT / 'interpretation.md'}")


if __name__ == "__main__":
    main()
