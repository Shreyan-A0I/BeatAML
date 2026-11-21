"""Download TCGA-LAML (GDC open tier): STAR-Counts expression + clinical, into data/raw/tcga_laml/.

    python scripts/get_tcga_laml.py
"""
import io, json, tarfile, urllib.request
from pathlib import Path

OUT = Path("data/raw/tcga_laml"); (OUT / "star").mkdir(parents=True, exist_ok=True)
API = "https://api.gdc.cancer.gov"

def post(path, body):
    req = urllib.request.Request(API + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600).read()

flt = {"op": "and", "content": [
    {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-LAML"}},
    {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
    {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}}]}
hits = json.loads(post("/files", {"filters": flt, "size": 500, "format": "JSON",
                                   "fields": "file_id,file_name,cases.submitter_id,cases.samples.sample_type"}))["data"]["hits"]
manifest = [{"file_id": h["file_id"], "case": h["cases"][0]["submitter_id"],
             "sample_type": h["cases"][0]["samples"][0]["sample_type"]} for h in hits]
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
todo = [m["file_id"] for m in manifest if not (OUT / "star" / f"{m['file_id']}.tsv").exists()]
for i in range(0, len(todo), 20):
    ids = todo[i:i + 20]
    tf = tarfile.open(fileobj=io.BytesIO(post("/data", {"ids": ids})), mode="r:gz")
    for mem in tf.getmembers():
        fid = mem.name.split("/")[0]
        if mem.isfile() and mem.name.endswith(".tsv"):
            (OUT / "star" / f"{fid}.tsv").write_bytes(tf.extractfile(mem).read())
    print(f"{min(i + 20, len(todo))}/{len(todo)}", flush=True)

fields = ",".join(["submitter_id", "demographic.age_at_index", "demographic.vital_status", "demographic.days_to_death",
                   "diagnoses.days_to_last_follow_up", "diagnoses.age_at_diagnosis"])
cases = json.loads(post("/cases", {"filters": {"op": "=", "content": {"field": "project.project_id", "value": "TCGA-LAML"}},
                                    "fields": fields, "size": 1000, "format": "JSON"}))["data"]["hits"]
(OUT / "cases.json").write_text(json.dumps(cases, indent=1))
print("cases", len(cases))
