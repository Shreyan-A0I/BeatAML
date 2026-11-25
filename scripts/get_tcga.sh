#!/usr/bin/env bash
# Download public TCGA-LAML data for src/tcga.py into data/raw/tcga_laml/ (cBioPortal, open access).
set -euo pipefail
D=data/raw/tcga_laml; mkdir -p "$D"; cd "$D"
A=https://www.cbioportal.org/api; S=laml_tcga_pan_can_atlas_2018
curl -sSfL -o data_mrna_seq_v2_rsem.txt https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/$S/data_mrna_seq_v2_rsem.txt
curl -sSf -H "Accept: application/json" -o clinical_patient_api.json "$A/studies/$S/clinical-data?clinicalDataType=PATIENT&projection=SUMMARY"
curl -sSf -X POST -H "Content-Type: application/json" -H "Accept: application/json" -d "{\"sampleListId\":\"${S}_all\"}" \
  -o mutations_api.json "$A/molecular-profiles/${S}_mutations/mutations/fetch?projection=DETAILED"
curl -sSf -H "Accept: application/json" -o laml2013_clinical_PATIENT.json "$A/studies/laml_tcga/clinical-data?clinicalDataType=PATIENT&projection=SUMMARY"
echo "TCGA-LAML files in $D"
