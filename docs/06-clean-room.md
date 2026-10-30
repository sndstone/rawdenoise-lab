# 6. Training and data provenance policy

Training starts from random initialisation. No third-party weights or outputs
are used as targets, labels, losses or model-selection signals.

Record data provenance for every training source: origin, exact dataset version
where applicable, licence terms, permitted use, capture date and camera body.
Keep this record with the local training manifests, and review eligibility before
including data in a run.

Training never uses non-commercial datasets in shipped models. Runs using
research-only data must remain separate, and their weights must not be exported
for shipment. Undocumented permissions do not establish eligibility.

Document the public papers and implementation sources used for model design;
preserve any required notices for adapted code. These are engineering provenance
requirements, not legal conclusions.
