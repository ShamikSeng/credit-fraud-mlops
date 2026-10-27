"""Generate observability/upi_dashboard.json from the ULB drift dashboard.

Run from the project root:  python observability/build_upi_dashboard.py

Both services export the same metric names, so the UPI dashboard is the ULB one with every
query's job filter switched to the UPI service (see observability/prometheus.yml), its own
uid/title, and descriptions that name the UPI inputs. Regenerate after editing the ULB one.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ULB_JOB, UPI_JOB = "fraud-detection-api", "fraud-detection-upi"

with open(os.path.join(HERE, "grafana_dashboard.json"), encoding="utf-8") as fh:
    dash = json.load(fh)

dash["uid"] = "upi-input-drift"
dash["title"] = "UPI Fraud Model - Input Drift"
dash["tags"] = ["fraud", "upi", "drift", "mlops"]
for panel in dash["panels"]:
    for t in panel.get("targets", []):
        t["expr"] = t["expr"].replace(f'job="{ULB_JOB}"', f'job="{UPI_JOB}"')
    if "description" in panel:
        panel["description"] = (panel["description"]
                                .replace("(V1..V28 + Amount)", "(14 UPI inputs)")
                                .replace("V1..V28 + Amount", "the 14 UPI inputs"))

with open(os.path.join(HERE, "upi_dashboard.json"), "w", encoding="utf-8") as fh:
    json.dump(dash, fh, indent=2, ensure_ascii=False)
    fh.write("\n")
print("wrote upi_dashboard.json")
