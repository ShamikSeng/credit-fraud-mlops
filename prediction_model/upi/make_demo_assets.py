"""Write the console's UPI presets and sample batch (frontend/presets_upi.json,
frontend/upi_sample_batch.csv). Run after `python -m prediction_model.upi.train`:

    python -m prediction_model.upi.make_demo_assets

The presets are real rows from the simulated stream: a typical legitimate payment, an
account-takeover fraud (the pattern the model learned), a festive-season social-engineering
scam (a pattern that only appears after day 90, which the model has never seen) and a
malformed request. Rows come from days >= 160, so the model has not seen any of them.

The sample batch is 300 payments from the held-out days [60, 90): unseen by the model but
from the regime it was trained for, so batch scoring shows its real holdout behaviour.
"""
import json
import os

import mlflow
import mlflow.sklearn
from prediction_model.config import config
from prediction_model.upi import modeling, spec
from prediction_model.upi.train import load_stream

FRONTEND = os.path.join(os.path.dirname(config.PACKAGE_ROOT), "frontend")
EVAL_START, EVAL_END = 160, 175


def _row(r):
    # No txn_time: presets are model inputs only.
    return {f: (r[f].item() if hasattr(r[f], "item") else r[f]) for f in spec.FEATURES}


def load_model():
    mlflow.set_tracking_uri(config.TRACKING_URI)
    exp = mlflow.get_experiment_by_name(spec.EXPERIMENT_NAME)
    run = mlflow.search_runs([exp.experiment_id], order_by=["metrics.pr_auc DESC"]).iloc[0]
    return mlflow.sklearn.load_model(f"runs:/{run.run_id}{config.MODEL_NAME}")


def main():
    df = load_stream()
    model = load_model()
    pool = modeling.time_slice(df, EVAL_START, EVAL_END).copy()
    pool["p"] = model.predict_proba(pool[spec.FEATURES])[:, 1]

    # A typical everyday payment: daytime grocery, known device and payee, established VPA.
    legit = pool[(pool.is_fraud == 0) & (pool.merchant_category == "grocery")
                 & pool.hour.between(10, 19) & (pool.is_new_device == 0) & (pool.is_new_payee == 0)
                 & (pool.payee_vpa_age_days > 180) & (pool.p < 0.01)].iloc[0]
    takeover = pool[(pool.fraud_type == "account_takeover")
                    & (pool.p > 0.9)].sort_values("p", ascending=False).iloc[0]
    scam = pool[(pool.fraud_type == "social_engineering") & (pool.p < 0.1)
                & (pool.is_new_device == 0) & (pool.payee_vpa_age_days <= 3)].iloc[0]
    bad = _row(legit)
    bad["merchant_category"] = "casino"   # not an allowed category -> HTTP 422

    presets = {
        "legit": {"label": "Legitimate payment", "values": _row(legit)},
        "takeover": {"label": "Account takeover", "values": _row(takeover)},
        "scam": {"label": "Festive-season scam", "values": _row(scam)},
        "bad": {"label": "Malformed input", "values": bad},
    }
    with open(os.path.join(FRONTEND, "presets_upi.json"), "w", encoding="utf-8") as fh:
        json.dump(presets, fh, indent=2)

    holdout = modeling.time_slice(df, spec.TRAIN_END_DAY, spec.HOLDOUT_END_DAY)
    frauds = holdout[holdout.is_fraud == 1].sample(n=30, random_state=7)
    legits = holdout[holdout.is_fraud == 0].sample(n=270, random_state=7)
    batch = frauds._append(legits).sample(frac=1, random_state=7)
    batch[spec.FEATURES + ["fraud_type", spec.TARGET]].to_csv(
        os.path.join(FRONTEND, "upi_sample_batch.csv"), index=False)

    b = modeling.evaluate(model, batch)
    caught = model.predict_proba(batch[spec.FEATURES])[:, 1] >= 0.5
    frauds_only = batch.assign(caught=caught)[batch.is_fraud == 1]
    by_type = frauds_only.groupby("fraud_type").caught.agg(["sum", "count"])
    print(f"presets: takeover p={takeover.p:.3f}, scam p={scam.p:.3f}")
    print(f"sample batch: {b['tp']}/30 frauds caught, {b['fp']} false alarms; by type: "
          + ", ".join(f"{t} {int(r['sum'])}/{int(r['count'])}" for t, r in by_type.iterrows()))


if __name__ == "__main__":
    main()
