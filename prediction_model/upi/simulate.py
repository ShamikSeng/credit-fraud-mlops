"""Seeded simulator for a UPI-style transaction stream with a documented drift.

Run:  python -m prediction_model.upi.simulate            (writes datasets/upi_transactions.csv)
      python -m prediction_model.upi.simulate --seed 7   (another replicate)

Why simulate: no public transaction-level UPI dataset with fraud labels exists (NPCI and the
banks publish only aggregates). Simulating gives a stream long enough to study drift and lets
us state exactly what changed and when, which a controlled experiment needs.

The stream has two regimes:

Days [0, DRIFT_START_DAY): normal months.
  Fraud (~0.28%) is account takeover: a fraudster on a NEW DEVICE, often at NIGHT, drains
  the account with LARGE P2P transfers to mule accounts in quick succession.

Days [DRIFT_START_DAY, SIM_DAYS): festive-sale season, which changes both inputs and fraud.
  Covariate drift: 35% more traffic, more e-commerce/travel/entertainment, bigger baskets,
    more evening shopping, more first-time payees (newly onboarded sellers), and a cashback
    campaign that moves some traffic to other UPI apps.
  Concept drift: most fraud (~70% of a higher ~0.45% rate) becomes social engineering, i.e.
    fake-merchant collect requests and QR scams. The VICTIM approves the payment on their OWN
    device, in the DAYTIME, for a MODEST amount, to a VPA created days ago. These are exactly
    the cases a model trained on account takeover does not recognise.

Every number below is a visible parameter; nothing is calibrated to, or claimed to match,
real NPCI statistics.
"""
import argparse
import os

import numpy as np
import pandas as pd

from prediction_model.upi import spec

P2M_CATEGORIES = [c for c in spec.CATEGORIES["merchant_category"] if c != "p2p_transfer"]

# Share of P2M traffic per merchant category, before and during the festive season.
P2M_MIX_PRE = dict(grocery=.22, food_delivery=.14, ecommerce=.13, fuel=.08, utilities=.10,
                   telecom_recharge=.10, travel=.05, entertainment=.05, healthcare=.05,
                   education=.03, gaming=.05)
P2M_MIX_POST = dict(grocery=.17, food_delivery=.13, ecommerce=.23, fuel=.06, utilities=.07,
                    telecom_recharge=.08, travel=.09, entertainment=.07, healthcare=.04,
                    education=.02, gaming=.04)
# Typical basket size relative to the payer's usual amount.
CATEGORY_AMOUNT = dict(p2p_transfer=1.0, grocery=0.8, food_delivery=0.6, ecommerce=1.6,
                       fuel=1.2, utilities=1.4, telecom_recharge=0.5, travel=3.0,
                       entertainment=0.9, healthcare=1.5, education=2.5, gaming=0.7)
FESTIVE_AMOUNT_BOOST = {"ecommerce": 1.6, "travel": 1.5, "entertainment": 1.4}

STATE_P = [.15, .11, .12, .09, .09, .08, .07, .07, .06, .05, .11]
BANK_P = [.27, .15, .13, .09, .07, .07, .06, .06, .05, .05]
APP_P = [.47, .35, .09, .03, .06]
CAMPAIGN_APP_P = [.25, .25, .35, .10, .05]   # app choice for campaign-driven payments
CAMPAIGN_SHARE = 0.30                        # share of festive payments made via the campaign

# Relative transaction volume by hour of day (legitimate traffic).
LEGIT_HOUR_W = np.array([.6, .4, .3, .25, .3, .6, 1.5, 3, 4.5, 5.5, 6, 6.2, 6.5, 6.2, 5.8,
                         5.6, 5.6, 6, 6.5, 7, 7, 6.2, 4.5, 2.5])
FESTIVE_EVENING = np.where(np.arange(24) >= 19, 1.8, 1.0)
WEEKDAY_VOLUME = [1.0, 1.0, 1.0, 1.0, 1.05, 1.15, 1.10]   # Monday..Sunday

FRAUD_RATE_PRE = 0.0028
FRAUD_RATE_POST = 0.0045
SCAM_SHARE_POST = 0.70      # share of festive-season fraud that is social engineering


def _probs(weights):
    w = np.asarray(weights, dtype=float)
    return w / w.sum()


def _make_users(rng, n_users):
    s = spec.CATEGORIES
    return {
        "state": rng.choice(s["state"], n_users, p=_probs(STATE_P)),
        "payer_bank": rng.choice(s["payer_bank"], n_users, p=_probs(BANK_P)),
        "upi_app": rng.choice(s["upi_app"], n_users, p=_probs(APP_P)),
        "device_type": rng.choice(s["device_type"], n_users, p=[.82, .18]),
        "avg_amount": np.exp(rng.normal(np.log(450), 0.7, n_users)),
        "activity": rng.lognormal(0.0, 0.8, n_users),
    }


def _hours(rng, n, festive):
    w = LEGIT_HOUR_W * (FESTIVE_EVENING if festive else 1.0)
    return rng.choice(24, n, p=_probs(w))


def _base_rows(rng, users, uid, festive):
    """Columns that come from the payer: state, bank, device, app (with campaign switching)."""
    app = users["upi_app"][uid].copy()
    if festive:
        switch = rng.random(len(uid)) < CAMPAIGN_SHARE
        app[switch] = rng.choice(spec.CATEGORIES["upi_app"], switch.sum(), p=_probs(CAMPAIGN_APP_P))
    return {
        "state": users["state"][uid],
        "payer_bank": users["payer_bank"][uid],
        "device_type": users["device_type"][uid],
        "upi_app": app,
    }


def _legit(rng, users, n, festive):
    n_users = len(users["avg_amount"])
    uid = rng.choice(n_users, n, p=_probs(users["activity"]))
    p2m = rng.random(n) < (0.78 if festive else 0.65)
    mix = P2M_MIX_POST if festive else P2M_MIX_PRE
    category = np.where(p2m, rng.choice(list(mix), n, p=_probs(list(mix.values()))), "p2p_transfer")

    mult = np.array([CATEGORY_AMOUNT[c] for c in category])
    if festive:
        mult *= np.array([FESTIVE_AMOUNT_BOOST.get(c, 1.0) for c in category])
    ratio = mult * rng.lognormal(0.0, 0.55, n)

    new_payee = rng.random(n) < np.where(p2m, 0.40 if festive else 0.20, 0.12)
    # New sellers onboarded for the sale have young VPAs, so legit new payees get younger too.
    new_payee_age = np.where(festive & p2m, 60, 120)
    vpa_age = np.where(new_payee,
                       rng.lognormal(np.log(new_payee_age), 1.0, n),
                       rng.lognormal(np.log(700), 0.8, n))
    rows = _base_rows(rng, users, uid, festive)
    rows.update({
        "txn_type": np.where(p2m, "P2M", "P2P"),
        "merchant_category": category,
        "hour": _hours(rng, n, festive),
        "is_new_device": (rng.random(n) < 0.02).astype(int),
        "is_new_payee": new_payee.astype(int),
        "payee_vpa_age_days": vpa_age,
        "txns_last_24h": rng.poisson(1.2 * np.sqrt(users["activity"][uid])) + 1,
        "amount_to_user_avg": ratio,
        "avg_amount": users["avg_amount"][uid],
        "fraud_type": "none",
        "is_fraud": 0,
    })
    return rows


def _account_takeover(rng, users, n, festive):
    """New device, often at night, large P2P transfers to mule accounts, in bursts."""
    uid = rng.integers(0, len(users["avg_amount"]), n)
    p2p = rng.random(n) < 0.85
    category = np.where(p2p, "p2p_transfer",
                        rng.choice(["ecommerce", "gaming", "travel"], n, p=[.5, .3, .2]))
    night = rng.random(n) < 0.50
    hour = np.where(night, rng.integers(0, 6, n), _hours(rng, n, festive))
    rows = _base_rows(rng, users, uid, festive)
    rows.update({
        "txn_type": np.where(p2p, "P2P", "P2M"),
        "merchant_category": category,
        "hour": hour,
        "is_new_device": (rng.random(n) < 0.78).astype(int),
        "is_new_payee": (rng.random(n) < 0.90).astype(int),
        "payee_vpa_age_days": rng.lognormal(np.log(70), 1.1, n),
        "txns_last_24h": rng.poisson(4, n) + 1,
        "amount_to_user_avg": rng.lognormal(np.log(4.0), 0.7, n),
        "avg_amount": users["avg_amount"][uid],
        "fraud_type": "account_takeover",
        "is_fraud": 1,
    })
    return rows


def _social_engineering(rng, users, n, festive):
    """Victim approves a collect request / QR on their own device, daytime, modest amount,
    to a freshly created (fake-merchant) VPA."""
    uid = rng.integers(0, len(users["avg_amount"]), n)
    p2m = rng.random(n) < 0.85
    category = np.where(p2m, rng.choice(["ecommerce", "gaming", "travel", "entertainment",
                                         "telecom_recharge"], n, p=[.45, .25, .15, .10, .05]),
                        "p2p_transfer")
    hour = np.clip(np.round(rng.normal(14, 3.5, n)), 8, 22).astype(int)
    rows = _base_rows(rng, users, uid, festive)
    rows.update({
        "txn_type": np.where(p2m, "P2M", "P2P"),
        "merchant_category": category,
        "hour": hour,
        "is_new_device": (rng.random(n) < 0.04).astype(int),
        "is_new_payee": (rng.random(n) < 0.92).astype(int),
        "payee_vpa_age_days": rng.lognormal(np.log(5), 0.7, n),
        "txns_last_24h": rng.poisson(1.5, n) + 1,
        "amount_to_user_avg": rng.lognormal(np.log(1.8), 0.5, n),
        "avg_amount": users["avg_amount"][uid],
        "fraud_type": "social_engineering",
        "is_fraud": 1,
    })
    return rows


def simulate(n_days=spec.SIM_DAYS, seed=42, n_users=30000, base_daily=1500,
             drift_day=spec.DRIFT_START_DAY):
    rng = np.random.default_rng(seed)
    users = _make_users(rng, n_users)
    start_dow = pd.Timestamp(spec.SIM_START_DATE).dayofweek
    blocks = []
    for day in range(n_days):
        festive = day >= drift_day
        dow = (start_dow + day) % 7
        n = int(base_daily * WEEKDAY_VOLUME[dow] * (1.35 if festive else 1.0) * rng.normal(1, 0.05))
        n_fraud = rng.binomial(n, FRAUD_RATE_POST if festive else FRAUD_RATE_PRE)
        n_scam = rng.binomial(n_fraud, SCAM_SHARE_POST) if festive else 0
        parts = [_legit(rng, users, n - n_fraud, festive),
                 _account_takeover(rng, users, n_fraud - n_scam, festive),
                 _social_engineering(rng, users, n_scam, festive)]
        day_df = pd.concat([pd.DataFrame(p) for p in parts if len(p["hour"])], ignore_index=True)
        minute = rng.integers(0, 60, len(day_df))
        day_df["txn_time"] = day + (day_df["hour"] + minute / 60) / 24
        day_df["day_of_week"] = dow
        blocks.append(day_df)

    df = pd.concat(blocks, ignore_index=True).sort_values("txn_time", kind="stable")
    df["amount"] = np.round(np.maximum(df["avg_amount"] * df["amount_to_user_avg"], 1.0), 2)
    df["amount_to_user_avg"] = np.round(df["amount"] / df["avg_amount"], 3)
    df["payee_vpa_age_days"] = np.maximum(np.round(df["payee_vpa_age_days"]), 1).astype(int)
    df["txn_time"] = df["txn_time"].round(5)
    df.insert(0, "txn_id", [f"UPI{seed:02d}{i:07d}" for i in range(len(df))])
    ts = pd.Timestamp(spec.SIM_START_DATE) + pd.to_timedelta(df["txn_time"], unit="D")
    df.insert(1, "timestamp", ts.dt.strftime("%Y-%m-%d %H:%M"))
    cols = ["txn_id", "timestamp", spec.TIME_COL] + spec.FEATURES + ["fraud_type", spec.TARGET]
    return df[cols].reset_index(drop=True)


def main():
    from prediction_model.config import config

    parser = argparse.ArgumentParser(description="Simulate a UPI-style transaction stream.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=spec.SIM_DAYS)
    parser.add_argument("--out", default=os.path.join(config.DATAPATH, spec.DATA_FILE))
    args = parser.parse_args()

    df = simulate(n_days=args.days, seed=args.seed)
    df.to_csv(args.out, index=False)
    pre = df[df.txn_time < spec.DRIFT_START_DAY]
    post = df[df.txn_time >= spec.DRIFT_START_DAY]
    print(f"Wrote {len(df):,} transactions over {args.days} days to {args.out}")
    print(f"  before day {spec.DRIFT_START_DAY}: {len(pre):,} txns, fraud rate {pre.is_fraud.mean():.3%}")
    print(f"  from day {spec.DRIFT_START_DAY}:  {len(post):,} txns, fraud rate {post.is_fraud.mean():.3%}")
    print("  fraud types:", df[df.is_fraud == 1].fraud_type.value_counts().to_dict())


if __name__ == "__main__":
    main()
