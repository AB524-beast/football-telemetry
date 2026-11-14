"""Layer A: next-day fatigue-risk model on REAL SoccerMon subjective data.
Files are WIDE: one row per date (dd.mm.yyyy), one column per player."""
import glob
import json
import os
import sys
from functools import reduce

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from config import NUM_PLAYERS, PROFILES_PATH

BASE = "data/raw/subjective"
WELLNESS = ["fatigue", "mood", "readiness", "sleep_duration",
            "sleep_quality", "soreness", "stress"]          # file stems in wellness/
LOAD_FILE = f"{BASE}/training-load/weekly_load.csv"
MIN_FATIGUE_OBS = 30                                         # drop players with almost no reports


def read_wide(path, name):
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "date"})          # first column is the date
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y", errors="coerce")
    df = df.dropna(subset=["date"])
    long = df.melt(id_vars="date", var_name="player", value_name=name)
    long[name] = pd.to_numeric(long[name], errors="coerce")
    return long


def build_table():
    frames = []
    for m in WELLNESS:
        p = f"{BASE}/wellness/{m}.csv"
        if not os.path.exists(p):
            print("skipping missing file:", p)
            continue
        frames.append(read_wide(p, m))
    if not os.path.exists(LOAD_FILE):
        sys.exit(f"Missing {LOAD_FILE}. Files found:\n" + "\n".join(glob.glob(f"{BASE}/**/*.csv", recursive=True)))
    frames.append(read_wide(LOAD_FILE, "weekly_load"))
    df = reduce(lambda a, b: a.merge(b, on=["date", "player"], how="outer"), frames)

    n_obs = df.groupby("player")["fatigue"].count()
    df = df[df["player"].isin(n_obs[n_obs >= MIN_FATIGUE_OBS].index)]
    print(f"players kept: {df['player'].nunique()}")

    parts = []
    for p, g in df.sort_values("date").groupby("player"):
        g = g.set_index("date").asfreq("D")                  # full daily calendar -> shift(-1) = next day
        g["player"] = p
        g["fatigue_7d"] = g["fatigue"].rolling(7, min_periods=3).mean()
        g["load_chronic28"] = g["weekly_load"].rolling(28, min_periods=14).mean()
        g["load_ratio"] = g["weekly_load"] / g["load_chronic28"].replace(0, np.nan)
        g["load_delta7"] = g["weekly_load"] - g["weekly_load"].shift(7)
        g["next_fatigue"] = g["fatigue"].shift(-1)
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    def tail_cut(s, higher_is_better, max_share=0.30):
        """Most extreme score values whose combined share is <= max_share (a real 'bad tail')."""
        vals = sorted(s.unique(), reverse=not higher_is_better)
        cut = vals[0]
        for v in vals:
            share = (s <= v).mean() if higher_is_better else (s >= v).mean()
            if share > max_share:
                break
            cut = v
        return cut


def fit_eval(df, feats, tr, te, label):
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=0)
    clf.fit(df.loc[tr, feats], df.loc[tr, "y"])
    y = df.loc[te, "y"]
    if y.nunique() < 2:
        print(f"[{label}] test set has one class, cannot score")
        return clf, None, None
    auc = roc_auc_score(y, clf.predict_proba(df.loc[te, feats])[:, 1])
    naive = -df.loc[te, "fatigue"] if HIGHER_IS_BETTER else df.loc[te, "fatigue"]
    nauc = roc_auc_score(y, naive)
    print(f"[{label}] train={int(tr.sum())} test={int(te.sum())} "
          f"test positive rate={y.mean():.2f}  ROC-AUC={auc:.3f}  naive(today's fatigue)={nauc:.3f}")
    return clf, auc, nauc


if __name__ == "__main__":
    df = build_table()

    both = df[["fatigue", "readiness"]].dropna()
    rho = spearmanr(both["fatigue"], both["readiness"])[0]
    HIGHER_IS_BETTER = rho > 0
    print(f"corr(fatigue, readiness) = {rho:.2f}  ->  "
          f"{'HIGHER fatigue score = better' if HIGHER_IS_BETTER else 'HIGHER fatigue score = more fatigued'}")

    df = df.dropna(subset=["next_fatigue", "fatigue"]).reset_index(drop=True)
    cut = tail_cut(df["next_fatigue"], HIGHER_IS_BETTER)
    df["y"] = ((df["next_fatigue"] <= cut) if HIGHER_IS_BETTER else (df["next_fatigue"] >= cut)).astype(int)
    feats = [c for c in WELLNESS + ["fatigue_7d", "weekly_load", "load_chronic28",
                                    "load_ratio", "load_delta7"] if c in df.columns]
    print(f"label: next-day fatigue score {'<=' if HIGHER_IS_BETTER else '>='} {cut}  "
          f"-> positive rate {df['y'].mean():.2f}   rows={len(df)} players={df['player'].nunique()}")

    # 1) unseen PLAYERS
    tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(df, groups=df["player"]))
    mask_tr = pd.Series(False, index=df.index); mask_tr.iloc[tr_i] = True
    fit_eval(df, feats, mask_tr, ~mask_tr, "unseen players")

    # 2) FUTURE dates (train on earlier, test on later) - the stricter, headline number
    split_date = pd.Timestamp("2021-01-01")
    t_tr, t_te = df["date"] < split_date, df["date"] >= split_date
    clf, auc, nauc = fit_eval(df, feats, t_tr, t_te, "future dates")

    joblib.dump({"model": clf, "features": feats}, "risk_model.joblib")
    pick = df[t_te].sample(NUM_PLAYERS, random_state=7)
    risk = clf.predict_proba(pick[feats])[:, 1]
    profiles = {f"player{i + 1:02d}": {"risk": round(float(r), 3)} for i, r in enumerate(risk)}
    os.makedirs(os.path.dirname(PROFILES_PATH), exist_ok=True)
    json.dump(profiles, open(PROFILES_PATH, "w"), indent=2)
    json.dump({"risk_auc_future": None if auc is None else round(auc, 3),
               "naive_auc_future": None if nauc is None else round(nauc, 3),
               "rows": int(len(df))}, open("risk_metrics.json", "w"))
    print("wrote", PROFILES_PATH)