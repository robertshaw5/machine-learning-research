import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
from dateutil.relativedelta import relativedelta

df = pd.read_csv("datasets/race_results_a_grade.csv", low_memory=False)
# String dtype keeps fractional SP (7/4f), N/P, and <NA> distinct from botched float inference.
if "sp" in df.columns:
    df["sp"] = df["sp"].astype("string")
# dog history: one row per dog per race (used for point-in-time form)
forms_df = pd.read_csv("datasets/full_dog_forms.csv", low_memory=False)


@dataclass
class FormIndex:
    """Pre-indexed forms: O(1) dog lookup + vectorized prior filter (avoids full-table scans)."""

    empty: pd.DataFrame
    by_dog: Dict[str, pd.DataFrame]
    whelp_first: Dict[str, pd.Timestamp]


def build_form_index(forms: pd.DataFrame) -> FormIndex:
    """One-time preprocessing: strip names, parse dates, race flag, yards; group by dog."""
    dist_str = forms["dist"].astype(str)
    f = forms.assign(
        _gk=forms["greyhound"].astype(str).str.strip(),
        _fd=pd.to_datetime(forms["date"], format="%d-%b-%y", errors="coerce"),
        _is_race=dist_str.str[-1] == "R",
        _yards=pd.to_numeric(
            dist_str.str.extract(r"^(\d+)", expand=False),
            errors="coerce",
        ),
    )
    empty = f.iloc[0:0].copy()
    by_dog: Dict[str, pd.DataFrame] = {}
    whelp_first: Dict[str, pd.Timestamp] = {}
    for name, grp in f.groupby("_gk", sort=False):
        by_dog[name] = grp
        wh = pd.to_datetime(grp["whelp_date"], format="%d-%b-%y", errors="coerce")
        wh = wh[wh.notna()]
        if len(wh):
            whelp_first[name] = wh.iloc[0]
    return FormIndex(empty=empty, by_dog=by_dog, whelp_first=whelp_first)


def get_prior_dog_form(line_or_dog, race_dt, fi: FormIndex, dog_key: Optional[str] = None):
    """Prior form rows strictly before race_dt. Pass (line, fi) or set dog_key + race_dt."""
    if dog_key is None:
        dog_key = str(line_or_dog["greyhound"]).strip()
    else:
        dog_key = str(dog_key).strip()
    g = fi.by_dog.get(dog_key)
    if g is None or len(g) == 0:
        return fi.empty
    # numpy comparison avoids extra index alignment overhead in hot path
    rd = pd.Timestamp(race_dt)
    mask = g["_fd"].to_numpy(dtype="datetime64[ns]") < rd.to_datetime64()
    return g.loc[mask]


def get_dataset_info(df):
    # one row = one dog in one race
    unique_races = df.drop_duplicates(
        subset=["meeting_date", "meeting_venue", "race_number"]
    ).shape[0]
    num_dogs = len(df)  # total dog results / rows
    unique_dogs = df["greyhound"].nunique()

    dates = pd.to_datetime(df["meeting_date"], format="%d-%b-%y")
    first_date = dates.min()
    latest_date = dates.max()

    print(
        f"The dataset has {unique_races} unique races, "
        f"{num_dogs} dog results (rows), "
        f"and {unique_dogs} unique dogs. "
        f"Dates run from {first_date.strftime('%d %b %Y')} to {latest_date.strftime('%d %b %Y')}."
    )


# only prior form for a dog prior to the race date — use get_prior_dog_form(..., fi) with FormIndex


def keep_only_races(form_df):
    # dist = yards + one letter at end (ignore letter for distance; T=trial, R=race)
    if "_is_race" in form_df.columns:
        return form_df[form_df["_is_race"]]
    return form_df[form_df["dist"].astype(str).str[-1] == "R"]


def get_grade(grade_str):
    # ordinal: lower = better class (AA0 best, then A0, A1…A10). Split grades use part before /
    if pd.isna(grade_str):
        return float("nan")
    s = str(grade_str).upper().split("/")[0].strip()
    if not s:
        return float("nan")
    if s.startswith("AA"):
        rest = s[2:]
        if rest.isdigit():
            return float(int(rest))
        return float("nan")
    if s.startswith("A") and len(s) > 1:
        rest = s[1:]
        if rest.isdigit():
            return float(int(rest) + 1)
    return float("nan")


def grade_change_flag(line, prior_races_df):
    # direction only vs last real race (not how big the change is): +1 better class, -1 worse, 0 same
    # better class = lower get_grade() number; prior_races_df = prior runs only (R), from FormIndex slice
    cur = get_grade(line.get("race_grade"))
    if len(prior_races_df) == 0 or pd.isna(cur):
        return float("nan")
    if "_fd" in prior_races_df.columns:
        tmp = prior_races_df.sort_values("_fd", kind="mergesort")
    else:
        d = pd.to_datetime(prior_races_df["date"], format="%d-%b-%y", errors="coerce")
        tmp = prior_races_df.assign(_d=d).sort_values("_d", kind="mergesort")
    pr = get_grade(tmp.iloc[-1]["race_grade"])
    if pd.isna(pr):
        return float("nan")
    if cur < pr:
        return 1
    if cur > pr:
        return -1
    return 0


def avg_est_time(races_df):
    # form file column is est_tm (estimated time); * means fastest of the night — strip for averaging
    cleaned = races_df["est_tm"].astype(str).str.rstrip("*")
    times = pd.to_numeric(cleaned, errors="coerce")
    return times.mean()


def avg_sectional_time(races_df):
    # sct_t = sectional time for that run
    secs = pd.to_numeric(races_df["sct_t"], errors="coerce")
    return secs.mean()


def bend_position_averages(races_df):
    # sct_p like 4444 or 1231 = position at bend 1,2,3,4; only rows with 4 digits count
    s = races_df["sct_p"].astype(str).str.strip()
    ok = s.str.match(r"^\d{4}$", na=False)
    s = s[ok]
    if len(s) == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    b1 = pd.to_numeric(s.str[0], errors="coerce")
    b2 = pd.to_numeric(s.str[1], errors="coerce")
    b3 = pd.to_numeric(s.str[2], errors="coerce")
    b4 = pd.to_numeric(s.str[3], errors="coerce")
    return (b1.mean(), b2.mean(), b3.mean(), b4.mean())


def count_wins(races_df):
    # place 1 = win
    place = pd.to_numeric(races_df["place"], errors="coerce")
    return int((place == 1).sum())


def prior_race_count(races_df):
    # races_df = prior runs only (already races, not trials)
    return len(races_df)


def career_win_rate(races_df):
    n = prior_race_count(races_df)
    if n == 0:
        return float("nan")
    return count_wins(races_df) / n


def prior_race_count_at_distance(races_df, line):
    race_yards = pd.to_numeric(line["race_distance"], errors="coerce")
    if pd.isna(race_yards):
        return float("nan")
    if "_yards" in races_df.columns:
        at_dist = races_df[races_df["_yards"] == race_yards]
    else:
        form_yards = pd.to_numeric(
            races_df["dist"].astype(str).str.extract(r"^(\d+)", expand=False),
            errors="coerce",
        )
        at_dist = races_df[form_yards == race_yards]
    return len(at_dist)


def win_rate_at_distance(races_df, line):
    race_yards = pd.to_numeric(line["race_distance"], errors="coerce")
    if pd.isna(race_yards):
        return float("nan")
    if "_yards" in races_df.columns:
        at_dist = races_df[races_df["_yards"] == race_yards]
    else:
        form_yards = pd.to_numeric(
            races_df["dist"].astype(str).str.extract(r"^(\d+)", expand=False),
            errors="coerce",
        )
        at_dist = races_df[form_yards == race_yards]
    n = len(at_dist)
    if n == 0:
        return float("nan")
    return count_wins(at_dist) / n


def win_rate_at_track(races_df, line):
    href = str(line.get("source_href", "") or "")
    m = re.search(r"[?&]track=([A-Za-z]+)", href)
    if not m:
        return float("nan")
    code = m.group(1).upper()
    same = races_df[races_df["track"].astype(str).str.strip().str.upper() == code]
    n = len(same)
    if n == 0:
        return float("nan")
    return count_wins(same) / n


def avg_place(races_df):
    # mean finishing position over prior races (lower is better)
    place = pd.to_numeric(races_df["place"], errors="coerce")
    return place.mean()


def avg_place_at_distance(races_df, line):
    # mean place only on runs at the same yard distance as this race
    race_yards = pd.to_numeric(line["race_distance"], errors="coerce")
    if pd.isna(race_yards):
        return float("nan")
    if "_yards" in races_df.columns:
        same_distance = races_df[races_df["_yards"] == race_yards]
    else:
        form_yards = pd.to_numeric(
            races_df["dist"].astype(str).str.extract(r"^(\d+)", expand=False),
            errors="coerce",
        )
        same_distance = races_df[form_yards == race_yards]
    return avg_place(same_distance)


def avg_place_at_track(races_df, line):
    # form file uses 3-letter track codes; same code is in source_href as track=YGL
    href = str(line.get("source_href", "") or "")
    m = re.search(r"[?&]track=([A-Za-z]+)", href)
    if not m:
        return float("nan")
    code = m.group(1).upper()
    same_track = races_df[races_df["track"].astype(str).str.strip().str.upper() == code]
    return avg_place(same_track)


def avg_est_time_at_distance(races_df, line):
    # yards from dist = digits only (letter at end ignored); must match line["race_distance"] in yards
    race_yards = pd.to_numeric(line["race_distance"], errors="coerce")
    if pd.isna(race_yards):
        return float("nan")
    if "_yards" in races_df.columns:
        same_distance = races_df[races_df["_yards"] == race_yards]
    else:
        form_yards = pd.to_numeric(
            races_df["dist"].astype(str).str.extract(r"^(\d+)", expand=False),
            errors="coerce",
        )
        same_distance = races_df[form_yards == race_yards]
    return avg_est_time(same_distance)


def last_k_races(races_df, k):
    # latest race first, then second-latest, ... — take the k most recent by date
    if len(races_df) == 0:
        return races_df.iloc[0:0]
    if "_fd" in races_df.columns:
        tmp = races_df[races_df["_fd"].notna()].sort_values("_fd", ascending=False, kind="mergesort")
        return tmp.head(k)
    d = pd.to_datetime(races_df["date"], format="%d-%b-%y", errors="coerce")
    tmp = races_df.assign(_d=d)
    tmp = tmp[tmp["_d"].notna()]
    tmp = tmp.sort_values("_d", ascending=False, kind="mergesort")
    return tmp.head(k).drop(columns=["_d"])


def avg_est_time_last_k_races(races_df, k):
    return avg_est_time(last_k_races(races_df, k))


def avg_sectional_time_last_k_races(races_df, k):
    return avg_sectional_time(last_k_races(races_df, k))


def avg_place_last_k_races(races_df, k):
    return avg_place(last_k_races(races_df, k))


def delta_est_time_last3_vs_last10(races_df):
    # last 10 avg minus last 3 avg: positive = faster recently (lower times in last 3) = improving
    return avg_est_time_last_k_races(races_df, 10) - avg_est_time_last_k_races(races_df, 3)


def delta_place_last3_vs_last10(races_df):
    # last 10 avg place minus last 3 avg: positive = better (lower) avg place in last 3 = improving
    return avg_place_last_k_races(races_df, 10) - avg_place_last_k_races(races_df, 3)


def bend_position_averages_last_k_races(races_df, k):
    return bend_position_averages(last_k_races(races_df, k))


def bend_position_delta_last3_vs_last10(races_df):
    # per bend: last10 avg minus last3 avg; positive = better (lower) bend positions in last 3
    last10 = bend_position_averages_last_k_races(races_df, 10)
    last3 = bend_position_averages_last_k_races(races_df, 3)
    return (
        last10[0] - last3[0],
        last10[1] - last3[1],
        last10[2] - last3[2],
        last10[3] - last3[3],
    )


def dog_age_months(line, race_dt, fi: FormIndex):
    # whelp_date from form data to this race’s meeting_date; returns age as a float (months)
    dog_name = str(line["greyhound"]).strip()
    whelp_dt = fi.whelp_first.get(dog_name)
    if whelp_dt is None or pd.isna(whelp_dt):
        return float("nan")
    if pd.isna(race_dt) or race_dt < whelp_dt:
        return float("nan")
    rd = relativedelta(race_dt, whelp_dt)
    return rd.years * 12 + rd.months + rd.days / 30.4375


def days_since_last_run(prior_all, race_dt):
    # days from most recent prior form date (any trial/race) to this race
    if len(prior_all) == 0:
        return float("nan")
    if "_fd" in prior_all.columns:
        last_run = prior_all["_fd"].max()
    else:
        last_run = pd.to_datetime(prior_all["date"], format="%d-%b-%y", errors="coerce").max()
    if pd.isna(last_run):
        return float("nan")
    return (race_dt - last_run).days


def current_race_trap(line):
    # trap box for this dog on this race row (from race_results)
    return pd.to_numeric(line["trap"], errors="coerce")


def parse_prize_euros(prize_str):
    if pd.isna(prize_str):
        return float("nan")
    s = str(prize_str).replace("€", "").replace(",", "").strip()
    return pd.to_numeric(s, errors="coerce")


_MONTH_ABBREV_TO_NUM = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def recover_fraction_from_excel_date_display(s: str) -> Optional[str]:
    """Excel reads UK-style odds a/b as a date (day a, month b); display looks like dd-Mmm.
    Recover the fraction: 08-Jan → 8/1, 05-Feb → 5/2, 11-Apr → 11/4."""
    s = s.strip().lstrip("'").strip()
    m = re.match(r"^0*(\d{1,2})-([A-Za-z]{3})\s*$", s)
    if not m:
        return None
    day = int(m.group(1))
    mon = m.group(2).lower()[:3]
    denom = _MONTH_ABBREV_TO_NUM.get(mon)
    if denom is None:
        return None
    return f"{day}/{denom}"


def sp_to_decimal(val):
    """UK fractional SP → decimal odds (profit/stake + 1). Uses the numeric fraction only: trailing
    letters after the denominator (e.g. 7/4f = 7/4 fav, 5/2cf) are ignored."""
    if val is None:
        return float("nan")
    try:
        if pd.isna(val):
            return float("nan")
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    s = s.lstrip("\t")
    if s.startswith("'"):
        s = s[1:].strip()
    # Excel-exported unicode fraction slash → ASCII for parsing
    s = s.replace("\u2044", "/")
    if not s or s.upper() in ("N/P", "N/M", "NAN", "<NA>"):
        return float("nan")
    # Evens = 1/1 → decimal 2.0
    if re.fullmatch(r"\s*(evs|evens?)\s*", s, re.IGNORECASE):
        return 2.0
    rec = recover_fraction_from_excel_date_display(s)
    if rec is not None:
        s = rec
    try:
        x = float(s.replace(",", ""))
        if x >= 1.01:
            return x
    except ValueError:
        pass
    # Leading num/num only — do not pick up stray digits elsewhere in the string
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b > 0:
            return a / b + 1.0
    return float("nan")


def sp_for_excel_csv(series: pd.Series) -> pd.Series:
    """Excel turns `5/2` into dates (e.g. 05-Feb). Use Unicode FRACTION SLASH ⁄ (U+2044) between
    stake/profit digits so the cell stays text, plus leading apostrophe. N/P unchanged."""
    def _one(v):
        if v is None:
            return v
        try:
            if pd.isna(v):
                return v
        except TypeError:
            pass
        t = str(v).strip()
        if not t or t.lower() in ("nan", "<na>"):
            return v
        if t.startswith("'"):
            t = t[1:]
        # 7/4f → 7⁄4f (only the slash between the two numbers)
        m = re.match(r"^(\d+)\s*/\s*(\d+)(.*)$", t)
        if m:
            t = m.group(1) + "\u2044" + m.group(2) + m.group(3)
        return "'" + t

    return series.map(_one)


def current_race_features(line):
    # current-race inputs only (no race_type, no per-runner grade — that is result-only). track_code from source_href when present.
    href = str(line.get("source_href", "") or "")
    m = re.search(r"[?&]track=([A-Za-z]+)", href)
    track_code = m.group(1).upper() if m else float("nan")
    return pd.Series(
        {
            "trap": current_race_trap(line),
            "race_number": pd.to_numeric(line.get("race_number"), errors="coerce"),
            "race_grade_ordinal": get_grade(line.get("race_grade")),
            "race_distance": pd.to_numeric(line.get("race_distance"), errors="coerce"),
            "race_surface": line.get("race_surface"),
            "meeting_venue": line.get("meeting_venue"),
            "meeting_date": line.get("meeting_date"),
            "wt": pd.to_numeric(line.get("wt"), errors="coerce"),
            "prize_eur": parse_prize_euros(line.get("prize")),
            "track_code": track_code,
        }
    )


def _race_id_series(race_df, race_id_col=None):
    r = race_df.reset_index(drop=True)
    if race_id_col is not None:
        s = r[race_id_col]
    elif "race_id" in r.columns:
        s = r["race_id"]
    else:
        s = (
            r["meeting_date"].astype(str)
            + "|"
            + r["meeting_venue"].astype(str)
            + "|"
            + r["race_number"].astype(str)
        )
    return s.astype(str).str.strip()


def features_for_row(line, race_dt, fi: FormIndex):
    prior_all = get_prior_dog_form(line, race_dt, fi)
    prior = keep_only_races(prior_all)
    bends = bend_position_averages(prior)
    bd3 = bend_position_averages_last_k_races(prior, 3)
    bd10 = bend_position_averages_last_k_races(prior, 10)
    bdel = bend_position_delta_last3_vs_last10(prior)
    cr = current_race_features(line)
    out = {
        "prior_avg_est_time": avg_est_time(prior),
        "prior_avg_sectional_time": avg_sectional_time(prior),
        "prior_avg_place": avg_place(prior),
        "prior_avg_place_at_distance": avg_place_at_distance(prior, line),
        "prior_avg_place_at_track": avg_place_at_track(prior, line),
        "prior_avg_est_time_at_distance": avg_est_time_at_distance(prior, line),
        "prior_est_time_last_3": avg_est_time_last_k_races(prior, 3),
        "prior_est_time_last_10": avg_est_time_last_k_races(prior, 10),
        "prior_delta_est_time_last3_vs_last10": delta_est_time_last3_vs_last10(prior),
        "prior_place_last_3": avg_place_last_k_races(prior, 3),
        "prior_place_last_10": avg_place_last_k_races(prior, 10),
        "prior_delta_place_last3_vs_last10": delta_place_last3_vs_last10(prior),
        "prior_grade_change_flag": grade_change_flag(line, prior),
        "prior_race_count": prior_race_count(prior),
        "career_win_rate": career_win_rate(prior),
        "prior_races_at_distance": prior_race_count_at_distance(prior, line),
        "win_rate_at_distance": win_rate_at_distance(prior, line),
        "win_rate_at_track": win_rate_at_track(prior, line),
        "prior_sectional_time_last_3": avg_sectional_time_last_k_races(prior, 3),
        "dog_age_months": dog_age_months(line, race_dt, fi),
        "days_since_last_run": days_since_last_run(prior_all, race_dt),
    }
    for j, v in enumerate(bends, 1):
        out[f"bend_avg_{j}_career"] = v
    for j, v in enumerate(bd3, 1):
        out[f"bend_avg_{j}_last3"] = v
    for j, v in enumerate(bd10, 1):
        out[f"bend_avg_{j}_last10"] = v
    for j, v in enumerate(bdel, 1):
        out[f"bend_delta_{j}_last3_vs_last10"] = v
    out.update(cr.to_dict())
    _sp = line.get("sp")
    if _sp is None or pd.isna(_sp):
        _sp = None
    out["sp"] = _sp
    out["sp_decimal"] = sp_to_decimal(_sp)
    # targets from this race result (for training — do not use as inputs when predicting pre-race)
    pos = pd.to_numeric(line.get("pos"), errors="coerce")
    out["target_pos"] = pos
    out["target_winner"] = float("nan") if pd.isna(pos) else (1.0 if float(pos) == 1.0 else 0.0)

    return out


def add_within_race_features(feat_df):
    # rank_* and diff_*_* are only within the same race_id (same race, all runners)
    out = feat_df.copy()
    out["race_id"] = out["race_id"].astype(str).str.strip()
    g = out.groupby("race_id", sort=False)
    # diff vs field best in this race: 0 = joint best; negative = worse (directional)
    # lower-is-better: diff = min_in_race - value; higher-is-better (win rates): diff = value - max_in_race

    lower_better = [
        "prior_avg_est_time",
        "prior_est_time_last_3",
        "prior_avg_est_time_at_distance",
        "prior_avg_place",
        "bend_avg_1_career",
        "bend_avg_2_career",
        "bend_avg_3_career",
        "bend_avg_4_career",
        "prior_avg_sectional_time",
        "prior_sectional_time_last_3",
    ]
    higher_better = ["win_rate_at_distance", "win_rate_at_track"]

    for col in lower_better:
        if col not in out.columns:
            continue
        # must rank within each race (groupby), not across the whole dataframe
        out[f"rank_{col}"] = g[col].rank(method="min", ascending=True, na_option="bottom")
        out[f"diff_{col}_vs_best"] = g[col].transform("min") - out[col]

    for col in higher_better:
        if col not in out.columns:
            continue
        out[f"rank_{col}"] = g[col].rank(method="min", ascending=False, na_option="bottom")
        out[f"diff_{col}_vs_best"] = out[col] - g[col].transform("max")

    return out


def build_feature_dataset(
    race_df,
    forms_df,
    race_id_col=None,
    max_workers=None,
    form_index: Optional[FormIndex] = None,
):
    # one row per dog per race; threads share FormIndex (dict of per-dog slices — no full-table scan per row)
    rid = _race_id_series(race_df, race_id_col)
    n = len(race_df)
    if n == 0:
        return pd.DataFrame()

    if max_workers is None:
        max_workers = min(32, max(4, (os.cpu_count() or 4) * 2))

    fi = form_index if form_index is not None else get_form_index(forms_df)
    race_dt = pd.to_datetime(race_df["meeting_date"], format="%d-%b-%y")

    def _row_features(i):
        line = race_df.iloc[i]
        d = features_for_row(line, race_dt.iloc[i], fi)
        d["race_id"] = str(rid.iloc[i]).strip()
        d["greyhound"] = line["greyhound"]
        return d

    chunksize = max(1, n // (max_workers * 8))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        rows = list(pool.map(_row_features, range(n), chunksize=chunksize))

    feat = pd.DataFrame(rows)
    return add_within_race_features(feat)


# Lazy index: building on import would slow down `import main` (large groupby). Call get_form_index().
_FORM_INDEX: Optional[FormIndex] = None
_FORM_INDEX_FORMS_ID: Optional[int] = None


def get_form_index(forms: Optional[pd.DataFrame] = None) -> FormIndex:
    """Return a cached FormIndex for `forms` (default: module-level forms_df)."""
    global _FORM_INDEX, _FORM_INDEX_FORMS_ID
    target = forms_df if forms is None else forms
    tid = id(target)
    if _FORM_INDEX is not None and _FORM_INDEX_FORMS_ID == tid:
        return _FORM_INDEX
    _FORM_INDEX = build_form_index(target)
    _FORM_INDEX_FORMS_ID = tid
    return _FORM_INDEX


if __name__ == "__main__":
    _t0 = time.perf_counter()
    _start_wall = datetime.now()
    print(f"Started at {_start_wall.isoformat(timespec='seconds')}")

    # drop .head() for full build
    feature_df = build_feature_dataset(df, forms_df, form_index=get_form_index())
    _export = feature_df.copy()
    if "sp" in _export.columns:
        _export["sp"] = sp_for_excel_csv(_export["sp"])
    _export.to_csv("feature_dataset_first18.csv", index=False, encoding="utf-8-sig")

    _t1 = time.perf_counter()
    _end_wall = datetime.now()
    print(feature_df.shape)
    print("wrote feature_dataset_first18.csv")
    if "sp" in feature_df.columns:
        s = feature_df["sp"].astype("string")
        n_sp = int((s.notna() & (s.str.strip() != "")).sum())
        print(
            f"SP column: {n_sp} / {len(feature_df)} rows have a value in race_results "
            f"(leading rows are often <NA> in the source CSV; search the file for e.g. 7/4 or N/P)."
        )
    print(f"Finished at {_end_wall.isoformat(timespec='seconds')}")
    print(f"Elapsed: {_t1 - _t0:.3f}s")
