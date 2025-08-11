import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

RULES = {
    "open_time": "10:00",
    "close_time": "03:00",
    "min_gap": 15,  # minutes
    "golden_start": "18:00",
    "golden_end": "21:00",  # inclusive
    "version_coeff": {"2D": 1.0, "3D": 1.1, "IMAX": 1.15},
    "version_total_caps": {"3D": 1200, "IMAX": 1500},  # minutes
    "genre_caps": {"Animation": (1, 5), "Horror": (0, 3), "Action": (2, 6), "Drama": (1, 6)},
    "genre_time_limits": {
        # genre: (earliest_start_inclusive or None, latest_start_inclusive or None)
        "Animation": (None, "19:00"),
        "Family": (None, "19:00"),
        "Horror": ("21:00", None),
        "Thriller": ("21:00", None),
    },
}

def ceil_to_30(runtime):
    return int(((runtime + 29) // 30) * 30)

def parse_hhmm(hhmm):
    return datetime.strptime(hhmm, "%H:%M")

def to_minutes(td):
    return td.days * 24 * 60 + td.seconds // 60

def in_quarter(hhmm):
    m = int(hhmm.split(":")[1])
    return m in (0, 15, 30, 45)

def check_schedule(schedule_csv, movies_csv):
    df_s = pd.read_csv(schedule_csv)
    df_m = pd.read_csv(movies_csv)

    # map movie info
    m = df_m.set_index("id").to_dict(orient="index")
    for k,v in m.items():
        v["rounded_runtime"] = ceil_to_30(v["runtime"])
        v["genres_list"] = [g.strip() for g in str(v["genres"]).split(",")]

    # helpers
    def show_end(dt_start, mid):
        return dt_start + timedelta(minutes=m[mid]["rounded_runtime"])

    # Accumulators
    issues = []
    version_minutes = defaultdict(int)
    genre_counts = defaultdict(int)

    # Time window constants
    open_dt = parse_hhmm(RULES["open_time"])
    close_dt = parse_hhmm(RULES["close_time"]) + timedelta(days=1)  # next day 03:00
    golden_start = parse_hhmm(RULES["golden_start"])
    golden_end = parse_hhmm(RULES["golden_end"])

    # Check per-room constraints
    for room, group in df_s.groupby("room"):
        # sort by start
        shows = []
        for _, r in group.iterrows():
            t = r["showtime"]
            if t < RULES["open_time"]:
                dt = parse_hhmm(t) + timedelta(days=1)  # after midnight
            else:
                dt = parse_hhmm(t)
            shows.append((dt, r))
        shows.sort(key=lambda x: x[0])

        # quarter check + end before 03:00
        for dt, r in shows:
            if not in_quarter(r["showtime"]):
                issues.append(f"{room} {r['showtime']} not on quarter")
            end = show_end(dt, r["id"])
            # must end strictly before 03:00
            if not (end < close_dt):
                issues.append(f"{room} {r['showtime']} id={r['id']} ends at {end.strftime('%H:%M')} not before 03:00")

        # gap >= 15min
        for i in range(1, len(shows)):
            prev_end = show_end(shows[i-1][0], shows[i-1][1]["id"])
            gap = to_minutes(shows[i][0] - prev_end)
            if gap < RULES["min_gap"]:
                issues.append(f"{room} gap {shows[i-1][1]['showtime']}→{shows[i][1]['showtime']} = {gap}min < 15")

        # 9h rolling window ≤ 7h playtime
        # discretize by checking windows anchored at each show start
        for i in range(len(shows)):
            window_start = shows[i][0]
            window_end = window_start + timedelta(hours=9)
            play = 0
            for dt, r in shows:
                st = dt
                ed = show_end(dt, r["id"])
                overlap = max(timedelta(0), min(ed, window_end) - max(st, window_start))
                play += to_minutes(overlap)
            if play > 420:
                issues.append(f"{room} 9h-window starting {window_start.strftime('%H:%M')} has {play}min > 420")

    # version total caps + genre counts + time limits
    for _, r in df_s.iterrows():
        mid = r["id"]
        ver = r["version"]
        rt = m[mid]["rounded_runtime"]
        if ver in ("3D", "IMAX"):
            version_minutes[ver] += rt

        # genre counts
        for g in m[mid]["genres_list"]:
            genre_counts[g] += 1

        # time limits by genre
        st = r["showtime"]
        dt = parse_hhmm(st) + (timedelta(days=1) if st < RULES["open_time"] else timedelta(0))
        for g, (earliest, latest) in RULES["genre_time_limits"].items():
            if g in m[mid]["genres_list"]:
                if earliest and dt < parse_hhmm(earliest):
                    issues.append(f"Time limit: {g} at {st} earlier than {earliest} (id={mid})")
                if latest and dt > parse_hhmm(latest):
                    issues.append(f"Time limit: {g} at {st} later than {latest} (id={mid})")

    # version caps
    for v, cap in RULES["version_total_caps"].items():
        if version_minutes[v] > cap:
            issues.append(f"Version cap: {v} {version_minutes[v]}min > {cap}min")

    # genre caps
    for g, (lo, hi) in RULES["genre_caps"].items():
        cnt = genre_counts.get(g, 0)
        if cnt < lo:
            issues.append(f"Genre cap: {g} count {cnt} < min {lo}")
        if cnt > hi:
            issues.append(f"Genre cap: {g} count {cnt} > max {hi}")

    print("Issues:")
    for it in issues:
        print(" -", it)

    print("\nVersion minutes:", dict(version_minutes))
    print("Genre counts (top):", dict(sorted(genre_counts.items(), key=lambda x: -x[1])[:10]))

if __name__ == "__main__":
    check_schedule(
        "Q3/df_result_2_copt_ours_new.csv",
        "Q3/df_movies_schedule_ours_new.csv",
    )