#!/usr/bin/env python3
"""
run_7day_pipeline.py

Fuse dynamic parameter updater + CBC MILP optimizer and run for each day in sample_simulated_table.csv.

Usage: python run_7day_pipeline.py
"""

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from pulp import PULP_CBC_CMD, LpProblem, LpVariable, LpMaximize, lpSum, value
import math
import warnings
warnings.filterwarnings("ignore")

# ========== CONFIG - edit paths if needed ==========
CONFIG = {
    "movies_csv": r"C:\Users\47797\Desktop\predicted_movies.csv",
    "observations_csv": r"C:\Users\47797\Desktop\sample_simulated_table.csv",
    "cinema_csv": r"C:\Users\47797\Desktop\df_cinema.csv",
    "movies_schedule_csv": r"C:\Users\47797\Desktop\df_movies_schedule.csv",
    "dynamic_params_path": "dynamic_params.json",
    "results_dir": "results"
}
# ===================================================

os.makedirs(CONFIG["results_dir"], exist_ok=True)

# ---------------- DynamicAttendanceModel (unchanged logic) ----------------
class DynamicAttendanceModel:
    def __init__(self, movies_df):
        self.movies_df = movies_df.copy()
        # preserve original ratings (base)
        self.original_ratings = dict(zip(self.movies_df["id"], self.movies_df["rating"]))

        # init params
        self.alpha_t_w = defaultdict(lambda: 1.0)
        self.phi_i = {mid: 1.0 for mid in self.movies_df["id"]}
        self.lambda_decay = 0.05

        # history for optional plotting
        self.history_alpha = []
        self.history_phi = []
        self.history_lambda = []

    def update_parameters(self, obs_df):
        # obs_df expected columns:
        # movie_id, day, time_slot, weekday_flag, baseline_attendance, simulated_attendance
        for _, row in obs_df.iterrows():
            mid = row["movie_id"]
            t = int(row["time_slot"])
            w = int(row["weekday_flag"])
            day_since_release = int(row["day"])
            baseline_att = row["baseline_attendance"]
            simulated_att = row["simulated_attendance"]

            # update alpha
            key = (t, w)
            if baseline_att > 0:
                ratio = simulated_att / baseline_att
                # EWMA-like with alpha=0.1 smoothing (can be adjusted)
                self.alpha_t_w[key] = self.alpha_t_w[key] * 0.9 + 0.1 * ratio

            # update phi
            if baseline_att > 0:
                phi_update = simulated_att / baseline_att
                self.phi_i[mid] = self.phi_i[mid] * 0.9 + 0.1 * phi_update

            # update lambda
            if simulated_att > 0 and baseline_att > 0:
                decay_update = -np.log(simulated_att / baseline_att) / max(day_since_release, 1)
                self.lambda_decay = self.lambda_decay * 0.9 + 0.1 * decay_update

        # save historical values
        self.history_alpha.append(dict(self.alpha_t_w))
        self.history_phi.append(dict(self.phi_i))
        self.history_lambda.append(self.lambda_decay)

    def save_parameters(self, path="dynamic_params.json"):
        # produce updated_ratings by applying phi multiplicative adjustments to base rating
        updated_ratings = {}
        for mid, phi_val in self.phi_i.items():
            base_rating = self.original_ratings.get(mid, 0.0)
            updated_ratings[str(mid)] = base_rating * phi_val

        params = {
            "alpha_t_w": {f"{t}_{w}": float(val) for (t, w), val in self.alpha_t_w.items()},
            "phi_i": {str(mid): float(val) for mid, val in self.phi_i.items()},
            "lambda_decay": float(self.lambda_decay),
            "updated_ratings": updated_ratings
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=4, ensure_ascii=False)
        print(f"[Dynamic] Parameters saved to {path}")

    def visualize_trends(self, save_path="params_trend.png"):
        # optional: minimal trend visualization (only first 4 alpha and 4 phi series to avoid clutter)
        days = list(range(1, len(self.history_lambda) + 1))
        fig, axes = plt.subplots(3, 1, figsize=(9, 10))

        if self.history_alpha:
            keys = list(self.history_alpha[-1].keys())[:4]
            for key in keys:
                vals = [h.get(key, np.nan) for h in self.history_alpha]
                axes[0].plot(days, vals, label=f"alpha_{key}")
            axes[0].legend()
            axes[0].set_title("alpha_t_w")

        if self.history_phi:
            keys = list(self.history_phi[-1].keys())[:4]
            for mid in keys:
                vals = [h.get(mid, np.nan) for h in self.history_phi]
                axes[1].plot(days, vals, label=f"phi_{mid}")
            axes[1].legend()
            axes[1].set_title("phi_i (sample)")

        axes[2].plot(days, self.history_lambda, marker='o')
        axes[2].set_title("lambda_decay")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"[Dynamic] Trend plot saved to {save_path}")

# ---------------- CinemaSchedulingOptimizer (CBC version, keep core logic) ----------------
class CinemaSchedulingOptimizer:
    def __init__(self, cinema_file, movies_file, dynamic_params_path=None):
        self.cinema_df = pd.read_csv(cinema_file)
        self.movies_df = pd.read_csv(movies_file)

        # If dynamic params JSON exists, inject updated ratings
        if dynamic_params_path and os.path.exists(dynamic_params_path):
            with open(dynamic_params_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            if "updated_ratings" in params:
                for mid_str, new_rating in params["updated_ratings"].items():
                    try:
                        mid = int(mid_str)
                        # ensure id column types align
                        if mid in self.movies_df['id'].values:
                            self.movies_df.loc[self.movies_df['id'] == mid, 'rating'] = float(new_rating)
                    except Exception:
                        pass

        # time settings and other constants (kept identical to your CBC code)
        self.start_hour = 10
        self.end_hour = 27
        self.time_slots = self._generate_time_slots()

        self.version_coeff = {'2D': 1.0, '3D': 1.1, 'IMAX': 1.15}
        self.basic_cost = 2.42
        self.fixed_cost = 90
        self.version_limits = {'3D': {'min': 0, 'max': 1200}, 'IMAX': {'min': 0, 'max': 1500}}
        self.genre_limits = {'Animation': {'min': 1, 'max': 5}, 'Horror': {'min':0, 'max':3},
                             'Action': {'min':2, 'max':6}, 'Drama': {'min':1, 'max':6}}
        self.genre_time_limits = {'Animation': {'latest_start': 19}, 'Family': {'latest_start': 19},
                                  'Horror': {'earliest_start': 21}, 'Thriller': {'earliest_start': 21}}
        self.prime_time_start = 18
        self.prime_time_end = 21
        self.prime_time_multiplier = 1.3

    def _generate_time_slots(self):
        slots = []
        current_hour = self.start_hour
        current_minute = 0
        while current_hour < self.end_hour:
            slots.append(f"{current_hour:02d}:{current_minute:02d}")
            current_minute += 15
            if current_minute >= 60:
                current_minute = 0
                current_hour += 1
        return slots

    def _get_versions(self, movie_id):
        movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
        versions = movie['version'].split(',')
        return [v.strip() for v in versions]

    def _round_up_to_30(self, runtime):
        return math.ceil(runtime / 30) * 30

    def _can_room_play_version(self, room, version):
        room_info = self.cinema_df[self.cinema_df['room'] == room].iloc[0]
        return bool(room_info.get(version, False))

    def _calculate_ticket_price(self, movie_id, version, is_prime_time=False):
        movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
        basic_price = movie['basic_price']
        if version == '2D':
            price = basic_price
        elif version == '3D':
            price = basic_price * 1.2
        elif version == 'IMAX':
            price = basic_price * 1.23
        if is_prime_time:
            price *= self.prime_time_multiplier
        return price

    def _calculate_attendance(self, capacity, rating):
        return math.floor(capacity * rating / 10)

    def _calculate_cost(self, capacity, version):
        version_coeff = self.version_coeff[version]
        return version_coeff * capacity * self.basic_cost + self.fixed_cost

    def _get_sharing_rate(self, movie_id):
        movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
        orig_lang = movie['original_language']
        if 'Mandarin' in orig_lang:
            return 0.43
        else:
            return 0.51

    def _is_prime_time(self, time_slot):
        hour = int(time_slot.split(':')[0])
        return self.prime_time_start <= hour < self.prime_time_end

    def _check_genre_time_constraint(self, movie_id, time_slot):
        movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
        genres = [g.strip() for g in movie['genres'].split(',')]
        hour = int(time_slot.split(':')[0])
        if hour < 24:
            pass
        else:
            hour - 24
        for genre in genres:
            if genre in self.genre_time_limits:
                constraints = self.genre_time_limits[genre]
                if 'earliest_start' in constraints:
                    earliest = constraints['earliest_start']
                    if hour < earliest and hour >= 10:
                        return False
                if 'latest_start' in constraints:
                    latest = constraints['latest_start']
                    if hour >= 24:
                        return False
                    elif hour >= latest:
                        return False
        return True

    def _time_slot_to_minutes(self, time_slot):
        hour, minute = map(int, time_slot.split(':'))
        return (hour - self.start_hour) * 60 + minute

    def optimize_schedule(self):
        prob = LpProblem("Cinema_Scheduling", LpMaximize)
        x = {}

        # create binary variables for feasible combinations
        for _, room_info in self.cinema_df.iterrows():
            room = room_info['room']
            x[room] = {}
            for _, movie in self.movies_df.iterrows():
                movie_id = movie['id']
                versions = self._get_versions(movie_id)
                x[room][movie_id] = {}
                for version in versions:
                    if not self._can_room_play_version(room, version):
                        continue
                    x[room][movie_id][version] = {}
                    for time_slot in self.time_slots:
                        if not self._check_genre_time_constraint(movie_id, time_slot):
                            continue
                        runtime = self._round_up_to_30(movie['runtime'])
                        start_minutes = self._time_slot_to_minutes(time_slot)
                        end_minutes = start_minutes + runtime
                        if end_minutes <= (self.end_hour - self.start_hour) * 60:
                            var_name = f"x_{room}_{movie_id}_{version}_{time_slot}"
                            x[room][movie_id][version][time_slot] = LpVariable(var_name, cat='Binary')

        # objective
        revenue_terms = []
        cost_terms = []
        for room in x:
            room_capacity = int(self.cinema_df[self.cinema_df['room'] == room]['capacity'].iloc[0])
            for movie_id in x[room]:
                movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                rating = float(movie['rating'])
                sharing_rate = self._get_sharing_rate(movie_id)
                for version in x[room][movie_id]:
                    for time_slot in x[room][movie_id][version]:
                        var = x[room][movie_id][version][time_slot]
                        is_prime = self._is_prime_time(time_slot)
                        ticket_price = self._calculate_ticket_price(movie_id, version, is_prime)
                        attendance = self._calculate_attendance(room_capacity, rating)
                        ticket_revenue = ticket_price * attendance
                        net_revenue = ticket_revenue * (1 - sharing_rate)
                        cost = self._calculate_cost(room_capacity, version)
                        revenue_terms.append(net_revenue * var)
                        cost_terms.append(cost * var)
        prob += lpSum(revenue_terms) - lpSum(cost_terms)

        # constraints (kept same as original)
        # (1) each room/time only one movie (consider overlaps)
        for room in x:
            for time_slot in self.time_slots:
                overlapping_vars = []
                for movie_id in x[room]:
                    movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                    runtime = self._round_up_to_30(movie['runtime'])
                    for version in x[room][movie_id]:
                        for start_time in x[room][movie_id][version]:
                            start_minutes = self._time_slot_to_minutes(start_time)
                            end_minutes = start_minutes + runtime
                            current_minutes = self._time_slot_to_minutes(time_slot)
                            if start_minutes <= current_minutes < end_minutes + 15:
                                overlapping_vars.append(x[room][movie_id][version][start_time])
                if overlapping_vars:
                    prob += lpSum(overlapping_vars) <= 1

        # (2) version total duration limits
        for version in ['3D', 'IMAX']:
            total_duration = []
            for room in x:
                for movie_id in x[room]:
                    if version in x[room][movie_id]:
                        movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                        runtime = self._round_up_to_30(movie['runtime'])
                        for time_slot in x[room][movie_id][version]:
                            total_duration.append(runtime * x[room][movie_id][version][time_slot])
            if total_duration:
                prob += lpSum(total_duration) <= self.version_limits[version]['max']
                prob += lpSum(total_duration) >= self.version_limits[version]['min']

        # (3) genre show count limits
        for genre, limits in self.genre_limits.items():
            genre_shows = []
            for room in x:
                for movie_id in x[room]:
                    movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                    movie_genres = [g.strip() for g in movie['genres'].split(',')]
                    if genre in movie_genres:
                        for version in x[room][movie_id]:
                            for time_slot in x[room][movie_id][version]:
                                genre_shows.append(x[room][movie_id][version][time_slot])
            if genre_shows:
                if 'max' in limits:
                    prob += lpSum(genre_shows) <= limits['max']
                if 'min' in limits:
                    prob += lpSum(genre_shows) >= limits['min']

        # (4) per-movie show count limit (<=3)
        for movie_id in self.movies_df['id']:
            movie_shows = []
            for room in x:
                if movie_id in x[room]:
                    for version in x[room][movie_id]:
                        for time_slot in x[room][movie_id][version]:
                            movie_shows.append(x[room][movie_id][version][time_slot])
            if movie_shows:
                prob += lpSum(movie_shows) <= 3

        # (5) equipment continuous runtime window (9h window -> max 420 min)
        for room in x:
            for start_hour in range(0, (self.end_hour - self.start_hour) - 8):
                window_duration = []
                for movie_id in x[room]:
                    movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                    runtime = self._round_up_to_30(movie['runtime'])
                    for version in x[room][movie_id]:
                        for time_slot in x[room][movie_id][version]:
                            slot_hour = self._time_slot_to_minutes(time_slot) // 60
                            if start_hour <= slot_hour < start_hour + 9:
                                window_duration.append(runtime * x[room][movie_id][version][time_slot])
                if window_duration:
                    prob += lpSum(window_duration) <= 420

        # solve with CBC (silently)
        prob.solve(PULP_CBC_CMD(msg=0))
        schedule_results = []
        if prob.status == 1:
            for room in x:
                room_capacity = int(self.cinema_df[self.cinema_df['room'] == room]['capacity'].iloc[0])
                for movie_id in x[room]:
                    movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                    for version in x[room][movie_id]:
                        for time_slot in x[room][movie_id][version]:
                            if x[room][movie_id][version][time_slot].varValue == 1:
                                attendance = self._calculate_attendance(room_capacity, movie['rating'])
                                schedule_results.append({
                                    'room': room,
                                    'showtime': time_slot,
                                    'id': movie_id,
                                    'version': version,
                                    'attendance': attendance
                                })
        schedule_results.sort(key=lambda x: (x['room'], x['showtime']))
        return schedule_results, prob.status, value(prob.objective) if prob.status == 1 else None

# ---------------- Orchestration (run 7-day loop) ----------------
def run_pipeline(config):
    movies_df = pd.read_csv(config["movies_csv"])
    obs_df = pd.read_csv(config["observations_csv"])
    # validate expected columns
    required_obs_cols = {"movie_id","day","time_slot","weekday_flag","baseline_attendance","simulated_attendance"}
    if not required_obs_cols.issubset(set(obs_df.columns)):
        raise ValueError(f"observations csv missing columns. need {required_obs_cols}")

    days = sorted(obs_df["day"].unique())
    print(f"Found days: {days}")

    # init dynamic model with base movies
    dyn_model = DynamicAttendanceModel(movies_df)

    for day in days:
        print("\n" + "="*60)
        print(f"[Pipeline] Day {day} start")
        day_df = obs_df[obs_df["day"] == day]

        # update dynamic params using today's observations (cold-start or streaming)
        dyn_model.update_parameters(day_df)
        # save dynamic params (this writes updated_ratings used by MILP)
        dyn_model.save_parameters(config["dynamic_params_path"])

        # optional: save or plot trends per day (commented out to reduce IO)
        # dyn_model.visualize_trends(save_path=os.path.join(config["results_dir"], f"params_trend_day{day}.png"))

        # instantiate optimizer which will read dynamic_params.json and update ratings
        optimizer = CinemaSchedulingOptimizer(config["cinema_csv"], config["movies_schedule_csv"],
                                              dynamic_params_path=config["dynamic_params_path"])

        # run optimizer (CBC)
        schedule, status, obj = optimizer.optimize_schedule()
        print(f"[Pipeline] Day {day} solver status: {status}, objective: {obj}")

        # save schedule
        out_path = os.path.join(config["results_dir"], f"day_{int(day)}.csv")
        pd.DataFrame(schedule).to_csv(out_path, index=False, encoding="utf-8")
        print(f"[Pipeline] Day {day} result saved to {out_path}")

    # end loop
    print("\n" + "="*60)
    print("[Pipeline] All days processed.")

if __name__ == "__main__":
    run_pipeline(CONFIG)
