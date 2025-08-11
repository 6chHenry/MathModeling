#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
融合版：每日动态参数更新 -> 生成 updated_ratings -> 调用 COPT MILP 求解
保存每天的排片结果到 output_results/day_{d}_schedule.csv

使用说明：
1. 修改下面的路径：PREDICTED_MOVIES_CSV, OBS_CSV, CINEMA_CSV, MOVIES_SCHEDULE_CSV
2. python run_dynamic_then_milp_copt.py
"""

import os
import json
import math
import time
import pandas as pd
import numpy as np
from collections import defaultdict

# --------------------------
# 配置路径（请按实际路径修改）
# --------------------------
PREDICTED_MOVIES_CSV = r"C:\Users\47797\Desktop\predicted_movies.csv"
OBS_CSV = r"C:\Users\47797\Desktop\sample_simulated_table.csv"
CINEMA_CSV = r"C:\Users\47797\Desktop\df_cinema.csv"
MOVIES_SCHEDULE_CSV = r"C:\Users\47797\Desktop\df_movies_schedule.csv"
DYNAMIC_PARAMS_PATH = "dynamic_params.json"
OUTPUT_DIR = r"C:\Users\47797\Desktop\output_results"   # 会自动创建

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 动态参数更新模块
# =========================
class DynamicAttendanceModel:
    """
    动态参数更新 （EWMA）
    会维护 alpha_t_w, phi_i, lambda_decay，并在 save_parameters 时输出 updated_ratings
    """
    def __init__(self, movies_df,
                 rho_alpha=0.30, rho_phi=0.30, rho_lambda=0.20):
        self.movies_df = movies_df.copy()
        # 保存原始 rating，用于生成 updated_ratings
        self.original_ratings = dict(zip(self.movies_df["id"], self.movies_df["rating"]))

        # 初始化参数（若需要可按表格设置初始 alpha）
        # 默认 key 为 (time_slot_index, weekday_flag)
        self.alpha_t_w = defaultdict(lambda: 1.0)
        # 你曾给出的假设表（可覆盖 alpha 初值）
        # 映射 time_slot index -> period grouping: 0 morning,1 afternoon,2 evening
        # 下面示例为方便起见，如果你的时间槽不是 0/1/2，请在使用前做映射或在输入 obs 中直接用 0/1/2。
        self.alpha_t_w[(0,0)] = 0.30
        self.alpha_t_w[(0,1)] = 0.50
        self.alpha_t_w[(1,0)] = 0.50
        self.alpha_t_w[(1,1)] = 0.75
        self.alpha_t_w[(2,0)] = 0.80
        self.alpha_t_w[(2,1)] = 0.95

        self.phi_i = {mid: 1.0 for mid in self.movies_df["id"]}
        self.lambda_decay = 0.05

        # EWMA smoothing factors
        self.rho_alpha = rho_alpha
        self.rho_phi = rho_phi
        self.rho_lambda = rho_lambda

        # history（仅在内存中，非必须）
        self.history_alpha = []
        self.history_phi = []
        self.history_lambda = []

    def update_parameters(self, obs_df):
        """
        使用当天 obs（含列 movie_id, day, time_slot, weekday_flag, baseline_attendance, simulated_attendance）
        逐条更新 alpha, phi, lambda（按 EWMA）
        """
        for _, row in obs_df.iterrows():
            mid = row["movie_id"]
            t = int(row["time_slot"])
            w = int(row["weekday_flag"])
            day_since_release = int(row["day"])
            baseline_att = row["baseline_attendance"]
            simulated_att = row["simulated_attendance"]

            # alpha_{t,w}
            key = (t, w)
            if baseline_att > 0:
                r_obs = simulated_att  # observed occupancy rate (in your data it's already ratio)
                prev_alpha = self.alpha_t_w.get(key, 1.0)
                self.alpha_t_w[key] = self.rho_alpha * float(r_obs) + (1 - self.rho_alpha) * prev_alpha

            # phi_i
            if baseline_att > 0:
                predicted = baseline_att
                ratio = simulated_att / predicted if predicted > 0 else 1.0
                prev_phi = self.phi_i.get(mid, 1.0)
                self.phi_i[mid] = self.rho_phi * ratio + (1 - self.rho_phi) * prev_phi

            # lambda
            if simulated_att > 0 and baseline_att > 0:
                # instant estimate of decay: -log(sim / base) / day
                # safeguard day_since_release >=1
                d = max(1, day_since_release)
                try:
                    est = -np.log(simulated_att / baseline_att) / d
                    prev_l = self.lambda_decay
                    self.lambda_decay = self.rho_lambda * est + (1 - self.rho_lambda) * prev_l
                except:
                    pass

        # save history snapshot
        self.history_alpha.append(dict(self.alpha_t_w))
        self.history_phi.append(dict(self.phi_i))
        self.history_lambda.append(self.lambda_decay)

    def save_parameters(self, path=DYNAMIC_PARAMS_PATH):
        """
        存储 alpha, phi, lambda，并根据 phi_i 修正原始 rating 生成 updated_ratings，
        供 MILP 程序读取（MILP 只读取 updated_ratings 并替换 movies_df 中的 rating）
        """
        updated_ratings = {}
        for mid, phi_val in self.phi_i.items():
            base_rating = self.original_ratings.get(mid, 0.0)
            updated_ratings[str(mid)] = float(base_rating * phi_val)

        params = {
            "alpha_t_w": {f"{t}_{w}": float(v) for (t,w), v in self.alpha_t_w.items()},
            "phi_i": {str(mid): float(v) for mid, v in self.phi_i.items()},
            "lambda_decay": float(self.lambda_decay),
            "updated_ratings": updated_ratings
        }
        with open(path, "w") as f:
            json.dump(params, f, indent=4)
        print(f"[Dynamic] saved dynamic params -> {path}")

# =========================
# COPT MILP 模块
# =========================
import coptpy as cp
from coptpy import COPT

class CinemaSchedulingOptimizerCOPT:
    def __init__(self, cinema_file, movies_file):
        self.cinema_df = pd.read_csv(cinema_file)
        self.movies_df = pd.read_csv(movies_file)

        # Load dynamic_params.json to update ratings (this is how we pass updated ratings)
        params_path = DYNAMIC_PARAMS_PATH
        try:
            if os.path.exists(params_path):
                with open(params_path, "r") as f_json:
                    params = json.load(f_json)
                if "updated_ratings" in params:
                    for mid, new_rating in params["updated_ratings"].items():
                        try:
                            mid_int = int(mid)
                            if mid_int in self.movies_df['id'].values:
                                self.movies_df.loc[self.movies_df['id'] == mid_int, 'rating'] = float(new_rating)
                        except ValueError:
                            pass
        except Exception as e:
            print("[COPT] warning: cannot load dynamic params:", e)

        # rest of original initialization
        self.start_hour = 10
        self.end_hour = 27
        self.time_slots = self._generate_time_slots()
        self.version_coeff = {'2D': 1.0, '3D': 1.1, 'IMAX': 1.15}
        self.basic_cost = 2.42
        self.fixed_cost = 90
        self.version_limits = {'3D': {'min': 0, 'max': 1200}, 'IMAX': {'min': 0, 'max': 1500}}
        self.genre_limits = {'Animation': {'min': 1, 'max': 5},
                             'Horror': {'min':0, 'max': 3},
                             'Action': {'min': 2, 'max': 6},
                             'Drama': {'min': 1, 'max': 6}}
        self.genre_time_limits = {'Animation': {'latest_start': 19},
                                  'Family': {'latest_start': 19},
                                  'Horror': {'earliest_start': 21},
                                  'Thriller': {'earliest_start': 21}}
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

    def _convert_to_display_time(self, time_slot):
        hour, minute = map(int, time_slot.split(':'))
        if hour >= 24:
            display_hour = hour - 24
            return f"{display_hour:02d}:{minute:02d}"
        else:
            return time_slot

    def _get_versions(self, movie_id):
        movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
        versions = movie['version'].split(',')
        return [v.strip() for v in versions]

    def _round_up_to_30(self, runtime):
        return math.ceil(runtime / 30) * 30

    def _can_room_play_version(self, room, version):
        room_info = self.cinema_df[self.cinema_df['room'] == room].iloc[0]
        return bool(room_info[version])

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
        original_language = movie['original_language']
        if 'Mandarin' in original_language:
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
            display_hour = hour
        else:
            display_hour = hour - 24
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
        print("[COPT] using COPT to optimize...")
        env = cp.Envr()
        model = env.createModel("Cinema_Scheduling")

        x = {}
        var_list = []

        # create vars
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
                            var = model.addVar(vtype=COPT.BINARY, name=var_name)
                            x[room][movie_id][version][time_slot] = var
                            var_list.append((room, movie_id, version, time_slot, var))

        # objective
        obj_expr = 0
        for room, movie_id, version, time_slot, var in var_list:
            room_capacity = self.cinema_df[self.cinema_df['room'] == room]['capacity'].iloc[0]
            movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
            rating = movie['rating']
            sharing_rate = self._get_sharing_rate(movie_id)
            is_prime = self._is_prime_time(time_slot)
            ticket_price = self._calculate_ticket_price(movie_id, version, is_prime)
            attendance = self._calculate_attendance(room_capacity, rating)
            ticket_revenue = ticket_price * attendance
            net_revenue = ticket_revenue * (1 - sharing_rate)
            cost = self._calculate_cost(room_capacity, version)
            net_profit = net_revenue - cost
            obj_expr += net_profit * var

        model.setObjective(obj_expr, COPT.MAXIMIZE)

        # constraints (as original)
        constraint_count = 0
        # (1) time conflicts
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
                    model.addConstr(cp.quicksum(overlapping_vars) <= 1,
                                    name=f"time_conflict_{room}_{time_slot}")
                    constraint_count += 1

        # (2) version duration limits
        for version in ['3D', 'IMAX']:
            total_duration_vars = []
            for room, movie_id, ver, time_slot, var in var_list:
                if ver == version:
                    movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                    runtime = self._round_up_to_30(movie['runtime'])
                    total_duration_vars.append(runtime * var)
            if total_duration_vars:
                model.addConstr(cp.quicksum(total_duration_vars) <= self.version_limits[version]['max'],
                                name=f"{version}_max_duration")
                model.addConstr(cp.quicksum(total_duration_vars) >= self.version_limits[version]['min'],
                                name=f"{version}_min_duration")
                constraint_count += 2

        # (3) genre counts
        for genre, limits in self.genre_limits.items():
            genre_vars = []
            for room, movie_id, version, time_slot, var in var_list:
                movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                movie_genres = [g.strip() for g in movie['genres'].split(',')]
                if genre in movie_genres:
                    genre_vars.append(var)
            if genre_vars:
                if 'max' in limits:
                    model.addConstr(cp.quicksum(genre_vars) <= limits['max'],
                                    name=f"{genre}_max_shows")
                    constraint_count += 1
                if 'min' in limits:
                    model.addConstr(cp.quicksum(genre_vars) >= limits['min'],
                                    name=f"{genre}_min_shows")
                    constraint_count += 1

        # (4) equipment continuous runtime windows
        for room in x:
            for window_start_minutes in range(0, (self.end_hour - self.start_hour) * 60 - 8 * 60 + 1, 15):
                window_duration_vars = []
                for room_id, movie_id, version, time_slot, var in var_list:
                    if room_id == room:
                        movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                        runtime = self._round_up_to_30(movie['runtime'])
                        slot_start_minutes = self._time_slot_to_minutes(time_slot)
                        slot_end_minutes = slot_start_minutes + runtime
                        window_end_minutes = window_start_minutes + 9 * 60
                        if (slot_start_minutes < window_end_minutes and slot_end_minutes > window_start_minutes):
                            overlap_start = max(slot_start_minutes, window_start_minutes)
                            overlap_end = min(slot_end_minutes, window_end_minutes)
                            overlap_duration = max(0, overlap_end - overlap_start)
                            if overlap_duration > 0:
                                window_duration_vars.append(overlap_duration * var)
                if window_duration_vars:
                    model.addConstr(cp.quicksum(window_duration_vars) <= 420,
                                    name=f"runtime_limit_{room}_{window_start_minutes}")
                    constraint_count += 1

        print(f"[COPT] added {constraint_count} constraints, {len(var_list)} variables")

        # solver params
        model.setParam(COPT.Param.TimeLimit, 300)
        model.setParam(COPT.Param.RelGap, 0.01)

        # solve with error handling (e.g., license)
        try:
            start_time = time.time()
            model.solve()
            solve_time = time.time() - start_time
            print(f"[COPT] solved in {solve_time:.2f}s, status={model.status}")
        except Exception as e:
            print("[COPT] solver error:", e)
            return [], -1, None

        # extract results
        schedule_results = []
        if model.status == COPT.OPTIMAL:
            print(f"[COPT] optimal objective: {model.objval:.2f}")
            for room, movie_id, version, time_slot, var in var_list:
                if var.x > 0.5:
                    room_capacity = self.cinema_df[self.cinema_df['room'] == room]['capacity'].iloc[0]
                    movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                    attendance = self._calculate_attendance(room_capacity, movie['rating'])
                    display_time = self._convert_to_display_time(time_slot)
                    schedule_results.append({
                        'room': room,
                        'showtime': display_time,
                        'id': movie_id,
                        'version': version,
                        'attendance': attendance
                    })
            schedule_results.sort(key=lambda x: (x['room'], self._sort_key_for_time(x['showtime'])))
            return schedule_results, 1, model.objval
        else:
            print("[COPT] no optimal solution, status:", model.status)
            return [], model.status, None

    def _sort_key_for_time(self, time_str):
        hour, minute = map(int, time_str.split(':'))
        if hour < 4:
            hour += 24
        return hour * 60 + minute

# =========================
# 主流程：7天循环
# =========================
def main():
    # load initial movie info
    movies_df = pd.read_csv(PREDICTED_MOVIES_CSV)   # must contain columns id, rating
    obs_df = pd.read_csv(OBS_CSV)                   # must contain day, movie_id, time_slot, weekday_flag, baseline_attendance, simulated_attendance

    # create dynamic model
    dyn = DynamicAttendanceModel(movies_df)

    days = sorted(obs_df['day'].unique())
    print(f"[Main] found days: {days}")

    for day in days:
        print(f"\n=== Day {day} processing ===")
        day_df = obs_df[obs_df['day'] == day]
        # update dynamic parameters using today's observations
        dyn.update_parameters(day_df)
        # save dynamic params (including updated_ratings) for MILP to read
        dyn.save_parameters(DYNAMIC_PARAMS_PATH)

        # run MILP (COPT) using updated ratings
        optimizer = CinemaSchedulingOptimizerCOPT(CINEMA_CSV, MOVIES_SCHEDULE_CSV)
        schedule, status, obj = optimizer.optimize_schedule()

        # save schedule for the day
        out_path = os.path.join(OUTPUT_DIR, f"day_{int(day)}_schedule.csv")
        if schedule:
            pd.DataFrame(schedule).to_csv(out_path, index=False)
            print(f"[Main] Day {day} schedule saved to {out_path}")
        else:
            print(f"[Main] Day {day} no schedule generated (status {status})")

    print("\n[Main] All days processed.")

if __name__ == "__main__":
    main()
