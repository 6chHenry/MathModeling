import pandas as pd
import numpy as np
from pulp import *
import math
from datetime import datetime, timedelta


class CinemaSchedulingOptimizer:
    def __init__(self, cinema_file, movies_file):
        """
        初始化排片优化器
        """
        self.cinema_df = pd.read_csv(cinema_file)
        self.movies_df = pd.read_csv(movies_file)

        # 营业时间设置 (10:00 到次日 03:00，即17小时)
        self.start_hour = 10
        self.end_hour = 27  # 次日3点用27表示
        self.time_slots = self._generate_time_slots()

        # 版本成本系数
        self.version_coeff = {'2D': 1.0, '3D': 1.1, 'IMAX': 1.15}

        # 基础参数
        self.basic_cost = 2.42  # 元/人
        self.fixed_cost = 90  # 元

        # 版本播放时长限制 (分钟)
        self.version_limits = {
            '3D': {'min': 0, 'max': 1200},
            'IMAX': {'min': 0, 'max': 1500}
        }

        # 题材播放次数限制
        self.genre_limits = {
            'Animation': {'min': 1, 'max': 5},
            'Horror': {'min': 0, 'max': 3},
            'Action': {'min': 2, 'max': 6},
            'Drama': {'min': 1, 'max': 6}
        }

        # 题材时间限制 (24小时制)
        self.genre_time_limits = {
            'Animation': {'latest_start': 19},  # 只能在白天播放，最晚19:00开始
            'Family': {'latest_start': 19},  # 只能在白天播放，最晚19:00开始
            'Horror': {'earliest_start': 21},  # 只能在晚上播放，最早21:00开始
            'Thriller': {'earliest_start': 21}  # 只能在晚上播放，最早21:00开始
        }

        # 黄金时段 (18:00-21:00)
        self.prime_time_start = 18
        self.prime_time_end = 21
        self.prime_time_multiplier = 1.3

    def _generate_time_slots(self):
        """生成时间段列表，每15分钟一个时间点"""
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
        """获取电影支持的版本列表"""
        movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
        versions = movie['version'].split(',')
        return [v.strip() for v in versions]

    def _round_up_to_30(self, runtime):
        """将播放时长向上取整到30分钟倍数"""
        return math.ceil(runtime / 30) * 30

    def _can_room_play_version(self, room, version):
        """检查放映厅是否支持特定版本"""
        room_info = self.cinema_df[self.cinema_df['room'] == room].iloc[0]
        return bool(room_info[version])

    def _calculate_ticket_price(self, movie_id, version, is_prime_time=False):
        """计算票价"""
        movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
        basic_price = movie['basic_price']

        # 版本价格调整
        if version == '2D':
            price = basic_price
        elif version == '3D':
            price = basic_price * 1.2
        elif version == 'IMAX':
            price = basic_price * 1.23

        # 黄金时段加价
        if is_prime_time:
            price *= self.prime_time_multiplier

        return price

    def _calculate_attendance(self, capacity, rating):
        """计算实际观影人数"""
        return math.floor(capacity * rating / 10)

    def _calculate_cost(self, capacity, version):
        """计算播放成本"""
        version_coeff = self.version_coeff[version]
        return version_coeff * capacity * self.basic_cost + self.fixed_cost

    def _get_sharing_rate(self, movie_id):
        """获取分成比例"""
        movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
        original_language = movie['original_language']
        # 如果包含普通话则为国产电影
        if 'Mandarin' in original_language:
            return 0.43
        else:
            return 0.51

    def _is_prime_time(self, time_slot):
        """判断是否为黄金时段"""
        hour = int(time_slot.split(':')[0])
        return self.prime_time_start <= hour < self.prime_time_end

    def _check_genre_time_constraint(self, movie_id, time_slot):
        """检查题材时间约束"""
        movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
        genres = [g.strip() for g in movie['genres'].split(',')]
        hour = int(time_slot.split(':')[0])

        # 转换27小时制到24小时制（次日凌晨）
        display_hour = hour if hour < 24 else hour - 24

        for genre in genres:
            if genre in self.genre_time_limits:
                constraints = self.genre_time_limits[genre]

                # 检查最早开始时间约束
                if 'earliest_start' in constraints:
                    earliest = constraints['earliest_start']
                    # 对于跨日的情况（如21:00到次日3:00），需要特殊处理
                    if hour < earliest and hour >= 10:  # 当日时间但早于最早时间
                        return False

                # 检查最晚开始时间约束
                if 'latest_start' in constraints:
                    latest = constraints['latest_start']
                    # 对于跨日的情况
                    if hour >= 24:  # 次日时间
                        return False  # 次日时间都不允许
                    elif hour >= latest:  # 当日但晚于最晚时间
                        return False

        return True

    def _time_slot_to_minutes(self, time_slot):
        """将时间段转换为从10:00开始的分钟数"""
        hour, minute = map(int, time_slot.split(':'))
        return (hour - self.start_hour) * 60 + minute

    def optimize_schedule(self):
        """主优化函数"""
        # 创建优化问题
        prob = LpProblem("Cinema_Scheduling", LpMaximize)

        # 决策变量：x[room][movie][version][time_slot] = 1 如果在该时间播放该电影
        x = {}

        # 为每个可行的组合创建决策变量
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

                        # 检查是否有足够时间播放完整部电影
                        runtime = self._round_up_to_30(movie['runtime'])
                        start_minutes = self._time_slot_to_minutes(time_slot)
                        end_minutes = start_minutes + runtime

                        if end_minutes <= (self.end_hour - self.start_hour) * 60:
                            var_name = f"x_{room}_{movie_id}_{version}_{time_slot}"
                            x[room][movie_id][version][time_slot] = LpVariable(var_name, cat='Binary')

        # 目标函数：最大化净收益
        revenue_terms = []
        cost_terms = []

        for room in x:
            room_capacity = self.cinema_df[self.cinema_df['room'] == room]['capacity'].iloc[0]

            for movie_id in x[room]:
                movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                rating = movie['rating']
                sharing_rate = self._get_sharing_rate(movie_id)

                for version in x[room][movie_id]:
                    for time_slot in x[room][movie_id][version]:
                        var = x[room][movie_id][version][time_slot]

                        # 计算收入
                        is_prime = self._is_prime_time(time_slot)
                        ticket_price = self._calculate_ticket_price(movie_id, version, is_prime)
                        attendance = self._calculate_attendance(room_capacity, rating)
                        ticket_revenue = ticket_price * attendance
                        net_revenue = ticket_revenue * (1 - sharing_rate)

                        # 计算成本
                        cost = self._calculate_cost(room_capacity, version)

                        # 添加到目标函数
                        revenue_terms.append(net_revenue * var)
                        cost_terms.append(cost * var)

        prob += lpSum(revenue_terms) - lpSum(cost_terms)

        # 约束1: 每个放映厅同一时间只能播放一部电影
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

                            # 检查时间重叠（包含15分钟清理时间）
                            if start_minutes <= current_minutes < end_minutes + 15:
                                overlapping_vars.append(x[room][movie_id][version][start_time])

                if overlapping_vars:
                    prob += lpSum(overlapping_vars) <= 1

        # 约束2: 版本总播放时长限制
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

        # 约束3: 题材播放次数限制
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

        # 约束5: 每部电影的放映次数约束（可根据需要设置）
        # 为了确保资源合理分配，可以设置每部电影的最大放映次数
        for movie_id in self.movies_df['id']:
            movie_shows = []

            for room in x:
                if movie_id in x[room]:
                    for version in x[room][movie_id]:
                        for time_slot in x[room][movie_id][version]:
                            movie_shows.append(x[room][movie_id][version][time_slot])

            if movie_shows:
                # 设置每部电影最多播放3次（可根据实际情况调整）
                prob += lpSum(movie_shows) <= 3
                # 设置每部电影最少播放1次（可选，确保每部电影都有机会）
                # prob += lpSum(movie_shows) >= 1
        for room in x:
            for start_hour in range(0, (self.end_hour - self.start_hour) - 8):  # 9小时窗口
                window_duration = []

                for movie_id in x[room]:
                    movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                    runtime = self._round_up_to_30(movie['runtime'])

                    for version in x[room][movie_id]:
                        for time_slot in x[room][movie_id][version]:
                            slot_hour = self._time_slot_to_minutes(time_slot) // 60

                            # 检查是否在当前9小时窗口内
                            if start_hour <= slot_hour < start_hour + 9:
                                window_duration.append(runtime * x[room][movie_id][version][time_slot])

                if window_duration:
                    prob += lpSum(window_duration) <= 420  # 7小时 = 420分钟

        # 求解
        prob.solve(PULP_CBC_CMD(msg=0))

        # 提取结果
        schedule_results = []

        if prob.status == 1:  # 最优解
            for room in x:
                room_capacity = self.cinema_df[self.cinema_df['room'] == room]['capacity'].iloc[0]

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

        # 按房间和时间排序
        schedule_results.sort(key=lambda x: (x['room'], x['showtime']))

        return schedule_results, prob.status, value(prob.objective) if prob.status == 1 else None


# 使用示例
def main():
    # 创建优化器实例
    optimizer = CinemaSchedulingOptimizer('D:\PythonProjects\MCM\input_data\df_cinema.csv', 'D:\PythonProjects\MCM\input_data\df_movies_schedule.csv')

    # 执行优化
    print("开始优化排片计划...")
    schedule, status, objective_value = optimizer.optimize_schedule()

    if status == 1:
        print(f"优化成功！最大净收益: {objective_value:.2f} 元")

        # 输出结果到CSV文件
        result_df = pd.DataFrame(schedule)
        result_df.to_csv('D:\PythonProjects\MCM\output_result\df_result_2.csv', index=False)

        print(f"排片计划已保存到 df_result_2_cbc.csv")
        print(f"总共安排了 {len(schedule)} 场放映")

        # 统计各种约束的满足情况
        print("\n=== 约束满足情况统计 ===")

        # 1. 题材播放次数统计
        genre_count = {}
        for item in schedule:
            movie = optimizer.movies_df[optimizer.movies_df['id'] == item['id']].iloc[0]
            movie_genres = [g.strip() for g in movie['genres'].split(',')]
            for genre in movie_genres:
                genre_count[genre] = genre_count.get(genre, 0) + 1

        print("题材播放次数:")
        for genre, limits in optimizer.genre_limits.items():
            count = genre_count.get(genre, 0)
            min_limit = limits.get('min', 0)
            max_limit = limits.get('max', '无限制')
            status = "✓" if count >= min_limit and (max_limit == '无限制' or count <= max_limit) else "✗"
            print(f"  {genre}: {count} 次 (要求: {min_limit}-{max_limit}) {status}")

        # 2. 版本播放时长统计
        version_duration = {'3D': 0, 'IMAX': 0}
        for item in schedule:
            if item['version'] in version_duration:
                movie = optimizer.movies_df[optimizer.movies_df['id'] == item['id']].iloc[0]
                runtime = optimizer._round_up_to_30(movie['runtime'])
                version_duration[item['version']] += runtime

        print("\n版本播放时长:")
        for version in ['3D', 'IMAX']:
            duration = version_duration[version]
            max_limit = optimizer.version_limits[version]['max']
            status = "✓" if duration <= max_limit else "✗"
            print(f"  {version}: {duration} 分钟 (上限: {max_limit} 分钟) {status}")

        # 3. 每部电影播放次数统计
        movie_count = {}
        for item in schedule:
            movie_count[item['id']] = movie_count.get(item['id'], 0) + 1

        print(f"\n每部电影播放次数:")
        for movie_id, count in movie_count.items():
            movie = optimizer.movies_df[optimizer.movies_df['id'] == movie_id].iloc[0]
            print(f"  电影{movie_id}: {count} 次")

        # 显示部分结果
        print("\n=== 排片计划预览 ===")
        print(result_df.head(10).to_string(index=False))

    else:
        print(f"优化失败，状态码: {status}")
        print("可能的原因:")
        print("1. 约束条件过于严格，无可行解")
        print("2. 数据存在问题")
        print("3. 模型设置需要调整")


if __name__ == "__main__":
    main()