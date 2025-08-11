import pandas as pd
import math
import time
import coptpy as cp
from coptpy import COPT

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
            'Horror': {'min':0, 'max': 3},
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
            slots.append(f"{current_hour:02d}:{current_minute:02d}")  # 补齐为两位数字
            current_minute += 15
            if current_minute >= 60:
                current_minute = 0
                current_hour += 1

        return slots

    def _convert_to_display_time(self, time_slot):
        """将内部时间格式转换为标准24小时制显示格式"""
        hour, minute = map(int, time_slot.split(':'))

        if hour >= 24:
            # 次日时间，转换为标准格式（例如：25:30 -> 01:30）
            display_hour = hour - 24
            return f"{display_hour:02d}:{minute:02d}"
        else:
            # 当日时间，保持原样
            return time_slot

    def _get_versions(self, movie_id):
        """获取电影支持的版本列表"""
        movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
        versions = movie['version'].split(',')
        return [v.strip() for v in versions]    # 去除空格

    def _round_up_to_30(self, runtime):
        """将播放时长向上取整到30分钟倍数"""
        return math.ceil(runtime / 30) * 30

    def _can_room_play_version(self, room, version):
        """检查放映厅是否支持特定版本"""
        room_info = self.cinema_df[self.cinema_df['room'] == room].iloc[0]
        return bool(room_info[version])

    def _calculate_ticket_price(self, movie_id, version, is_prime_time=False):
        """计算票价"""
        movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]   # 取出电影对应的那一行
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
        if hour < 24:
            pass
        else:
            hour - 24

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

    def optimize_schedule(self, use_copt=True):
        return self._optimize_with_copt()

    def _optimize_with_copt(self):
        """使用COPT求解器的优化函数"""
        print("使用COPT求解器进行优化...")

        # 创建COPT环境和模型
        env = cp.Envr()
        model = env.createModel("Cinema_Scheduling")

        # 决策变量字典    x[room][movie][version][showtime]
        x = {}
        var_list = []  # 存储所有变量以便后续访问

        # 创建决策变量
        print("创建决策变量...")
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
                            var_name = f"x_{room}_{movie_id}_{version}_{time_slot}"     # 为变量命名
                            var = model.addVar(vtype=COPT.BINARY, name=var_name)        # 二进制变量
                            x[room][movie_id][version][time_slot] = var
                            var_list.append((room, movie_id, version, time_slot, var))

                        # 满足所有约束条件方才创建变量

        # 目标函数
        print("设置目标函数...")
        obj_expr = 0


        # 枚举了所有可能性
        for room, movie_id, version, time_slot, var in var_list:
            room_capacity = self.cinema_df[self.cinema_df['room'] == room]['capacity'].iloc[0]
            movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
            rating = movie['rating']
            sharing_rate = self._get_sharing_rate(movie_id)

            # 计算收入
            is_prime = self._is_prime_time(time_slot)
            ticket_price = self._calculate_ticket_price(movie_id, version, is_prime)
            attendance = self._calculate_attendance(room_capacity, rating)
            ticket_revenue = ticket_price * attendance
            net_revenue = ticket_revenue * (1 - sharing_rate)

            # 计算成本
            cost = self._calculate_cost(room_capacity, version)

            # 净收益
            net_profit = net_revenue - cost
            obj_expr += net_profit * var

        model.setObjective(obj_expr, COPT.MAXIMIZE)

        # 添加约束
        print("添加约束条件...")
        constraint_count = 0

        # 约束1: 每个放映厅同一时间只能播放一部电影（包含15分钟清理时间）
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
                    model.addConstr(cp.quicksum(overlapping_vars) <= 1,
                                    name=f"time_conflict_{room}_{time_slot}")
                    constraint_count += 1

        # 约束2: 版本总播放时长限制
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

        # 约束3: 题材播放次数限制
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
                if 'min' in limits:  # 注意：Horror没有min限制
                    model.addConstr(cp.quicksum(genre_vars) >= limits['min'],
                                    name=f"{genre}_min_shows")
                    constraint_count += 1


        # 约束4: 设备连续运行时长限制（每连续9小时内累计播放不超过7小时）
        for room in x:
            # 遍历所有可能的9小时滑动窗口
            for window_start_minutes in range(0, (self.end_hour - self.start_hour) * 60 - 8 * 60 + 1, 15):  # 每15分钟一个窗口
                window_duration_vars = []

                for room_id, movie_id, version, time_slot, var in var_list:
                    if room_id == room:
                        movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                        runtime = self._round_up_to_30(movie['runtime'])
                        slot_start_minutes = self._time_slot_to_minutes(time_slot)
                        slot_end_minutes = slot_start_minutes + runtime

                        # 检查电影播放时间是否与9小时窗口有重叠
                        window_end_minutes = window_start_minutes + 9 * 60

                        # 如果电影在窗口内开始或在窗口内结束，则计入该窗口的播放时长
                        if (slot_start_minutes < window_end_minutes and slot_end_minutes > window_start_minutes):
                            # 计算重叠的时长
                            overlap_start = max(slot_start_minutes, window_start_minutes)
                            overlap_end = min(slot_end_minutes, window_end_minutes)
                            overlap_duration = max(0, overlap_end - overlap_start)

                            if overlap_duration > 0:
                                window_duration_vars.append(overlap_duration * var)

                if window_duration_vars:
                    model.addConstr(cp.quicksum(window_duration_vars) <= 420,  # 7小时 = 420分钟
                                    name=f"runtime_limit_{room}_{window_start_minutes}")
                    constraint_count += 1

        print(f"共添加了 {constraint_count} 个约束条件")
        print(f"共有 {len(var_list)} 个决策变量")

        # 设置求解参数
        model.setParam(COPT.Param.TimeLimit, 300)  # 5分钟时间限制
        model.setParam(COPT.Param.RelGap, 0.01)  # 1%相对差距

        # 求解
        print("开始求解...")
        start_time = time.time()
        model.solve()
        solve_time = time.time() - start_time

        print(f"求解完成，耗时: {solve_time:.2f} 秒")
        print(f"求解状态: {model.status}")

        # 提取结果
        schedule_results = []

        if model.status == COPT.OPTIMAL:
            print(f"最优目标值: {model.objval:.2f} 元")

            for room, movie_id, version, time_slot, var in var_list:
                if var.x > 0.5:  # 二进制变量值接近1
                    room_capacity = self.cinema_df[self.cinema_df['room'] == room]['capacity'].iloc[0]
                    movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                    attendance = self._calculate_attendance(room_capacity, movie['rating'])

                    # 转换时间格式为标准24小时制
                    display_time = self._convert_to_display_time(time_slot)

                    schedule_results.append({
                        'room': room,
                        'showtime': display_time,  # 使用转换后的标准时间格式
                        'id': movie_id,
                        'version': version,
                        'attendance': attendance
                    })

            # 按房间和时间排序（这里需要特殊处理跨日时间排序）
            schedule_results.sort(key=lambda x: (x['room'], self._sort_key_for_time(x['showtime'])))

            return schedule_results, 1, model.objval  # 1表示最优解

        else:
            print("未找到最优解")
            return [], model.status, None

    def _sort_key_for_time(self, time_str):
        """用于时间排序的辅助函数，处理跨日时间"""
        hour, minute = map(int, time_str.split(':'))
        # 如果是凌晨时间（0-3点），加24转换为排序用的时间
        if hour < 4:
            hour += 24
        return hour * 60 + minute


# 使用示例
def main():
    # 创建优化器实例
    optimizer = CinemaSchedulingOptimizer(r'F:\MathModeling\input_data\df_cinema.csv',
                                          r'F:\MathModeling\Q3\df_movies_schedule_ours_new.csv')

    # 执行优化
    print("开始优化排片计划...")
    schedule, status, objective_value = optimizer.optimize_schedule()

    if status == 1:
        print(f"优化成功！最大净收益: {objective_value:.2f} 元")

        # 输出结果到CSV文件
        result_df = pd.DataFrame(schedule)
        result_df.to_csv(r'F:\MathModeling\Q3\df_result_2_copt_ours_new.csv', index=False)

        print("排片计划已保存到 df_result_2_copt_ours_new.csv")
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
            status_check = "✓" if count >= min_limit and (max_limit == '无限制' or count <= max_limit) else "✗"
            print(f"  {genre}: {count} 次 (要求: {min_limit}-{max_limit}) {status_check}")

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
            status_check = "✓" if duration <= max_limit else "✗"
            print(f"  {version}: {duration} 分钟 (上限: {max_limit} 分钟) {status_check}")

        # 显示时间范围
        if schedule:
            times = [item['showtime'] for item in schedule]
            print(f"排片时间范围: {min(times)} - {max(times)}")


    else:
        print(f"优化失败，状态码: {status}")
        print("可能的原因:")
        print("1. 约束条件过于严格，无可行解")
        print("2. 数据存在问题")
        print("3. 模型设置需要调整")


if __name__ == "__main__":
    main()
