import pandas as pd
import math
from datetime import timedelta, date
import coptpy as cp
from coptpy import COPT
import matplotlib.pyplot as plt
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

class MultiPeriodCinemaScheduler:
    """
    多周期多放映厅排片优化模型
    
    基于历史上座率数据，对未来一周（含工作日、休息日）进行排片优化，
    考虑工作日和休息日的上座率差异，以及各种运营约束条件。
    """
    
    def __init__(self, cinema_file: str, movies_file: str, 
                 historical_attendance_file: Optional[str] = None):
        """
        初始化多周期排片优化器
        
        Args:
            cinema_file: 放映厅信息文件路径
            movies_file: 电影信息文件路径
            historical_attendance_file: 历史上座率数据文件路径（可选）
        """
        # 读取基础数据
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
        
        # 题材播放次数限制（每日）
        self.daily_genre_limits = {
            'Animation': {'min': 0, 'max': 2},
            'Horror': {'min': 0, 'max': 1},
            'Action': {'min': 0, 'max': 2},
            'Drama': {'min': 0, 'max': 2}
        }
        
        # 题材播放次数限制（每周）
        self.weekly_genre_limits = {
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
        
        # 多周期设置
        self.schedule_days = 7  # 一周
        self.day_types = {
            0: 'weekday',   # 周一
            1: 'weekday',   # 周二
            2: 'weekday',   # 周三
            3: 'weekday',   # 周四
            4: 'weekday',   # 周五
            5: 'weekend',   # 周六
            6: 'weekend'    # 周日
        }
        
        # 上座率调整因子（基于历史数据假设）
        self.attendance_factors = {
            'weekday': {
                'morning': 0.7,    # 工作日上午上座率因子
                'afternoon': 0.8,  # 工作日下午上座率因子
                'prime': 1.0,      # 工作日黄金时段上座率因子
                'evening': 0.9,    # 工作日晚上上座率因子
                'night': 0.6       # 工作日凌晨上座率因子
            },
            'weekend': {
                'morning': 0.9,    # 周末上午上座率因子
                'afternoon': 1.1,  # 周末下午上座率因子
                'prime': 1.3,      # 周末黄金时段上座率因子
                'evening': 1.2,    # 周末晚上上座率因子
                'night': 0.8       # 周日凌晨上座率因子
            }
        }
        
        # 历史上座率数据（如果有）
        self.historical_attendance = None
        if historical_attendance_file:
            self.historical_attendance = pd.read_csv(historical_attendance_file)
            self._analyze_historical_data()
        
        # 电影热度衰减因子（每周衰减）
        self.movie_decay_factors = {}
        self._initialize_movie_decay_factors()
        
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
    
    def _initialize_movie_decay_factors(self):
        """初始化电影热度衰减因子"""
        for _, movie in self.movies_df.iterrows():
            movie_id = movie['id']
            # 基于评分设置初始热度，评分越高衰减越慢
            rating = movie['rating']
            # 评分8.0以上的电影衰减较慢，评分6.0以下的电影衰减较快
            if rating >= 8.0:
                decay_rate = 0.95  # 每周保留95%的热度
            elif rating >= 7.0:
                decay_rate = 0.90  # 每周保留90%的热度
            else:
                decay_rate = 0.85  # 每周保留85%的热度
            
            self.movie_decay_factors[movie_id] = decay_rate
    
    def _analyze_historical_data(self):
        """分析历史上座率数据，调整上座率因子"""
        if self.historical_attendance is None:
            return
        
        # 按日期类型和时段分析历史上座率
        self.historical_attendance['date'] = pd.to_datetime(self.historical_attendance['date'])
        self.historical_attendance['day_of_week'] = self.historical_attendance['date'].dt.dayofweek
        self.historical_attendance['day_type'] = self.historical_attendance['day_of_week'].apply(
            lambda x: 'weekend' if x >= 5 else 'weekday'
        )
        
        # 分析不同时段的上座率模式
        for day_type in ['weekday', 'weekend']:
            day_data = self.historical_attendance[self.historical_attendance['day_type'] == day_type]
            
            if len(day_data) > 0:
                # 计算各时段的平均上座率
                for time_period in ['morning', 'afternoon', 'prime', 'evening', 'night']:
                    period_data = self._filter_by_time_period(day_data, time_period)
                    if len(period_data) > 0:
                        avg_attendance = period_data['attendance_rate'].mean()
                        # 调整上座率因子
                        self.attendance_factors[day_type][time_period] = avg_attendance
    
    def _filter_by_time_period(self, data, time_period):
        """根据时间段筛选数据"""
        if time_period == 'morning':
            return data[(data['showtime'] >= '10:00') & (data['showtime'] < '12:00')]
        elif time_period == 'afternoon':
            return data[(data['showtime'] >= '12:00') & (data['showtime'] < '18:00')]
        elif time_period == 'prime':
            return data[(data['showtime'] >= '18:00') & (data['showtime'] < '21:00')]
        elif time_period == 'evening':
            return data[(data['showtime'] >= '21:00') & (data['showtime'] < '24:00')]
        elif time_period == 'night':
            return data[(data['showtime'] >= '00:00') & (data['showtime'] < '03:00')]
        return pd.DataFrame()
    
    def _get_time_period(self, time_slot):
        """获取时间段类型"""
        hour = int(time_slot.split(':')[0])
        if hour < 24:
            display_hour = hour
        else:
            display_hour = hour - 24
        
        if 10 <= display_hour < 12:
            return 'morning'
        elif 12 <= display_hour < 18:
            return 'afternoon'
        elif 18 <= display_hour < 21:
            return 'prime'
        elif 21 <= display_hour < 24:
            return 'evening'
        else:  # 0-3点
            return 'night'
    
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
        else:
            # 默认按2D处理
            price = basic_price
        
        # 黄金时段加价
        if is_prime_time:
            price *= self.prime_time_multiplier
        
        return price
    
    def _calculate_attendance(self, capacity, rating, day_type, time_slot, movie_id, day_offset=0):
        """计算实际观影人数，考虑日期类型、时间段和电影热度衰减"""
        # 基础上座率
        base_attendance = math.floor(capacity * rating / 10)
        
        # 获取时间段类型
        time_period = self._get_time_period(time_slot)
        
        # 应用上座率因子
        attendance_factor = self.attendance_factors[day_type][time_period]
        
        # 应用电影热度衰减（基于天数偏移）
        decay_factor = self.movie_decay_factors[movie_id] ** day_offset
        
        # 计算最终上座率
        adjusted_attendance = math.floor(base_attendance * attendance_factor * decay_factor)
        
        # 确保不超过容量
        return min(adjusted_attendance, capacity)
    
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
    
    def optimize_multi_period_schedule(self, start_date: Optional[str] = None):
        """
        优化多周期排片计划
        
        Args:
            start_date: 开始日期，格式为'YYYY-MM-DD'，如果为None则使用当前日期
        
        Returns:
            tuple: (排片结果列表, 求解状态, 目标函数值)
        """
        if start_date is None:
            start_date = date.today().strftime('%Y-%m-%d')
        
        return self._optimize_with_copt(start_date)
    
    def _optimize_with_copt(self, start_date):
        """使用COPT求解器的多周期优化函数"""
        print("使用COPT求解器进行多周期排片优化...")
        
        # 创建COPT环境和模型
        env = cp.Envr()
        model = env.createModel("Multi_Period_Cinema_Scheduling")
        
        # 决策变量字典 x[day][room][movie][version][showtime]
        x = {}
        var_list = []  # 存储所有变量以便后续访问
        
        # 创建决策变量
        print("创建多周期决策变量...")
        for day in range(self.schedule_days):
            day_type = self.day_types[day]
            x[day] = {}
            
            for _, room_info in self.cinema_df.iterrows():
                room = room_info['room']
                x[day][room] = {}
                
                for _, movie in self.movies_df.iterrows():
                    movie_id = movie['id']
                    versions = self._get_versions(movie_id)
                    x[day][room][movie_id] = {}
                    
                    for version in versions:
                        if not self._can_room_play_version(room, version):
                            continue
                            
                        x[day][room][movie_id][version] = {}
                        
                        for time_idx, time_slot in enumerate(self.time_slots):
                            # 检查题材时间约束
                            if not self._check_genre_time_constraint(movie_id, time_slot):
                                continue
                            
                            # 创建二进制决策变量
                            var_name = f"x_{day}_{room}_{movie_id}_{version}_{time_idx}"
                            var = model.addVar(
                                vtype=COPT.BINARY,
                                name=var_name
                            )
                            x[day][room][movie_id][version][time_slot] = var
                            var_list.append(var)
        
        print(f"创建了 {len(var_list)} 个决策变量")
        
        # 目标函数：最大化7天总净收益
        print("构建目标函数...")
        objective = 0
        
        for day in range(self.schedule_days):
            day_type = self.day_types[day]
            
            for room, room_data in x[day].items():
                room_info = self.cinema_df[self.cinema_df['room'] == room].iloc[0]
                capacity = room_info['capacity']
                
                for movie_id, movie_data in room_data.items():
                    movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                    sharing_rate = self._get_sharing_rate(movie_id)
                    
                    for version, version_data in movie_data.items():
                        for time_slot, var in version_data.items():
                            # 计算收益
                            is_prime = self._is_prime_time(time_slot)
                            ticket_price = self._calculate_ticket_price(movie_id, version, is_prime)
                            attendance = self._calculate_attendance(
                                capacity, movie['rating'], day_type, time_slot, movie_id, day
                            )
                            cost = self._calculate_cost(capacity, version)
                            
                            # 净收益 = 票房收入 * (1 - 分成比例) - 播放成本
                            revenue = ticket_price * attendance * (1 - sharing_rate)
                            net_profit = revenue - cost
                            
                            objective += net_profit * var
        
        model.setObjective(objective, COPT.MAXIMIZE)
        
        # 约束条件
        print("添加约束条件...")
        
        # 1. 放映厅时间冲突约束
        print("添加放映厅时间冲突约束...")
        for day in range(self.schedule_days):
            for room in x[day].keys():
                room_vars = []
                
                # 收集该厅该天的所有场次变量及其时间信息
                for movie_id in x[day][room].keys():
                    movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                    runtime = self._round_up_to_30(movie['runtime'])
                    
                    for version in x[day][room][movie_id].keys():
                        for time_slot, var in x[day][room][movie_id][version].items():
                            start_minutes = self._time_slot_to_minutes(time_slot)
                            end_minutes = start_minutes + runtime + 15  # 加上15分钟间隔
                            room_vars.append((var, start_minutes, end_minutes))
                
                # 对于每对可能冲突的场次，添加约束
                for i in range(len(room_vars)):
                    for j in range(i + 1, len(room_vars)):
                        var1, start1, end1 = room_vars[i]
                        var2, start2, end2 = room_vars[j]
                        
                        # 如果时间重叠，则两个场次不能同时选择
                        if not (end1 <= start2 or end2 <= start1):
                            model.addConstr(
                                var1 + var2 <= 1,
                                name=f"conflict_{day}_{room}_{i}_{j}"
                            )
        
        # 2. 连续运行时间约束（任意连续9小时内，单厅播放时长≤7小时）
        print("添加连续运行时间约束...")
        window_hours = 9
        max_play_hours = 7
        
        for day in range(self.schedule_days):
            for room in x[day].keys():
                # 遍历每个9小时窗口
                for window_start in range(0, 24 * 60 - window_hours * 60 + 1, 15):
                    window_end = window_start + window_hours * 60
                    window_vars = []
                    
                    for movie_id in x[day][room].keys():
                        movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                        runtime = self._round_up_to_30(movie['runtime'])
                        
                        for version in x[day][room][movie_id].keys():
                            for time_slot, var in x[day][room][movie_id][version].items():
                                start_minutes = self._time_slot_to_minutes(time_slot)
                                end_minutes = start_minutes + runtime
                                
                                # 如果场次在窗口内
                                if start_minutes >= window_start and end_minutes <= window_end:
                                    window_vars.append(var * runtime)
                                elif start_minutes < window_end and end_minutes > window_start:
                                    # 部分重叠的情况
                                    overlap_start = max(start_minutes, window_start)
                                    overlap_end = min(end_minutes, window_end)
                                    overlap_duration = overlap_end - overlap_start
                                    if overlap_duration > 0:
                                        window_vars.append(var * overlap_duration)
                    
                    if window_vars:
                        model.addConstr(
                            cp.quicksum(window_vars) <= max_play_hours * 60,
                            name=f"continuous_time_{day}_{room}_{window_start}"
                        )
        
        # 3. 版本总时长限制
        print("添加版本总时长限制...")
        for day in range(self.schedule_days):
            version_duration = {'3D': 0, 'IMAX': 0}
            
            for room in x[day].keys():
                for movie_id in x[day][room].keys():
                    movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                    runtime = self._round_up_to_30(movie['runtime'])
                    
                    for version in x[day][room][movie_id].keys():
                        if version in ['3D', 'IMAX']:
                            for time_slot, var in x[day][room][movie_id][version].items():
                                version_duration[version] += var * runtime
            
            # 添加约束
            if version_duration['3D'] > 0:
                model.addConstr(
                    version_duration['3D'] <= self.version_limits['3D']['max'],
                    name=f"3D_limit_{day}"
                )
            if version_duration['IMAX'] > 0:
                model.addConstr(
                    version_duration['IMAX'] <= self.version_limits['IMAX']['max'],
                    name=f"IMAX_limit_{day}"
                )
        
        # 4. 题材播放次数约束（每日和每周）
        print("添加题材播放次数约束...")
        
        # 每日约束
        for day in range(self.schedule_days):
            genre_counts = {}
            
            for room in x[day].keys():
                for movie_id in x[day][room].keys():
                    movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                    genres = [g.strip() for g in movie['genres'].split(',')]
                    
                    for genre in genres:
                        if genre in self.daily_genre_limits:
                            if genre not in genre_counts:
                                genre_counts[genre] = []
                            
                            for version in x[day][room][movie_id].keys():
                                for time_slot, var in x[day][room][movie_id][version].items():
                                    genre_counts[genre].append(var)
            
            # 添加每日题材约束
            for genre, vars_list in genre_counts.items():
                if genre in self.daily_genre_limits:
                    limits = self.daily_genre_limits[genre]
                    
                    if limits['max'] < float('inf'):
                        model.addConstr(
                            cp.quicksum(vars_list) <= limits['max'],
                            name=f"daily_{genre}_max_{day}"
                        )
                    
                    if limits['min'] > 0:
                        model.addConstr(
                            cp.quicksum(vars_list) >= limits['min'],
                            name=f"daily_{genre}_min_{day}"
                        )
        
        # 每周约束
        weekly_genre_counts = {}
        for day in range(self.schedule_days):
            for room in x[day].keys():
                for movie_id in x[day][room].keys():
                    movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                    genres = [g.strip() for g in movie['genres'].split(',')]
                    
                    for genre in genres:
                        if genre in self.weekly_genre_limits:
                            if genre not in weekly_genre_counts:
                                weekly_genre_counts[genre] = []
                            
                            for version in x[day][room][movie_id].keys():
                                for time_slot, var in x[day][room][movie_id][version].items():
                                    weekly_genre_counts[genre].append(var)
        
        # 添加每周题材约束
        for genre, vars_list in weekly_genre_counts.items():
            if genre in self.weekly_genre_limits:
                limits = self.weekly_genre_limits[genre]
                
                if limits['max'] < float('inf'):
                    model.addConstr(
                        cp.quicksum(vars_list) <= limits['max'],
                        name=f"weekly_{genre}_max"
                    )
                
                if limits['min'] > 0:
                    model.addConstr(
                        cp.quicksum(vars_list) >= limits['min'],
                        name=f"weekly_{genre}_min"
                    )
        
        # 5. 营业时间约束（确保所有场次在营业时间内结束）
        print("添加营业时间约束...")
        for day in range(self.schedule_days):
            for room in x[day].keys():
                for movie_id in x[day][room].keys():
                    movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                    runtime = self._round_up_to_30(movie['runtime'])
                    
                    for version in x[day][room][movie_id].keys():
                        for time_slot, var in x[day][room][movie_id][version].items():
                            start_minutes = self._time_slot_to_minutes(time_slot)
                            end_minutes = start_minutes + runtime
                            
                            # 确保在营业时间内结束（17小时 = 1020分钟）
                            if end_minutes > 17 * 60:  # 超过次日3:00
                                # 这个时间段不可选择
                                model.addConstr(var == 0, name=f"business_hours_{day}_{room}_{movie_id}_{version}_{time_slot}")
        
        # 求解模型
        print("开始求解模型...")
        model.setParam(COPT.Param.TimeLimit, 300)  # 5分钟时间限制
        model.setParam(COPT.Param.LogToConsole, 1)
        
        model.solve()
        
        # 检查求解状态
        status = model.status
        print(f"求解状态: {status}")
        
        if status == COPT.OPTIMAL:
            print("找到最优解")
            obj_value = model.objval
            print(f"最优目标函数值: {obj_value:.2f}")
            
            # 提取解
            schedule_results = []
            current_date = pd.to_datetime(start_date)
            
            for day in range(self.schedule_days):
                day_date = current_date + timedelta(days=day)
                day_type = self.day_types[day]
                
                for room in x[day].keys():
                    for movie_id in x[day][room].keys():
                        for version in x[day][room][movie_id].keys():
                            for time_slot, var in x[day][room][movie_id][version].items():
                                if var.x > 0.5:  # 变量值接近1
                                    movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                                    room_info = self.cinema_df[self.cinema_df['room'] == room].iloc[0]
                                    
                                    # 计算相关指标
                                    capacity = room_info['capacity']
                                    runtime = self._round_up_to_30(movie['runtime'])
                                    is_prime = self._is_prime_time(time_slot)
                                    ticket_price = self._calculate_ticket_price(movie_id, version, is_prime)
                                    attendance = self._calculate_attendance(
                                        capacity, movie['rating'], day_type, time_slot, movie_id, day
                                    )
                                    cost = self._calculate_cost(capacity, version)
                                    sharing_rate = self._get_sharing_rate(movie_id)
                                    revenue = ticket_price * attendance * (1 - sharing_rate)
                                    net_profit = revenue - cost
                                    
                                    schedule_results.append({
                                        'date': day_date.strftime('%Y-%m-%d'),
                                        'day_type': day_type,
                                        'room': room,
                                        'movie_id': movie_id,
                                        'movie_title': movie.get('title', f'Movie_{movie_id}'),
                                        'version': version,
                                        'showtime': self._convert_to_display_time(time_slot),
                                        'runtime': runtime,
                                        'capacity': capacity,
                                        'attendance': attendance,
                                        'ticket_price': ticket_price,
                                        'revenue': revenue,
                                        'cost': cost,
                                        'net_profit': net_profit,
                                        'sharing_rate': sharing_rate,
                                        'is_prime_time': is_prime
                                    })
            
            return schedule_results, status, obj_value
            
        else:
            print(f"未找到最优解，求解状态: {status}")
            return [], status, 0
    
    def generate_analysis_report(self, schedule_results):
        """生成排片分析报告"""
        if not schedule_results:
            print("没有排片结果可分析")
            return
        
        df_schedule = pd.DataFrame(schedule_results)
        
        print("\n=== 多周期排片分析报告 ===")
        print(f"排片周期: {df_schedule['date'].min()} 到 {df_schedule['date'].max()}")
        print(f"总场次数: {len(df_schedule)}")
        print(f"总净收益: {df_schedule['net_profit'].sum():.2f} 元")
        print(f"平均每场净收益: {df_schedule['net_profit'].mean():.2f} 元")
        
        # 按日期分析
        print("\n--- 每日收益分析 ---")
        daily_analysis = df_schedule.groupby('date').agg({
            'net_profit': 'sum',
            'attendance': 'sum',
            'movie_id': 'count'
        }).round(2)
        daily_analysis.columns = ['每日净收益', '每日观影人数', '每日场次数']
        print(daily_analysis)
        
        # 按放映厅分析
        print("\n--- 放映厅利用率分析 ---")
        room_analysis = df_schedule.groupby('room').agg({
            'net_profit': 'sum',
            'attendance': 'sum',
            'movie_id': 'count'
        }).round(2)
        room_analysis.columns = ['放映厅收益', '观影人数', '场次数']
        print(room_analysis)
        
        # 按版本分析
        print("\n--- 版本收益分析 ---")
        version_analysis = df_schedule.groupby('version').agg({
            'net_profit': 'sum',
            'attendance': 'sum',
            'movie_id': 'count'
        }).round(2)
        version_analysis.columns = ['版本收益', '观影人数', '场次数']
        print(version_analysis)
        
        # 黄金时段分析
        print("\n--- 黄金时段分析 ---")
        prime_analysis = df_schedule.groupby('is_prime_time').agg({
            'net_profit': 'sum',
            'attendance': 'sum',
            'movie_id': 'count'
        }).round(2)
        # 重置索引并重命名
        prime_analysis = prime_analysis.reset_index()
        prime_analysis['is_prime_time'] = prime_analysis['is_prime_time'].map({False: '非黄金时段', True: '黄金时段'})
        prime_analysis = prime_analysis.set_index('is_prime_time')
        prime_analysis.columns = ['收益', '观影人数', '场次数']
        print(prime_analysis)
    
    def save_schedule_to_csv(self, schedule_results, filename):
        """保存排片结果到CSV文件"""
        if schedule_results:
            df_schedule = pd.DataFrame(schedule_results)
            df_schedule.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"排片结果已保存到: {filename}")
    
    def visualize_schedule(self, schedule_results):
        """可视化排片结果"""
        if not schedule_results:
            print("没有排片结果可可视化")
            return
        
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        df_schedule = pd.DataFrame(schedule_results)
        
        # 创建多个子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 每日收益趋势
        daily_profit = df_schedule.groupby('date')['net_profit'].sum()
        axes[0, 0].plot(daily_profit.index, daily_profit.values, marker='o', linewidth=2, markersize=8)
        axes[0, 0].set_title('每日净收益趋势', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('净收益 (元)', fontsize=12)
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 放映厅利用率
        room_counts = df_schedule.groupby('room').size()
        axes[0, 1].bar(room_counts.index, room_counts.values, color='skyblue', alpha=0.8)
        axes[0, 1].set_title('放映厅场次分布', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('场次数', fontsize=12)
        axes[0, 1].set_xlabel('放映厅', fontsize=12)
        
        # 3. 版本收益对比
        version_profit = df_schedule.groupby('version')['net_profit'].sum()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        axes[1, 0].pie(version_profit.values, labels=version_profit.index, autopct='%1.1f%%',
                      colors=colors[:len(version_profit)], startangle=90)
        axes[1, 0].set_title('版本收益占比', fontsize=14, fontweight='bold')
        
        # 4. 时段分布热力图
        df_schedule['hour'] = df_schedule['showtime'].str.split(':').str[0].astype(int)
        time_dist = df_schedule.groupby(['date', 'hour']).size().unstack(fill_value=0)
        
        im = axes[1, 1].imshow(time_dist.values, cmap='YlOrRd', aspect='auto')
        axes[1, 1].set_title('每日各时段场次热力图', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('小时', fontsize=12)
        axes[1, 1].set_ylabel('日期', fontsize=12)
        axes[1, 1].set_xticks(range(len(time_dist.columns)))
        axes[1, 1].set_xticklabels(time_dist.columns)
        axes[1, 1].set_yticks(range(len(time_dist.index)))
        axes[1, 1].set_yticklabels(time_dist.index, rotation=45)
        
        # 添加颜色条
        plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
        
        plt.tight_layout()
        plt.savefig('multi_period_schedule_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """主函数：演示多周期排片优化"""
    try:
        # 初始化排片优化器
        print("初始化多周期排片优化器...")
        scheduler = MultiPeriodCinemaScheduler(
            cinema_file='../input_data/df_cinema.csv',
            movies_file='../input_data/df_movies_schedule_ours.csv'
        )
        
        # 优化排片计划
        print("开始多周期排片优化...")
        schedule_results, status, obj_value = scheduler.optimize_multi_period_schedule('2024-08-12')
        
        if schedule_results:
            # 生成分析报告
            scheduler.generate_analysis_report(schedule_results)
            
            # 保存结果
            scheduler.save_schedule_to_csv(schedule_results, 'multi_period_schedule_results.csv')
            
            # 可视化
            scheduler.visualize_schedule(schedule_results)
            
            print(f"\n优化完成！总净收益: {obj_value:.2f} 元")
            
        else:
            print("优化失败，可能的原因：")
            print("1. 约束条件过于严格")
            print("2. 数据存在问题")
            print("3. 模型设置需要调整")
            
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
