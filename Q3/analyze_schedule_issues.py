import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

class ScheduleAnalyzer:
    def __init__(self, cinema_file, movies_file, schedule_file):
        """
        初始化排片分析器
        """
        self.cinema_df = pd.read_csv(cinema_file)
        self.movies_df = pd.read_csv(movies_file)
        self.schedule_df = pd.read_csv(schedule_file)
        
        # 营业时间设置
        self.start_hour = 10
        self.end_hour = 27  # 次日3点用27表示
        
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
            'Thriller': {'earliest_start':  21}  # 只能在晚上播放，最早21:00开始
        }
        
        # 黄金时段 (18:00-21:00)
        self.prime_time_start = 18
        self.prime_time_end = 21
        self.prime_time_multiplier = 1.3
        
        # 预处理数据
        self._preprocess_data()
    
    def _preprocess_data(self):
        """预处理数据"""
        # 为电影数据添加时长取整
        self.movies_df['rounded_runtime'] = self.movies_df['runtime'].apply(lambda x: math.ceil(x / 30) * 30)
        
        # 为排片数据添加时间解析
        self.schedule_df['start_hour'] = self.schedule_df['showtime'].apply(lambda x: int(x.split(':')[0]))
        self.schedule_df['start_minute'] = self.schedule_df['showtime'].apply(lambda x: int(x.split(':')[1]))
        
        # 计算每场电影的结束时间
        self.schedule_df['end_time'] = self.schedule_df.apply(
            lambda row: self._calculate_end_time(row['showtime'], row['id']), axis=1
        )
        
        # 标记黄金时段
        self.schedule_df['is_prime_time'] = self.schedule_df['start_hour'].apply(
            lambda x: self.prime_time_start <= x < self.prime_time_end
        )
    
    def _calculate_end_time(self, showtime, movie_id):
        """计算电影结束时间"""
        movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
        runtime = movie['rounded_runtime']
        
        hour, minute = map(int, showtime.split(':'))
        end_minute = minute + runtime
        end_hour = hour + end_minute // 60
        end_minute = end_minute % 60
        
        return f"{end_hour:02d}:{end_minute:02d}"
    
    def _time_to_minutes(self, time_str):
        """将时间字符串转换为分钟数"""
        hour, minute = map(int, time_str.split(':'))
        return hour * 60 + minute
    
    def analyze_time_conflicts(self):
        """分析时间冲突"""
        print("=== 时间冲突分析 ===")
        conflicts = []
        
        for room in self.schedule_df['room'].unique():
            room_schedule = self.schedule_df[self.schedule_df['room'] == room].copy()
            room_schedule = room_schedule.sort_values('showtime')
            
            for i in range(len(room_schedule) - 1):
                current = room_schedule.iloc[i]
                next_movie = room_schedule.iloc[i + 1]
                
                current_end = self._time_to_minutes(current['end_time'])
                next_start = self._time_to_minutes(next_movie['showtime'])
                
                # 检查是否有足够的15分钟间隔
                if next_start - current_end < 15:
                    conflicts.append({
                        'room': room,
                        'current_movie': current['id'],
                        'current_end': current['end_time'],
                        'next_movie': next_movie['id'],
                        'next_start': next_movie['showtime'],
                        'gap': next_start - current_end
                    })
        
        if conflicts:
            print(f"发现 {len(conflicts)} 个时间冲突:")
            for conflict in conflicts:
                print(f"  放映厅 {conflict['room']}: 电影 {conflict['current_movie']} 结束于 {conflict['current_end']}, "
                      f"电影 {conflict['next_movie']} 开始于 {conflict['next_start']}, 间隔仅 {conflict['gap']} 分钟")
        else:
            print("未发现时间冲突")
        
        return conflicts
    
    def analyze_genre_distribution(self):
        """分析题材分布"""
        print("\n=== 题材分布分析 ===")
        
        # 统计各题材播放次数
        genre_counts = defaultdict(int)
        movie_genres = {}
        
        for _, movie in self.movies_df.iterrows():
            genres = [g.strip() for g in movie['genres'].split(',')]
            movie_genres[movie['id']] = genres
        
        for _, schedule in self.schedule_df.iterrows():
            genres = movie_genres[schedule['id']]
            for genre in genres:
                genre_counts[genre] += 1
        
        print("各题材播放次数:")
        for genre, count in sorted(genre_counts.items()):
            if genre in self.genre_limits:
                limits = self.genre_limits[genre]
                min_limit = limits.get('min', 0)
                max_limit = limits.get('max', float('inf'))
                status = "✓" if min_limit <= count <= max_limit else "✗"
                print(f"  {genre}: {count} 次 (要求: {min_limit}-{max_limit}) {status}")
            else:
                print(f"  {genre}: {count} 次 (无限制)")
        
        return genre_counts
    
    def analyze_genre_time_constraints(self):
        """分析题材时间约束"""
        print("\n=== 题材时间约束分析 ===")
        violations = []
        
        movie_genres = {}
        for _, movie in self.movies_df.iterrows():
            genres = [g.strip() for g in movie['genres'].split(',')]
            movie_genres[movie['id']] = genres
        
        for _, schedule in self.schedule_df.iterrows():
            movie_id = schedule['id']
            start_hour = schedule['start_hour']
            genres = movie_genres[movie_id]
            
            for genre in genres:
                if genre in self.genre_time_limits:
                    constraints = self.genre_time_limits[genre]
                    
                    # 检查最早开始时间
                    if 'earliest_start' in constraints:
                        earliest = constraints['earliest_start']
                        if start_hour < earliest and start_hour >= 10:  # 当日时间但早于最早时间
                            violations.append({
                                'movie_id': movie_id,
                                'genre': genre,
                                'showtime': schedule['showtime'],
                                'violation': f"开始时间 {start_hour}:00 早于最早允许时间 {earliest}:00"
                            })
                    
                    # 检查最晚开始时间
                    if 'latest_start' in constraints:
                        latest = constraints['latest_start']
                        if start_hour >= latest and start_hour < 24:  # 当日时间但晚于最晚时间
                            violations.append({
                                'movie_id': movie_id,
                                'genre': genre,
                                'showtime': schedule['showtime'],
                                'violation': f"开始时间 {start_hour}:00 晚于最晚允许时间 {latest}:00"
                            })
        
        if violations:
            print(f"发现 {len(violations)} 个题材时间约束违规:")
            for violation in violations:
                print(f"  电影 {violation['movie_id']} ({violation['genre']}): {violation['showtime']} - {violation['violation']}")
        else:
            print("未发现题材时间约束违规")
        
        return violations
    
    def analyze_version_limits(self):
        """分析版本播放时长限制"""
        print("\n=== 版本播放时长限制分析 ===")
        
        version_durations = defaultdict(int)
        
        for _, schedule in self.schedule_df.iterrows():
            movie_id = schedule['id']
            version = schedule['version']
            movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
            runtime = movie['rounded_runtime']
            
            if version in ['3D', 'IMAX']:
                version_durations[version] += runtime
        
        violations = []
        for version, duration in version_durations.items():
            if version in self.version_limits:
                limits = self.version_limits[version]
                max_limit = limits['max']
                if duration > max_limit:
                    violations.append({
                        'version': version,
                        'duration': duration,
                        'limit': max_limit,
                        'violation': f"总播放时长 {duration} 分钟超过限制 {max_limit} 分钟"
                    })
        
        print("各版本播放时长:")
        for version, duration in version_durations.items():
            if version in self.version_limits:
                limit = self.version_limits[version]['max']
                status = "✓" if duration <= limit else "✗"
                print(f"  {version}: {duration} 分钟 (限制: ≤{limit} 分钟) {status}")
        
        if violations:
            print(f"\n发现 {len(violations)} 个版本时长限制违规:")
            for violation in violations:
                print(f"  {violation['version']}: {violation['violation']}")
        
        return violations
    
    def analyze_movie_distribution(self):
        """分析电影分布均衡性"""
        print("\n=== 电影分布均衡性分析 ===")
        
        # 统计每部电影的播放次数
        movie_counts = self.schedule_df['id'].value_counts()
        
        print("各电影播放次数:")
        for movie_id, count in movie_counts.items():
            movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
            rating = movie['rating']
            print(f"  电影 {movie_id} (评分: {rating:.2f}): {count} 次")
        
        # 计算统计信息
        mean_count = movie_counts.mean()
        std_count = movie_counts.std()
        
        print(f"\n统计信息:")
        print(f"  平均播放次数: {mean_count:.2f}")
        print(f"  标准差: {std_count:.2f}")
        print(f"  变异系数: {std_count/mean_count:.2f}")
        
        # 找出播放次数过多或过少的电影
        high_frequency = movie_counts[movie_counts > mean_count + std_count]
        low_frequency = movie_counts[movie_counts < mean_count - std_count]
        
        if not high_frequency.empty:
            print(f"\n播放次数过多的电影 (>{mean_count + std_count:.1f}):")
            for movie_id, count in high_frequency.items():
                movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                print(f"  电影 {movie_id} (评分: {movie['rating']:.2f}): {count} 次")
        
        if not low_frequency.empty:
            print(f"\n播放次数过少的电影 (<{mean_count - std_count:.1f}):")
            for movie_id, count in low_frequency.items():
                movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                print(f"  电影 {movie_id} (评分: {movie['rating']:.2f}): {count} 次")
        
        return movie_counts
    
    def analyze_room_utilization(self):
        """分析放映厅利用率"""
        print("\n=== 放映厅利用率分析 ===")
        
        room_utilization = {}
        
        for room in self.schedule_df['room'].unique():
            room_schedule = self.schedule_df[self.schedule_df['room'] == room]
            room_capacity = self.cinema_df[self.cinema_df['room'] == room]['capacity'].iloc[0]
            
            # 计算总放映时长
            total_runtime = 0
            for _, schedule in room_schedule.iterrows():
                movie_id = schedule['id']
                movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                runtime = movie['rounded_runtime']
                total_runtime += runtime
            
            # 计算可用时间 (17小时 = 1020分钟)
            available_time = 17 * 60
            
            # 计算利用率
            utilization = total_runtime / available_time * 100
            
            # 计算总观影人数
            total_attendance = room_schedule['attendance'].sum()
            
            # 计算理论最大观影人数
            max_possible_attendance = len(room_schedule) * room_capacity
            
            # 计算上座率
            attendance_rate = total_attendance / max_possible_attendance * 100
            
            room_utilization[room] = {
                'total_runtime': total_runtime,
                'utilization': utilization,
                'total_attendance': total_attendance,
                'attendance_rate': attendance_rate,
                'show_count': len(room_schedule)
            }
            
            print(f"放映厅 {room}:")
            print(f"  放映场次: {len(room_schedule)} 场")
            print(f"  总放映时长: {total_runtime} 分钟")
            print(f"  时间利用率: {utilization:.1f}%")
            print(f"  总观影人数: {total_attendance}")
            print(f"  上座率: {attendance_rate:.1f}%")
        
        return room_utilization
    
    def analyze_prime_time_distribution(self):
        """分析黄金时段分布"""
        print("\n=== 黄金时段分布分析 ===")
        
        prime_time_schedule = self.schedule_df[self.schedule_df['is_prime_time']]
        non_prime_schedule = self.schedule_df[~self.schedule_df['is_prime_time']]
        
        print(f"黄金时段场次: {len(prime_time_schedule)} 场 ({len(prime_time_schedule)/len(self.schedule_df)*100:.1f}%)")
        print(f"非黄金时段场次: {len(non_prime_schedule)} 场 ({len(non_prime_schedule)/len(self.schedule_df)*100:.1f}%)")
        
        # 统计黄金时段各电影播放情况
        prime_time_movies = prime_time_schedule['id'].value_counts()
        
        print("\n黄金时段电影分布:")
        for movie_id, count in prime_time_movies.items():
            movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
            rating = movie['rating']
            print(f"  电影 {movie_id} (评分: {rating:.2f}): {count} 次")
        
        return prime_time_schedule, non_prime_schedule
    
    def analyze_continuous_runtime_constraints(self):
        """分析连续运行时长限制"""
        print("\n=== 连续运行时长限制分析 ===")
        
        violations = []
        
        for room in self.schedule_df['room'].unique():
            room_schedule = self.schedule_df[self.schedule_df['room'] == room].copy()
            room_schedule = room_schedule.sort_values('showtime')
            
            # 检查每连续9小时窗口
            for window_start in range(10, 28):  # 从10:00到次日4:00
                window_end = window_start + 9
                
                # 计算窗口内的总播放时长
                window_runtime = 0
                for _, schedule in room_schedule.iterrows():
                    start_hour = schedule['start_hour']
                    movie_id = schedule['id']
                    movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
                    runtime = movie['rounded_runtime']
                    
                    # 计算电影结束时间
                    end_hour = start_hour + runtime // 60
                    
                    # 检查电影是否在窗口内播放
                    if not (end_hour <= window_start or start_hour >= window_end):
                        # 计算重叠时长
                        overlap_start = max(start_hour, window_start)
                        overlap_end = min(end_hour, window_end)
                        overlap_duration = (overlap_end - overlap_start) * 60  # 转换为分钟
                        
                        if overlap_duration > 0:
                            window_runtime += overlap_duration
                
                # 检查是否超过7小时限制
                if window_runtime > 7 * 60:
                    violations.append({
                        'room': room,
                        'window_start': window_start,
                        'window_end': window_end,
                        'runtime': window_runtime,
                        'violation': f"连续9小时内播放时长 {window_runtime} 分钟超过限制 420 分钟"
                    })
        
        if violations:
            print(f"发现 {len(violations)} 个连续运行时长限制违规:")
            for violation in violations:
                print(f"  放映厅 {violation['room']}: {violation['window_start']}:00-{violation['window_end']}:00 - {violation['violation']}")
        else:
            print("未发现连续运行时长限制违规")
        
        return violations
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("=== Q3排片方案综合分析报告 ===\n")
        
        # 执行各项分析
        time_conflicts = self.analyze_time_conflicts()
        genre_counts = self.analyze_genre_distribution()
        genre_time_violations = self.analyze_genre_time_constraints()
        version_violations = self.analyze_version_limits()
        movie_counts = self.analyze_movie_distribution()
        room_utilization = self.analyze_room_utilization()
        prime_time_schedule, non_prime_schedule = self.analyze_prime_time_distribution()
        runtime_violations = self.analyze_continuous_runtime_constraints()
        
        # 总结问题
        print("\n=== 问题总结 ===")
        issues = []
        
        if time_conflicts:
            issues.append(f"时间冲突: {len(time_conflicts)} 个")
        
        if genre_time_violations:
            issues.append(f"题材时间约束违规: {len(genre_time_violations)} 个")
        
        if version_violations:
            issues.append(f"版本时长限制违规: {len(version_violations)} 个")
        
        if runtime_violations:
            issues.append(f"连续运行时长限制违规: {len(runtime_violations)} 个")
        
        # 检查电影分布均衡性
        movie_counts_std = movie_counts.std()
        movie_counts_mean = movie_counts.mean()
        if movie_counts_std / movie_counts_mean > 0.5:  # 变异系数大于0.5认为不均衡
            issues.append("电影分布不均衡")
        
        # 检查黄金时段利用
        prime_time_ratio = len(prime_time_schedule) / len(self.schedule_df)
        if prime_time_ratio < 0.2:  # 黄金时段场次少于20%
            issues.append("黄金时段利用不足")
        
        if issues:
            print("发现以下问题:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("未发现明显问题")
        
        return {
            'time_conflicts': time_conflicts,
            'genre_counts': genre_counts,
            'genre_time_violations': genre_time_violations,
            'version_violations': version_violations,
            'movie_counts': movie_counts,
            'room_utilization': room_utilization,
            'runtime_violations': runtime_violations,
            'issues': issues
        }

# 使用示例
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = ScheduleAnalyzer(
        r'F:\MathModeling\input_data\df_cinema.csv',
        r'F:\MathModeling\input_data\df_movies_schedule_ours.csv',
        r'F:\MathModeling\Q3\df_result_2_copt_ours_new.csv'
    )
    
    # 生成综合分析报告
    report = analyzer.generate_comprehensive_report()