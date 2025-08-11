import csv
import math
from collections import defaultdict

class SimpleScheduleAnalyzer:
    def __init__(self, cinema_file, movies_file, schedule_file):
        """
        初始化排片分析器
        """
        # 读取CSV文件
        self.cinema_data = self._read_csv(cinema_file)
        self.movies_data = self._read_csv(movies_file)
        self.schedule_data = self._read_csv(schedule_file)
        
        # 营业时间设置
        self.start_hour = 10
        self.end_hour = 27  # 次日3点用27表示
        
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
        
        # 预处理数据
        self._preprocess_data()
    
    def _read_csv(self, filename):
        """读取CSV文件"""
        data = []
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        return data
    
    def _preprocess_data(self):
        """预处理数据"""
        # 为电影数据添加时长取整
        for movie in self.movies_data:
            runtime = float(movie['runtime'])
            movie['rounded_runtime'] = math.ceil(runtime / 30) * 30
        
        # 为排片数据添加时间解析
        for schedule in self.schedule_data:
            time_parts = schedule['showtime'].split(':')
            schedule['start_hour'] = int(time_parts[0])
            schedule['start_minute'] = int(time_parts[1])
            
            # 计算结束时间
            movie_id = schedule['id']
            movie = next(m for m in self.movies_data if m['id'] == movie_id)
            runtime = movie['rounded_runtime']
            
            end_hour = schedule['start_hour'] + runtime // 60
            end_minute = schedule['start_minute'] + runtime % 60
            if end_minute >= 60:
                end_hour += 1
                end_minute -= 60
            
            schedule['end_hour'] = end_hour
            schedule['end_minute'] = end_minute
            schedule['end_time'] = f"{end_hour:02d}:{end_minute:02d}"
            
            # 标记黄金时段
            schedule['is_prime_time'] = self.prime_time_start <= schedule['start_hour'] < self.prime_time_end
    
    def _time_to_minutes(self, hour, minute):
        """将时间转换为分钟数"""
        return hour * 60 + minute
    
    def analyze_time_conflicts(self):
        """分析时间冲突"""
        print("=== 时间冲突分析 ===")
        conflicts = []
        
        # 按放映厅分组
        room_schedules = defaultdict(list)
        for schedule in self.schedule_data:
            room_schedules[schedule['room']].append(schedule)
        
        for room, schedules in room_schedules.items():
            # 按开始时间排序
            schedules.sort(key=lambda x: (x['start_hour'], x['start_minute']))
            
            for i in range(len(schedules) - 1):
                current = schedules[i]
                next_movie = schedules[i + 1]
                
                current_end = self._time_to_minutes(current['end_hour'], current['end_minute'])
                next_start = self._time_to_minutes(next_movie['start_hour'], next_movie['start_minute'])
                
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
        
        for movie in self.movies_data:
            genres = [g.strip() for g in movie['genres'].split(',')]
            movie_genres[movie['id']] = genres
        
        for schedule in self.schedule_data:
            genres = movie_genres[schedule['id']]
            for genre in genres:
                genre_counts[genre] += 1
        
        print("各题材播放次数:")
        for genre, count in sorted(genre_counts.items()):
            if genre in self.genre_limits:
                limits = self.genre_limits[genre]
                min_limit = limits.get('min', 0)
                max_limit = limits.get('max', float('inf'))
                status = "OK" if min_limit <= count <= max_limit else "NG"
                print(f"  {genre}: {count} 次 (要求: {min_limit}-{max_limit}) {status}")
            else:
                print(f"  {genre}: {count} 次 (无限制)")
        
        return genre_counts
    
    def analyze_genre_time_constraints(self):
        """分析题材时间约束"""
        print("\n=== 题材时间约束分析 ===")
        violations = []
        
        movie_genres = {}
        for movie in self.movies_data:
            genres = [g.strip() for g in movie['genres'].split(',')]
            movie_genres[movie['id']] = genres
        
        for schedule in self.schedule_data:
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
        
        for schedule in self.schedule_data:
            movie_id = schedule['id']
            version = schedule['version']
            movie = next(m for m in self.movies_data if m['id'] == movie_id)
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
                status = "OK" if duration <= limit else "NG"
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
        movie_counts = defaultdict(int)
        for schedule in self.schedule_data:
            movie_counts[schedule['id']] += 1
        
        print("各电影播放次数:")
        for movie_id, count in movie_counts.items():
            movie = next(m for m in self.movies_data if m['id'] == movie_id)
            rating = float(movie['rating'])
            print(f"  电影 {movie_id} (评分: {rating:.2f}): {count} 次")
        
        # 计算统计信息
        counts = list(movie_counts.values())
        mean_count = sum(counts) / len(counts)
        variance = sum((x - mean_count) ** 2 for x in counts) / len(counts)
        std_count = math.sqrt(variance)
        
        print("\n统计信息:")
        print(f"  平均播放次数: {mean_count:.2f}")
        print(f"  标准差: {std_count:.2f}")
        print(f"  变异系数: {std_count/mean_count:.2f}")
        
        # 找出播放次数过多或过少的电影
        high_frequency = {mid: cnt for mid, cnt in movie_counts.items() if cnt > mean_count + std_count}
        low_frequency = {mid: cnt for mid, cnt in movie_counts.items() if cnt < mean_count - std_count}
        
        if high_frequency:
            print(f"\n播放次数过多的电影 (>{mean_count + std_count:.1f}):")
            for movie_id, count in high_frequency.items():
                movie = next(m for m in self.movies_data if m['id'] == movie_id)
                print(f"  电影 {movie_id} (评分: {movie['rating']}): {count} 次")
        
        if low_frequency:
            print(f"\n播放次数过少的电影 (<{mean_count - std_count:.1f}):")
            for movie_id, count in low_frequency.items():
                movie = next(m for m in self.movies_data if m['id'] == movie_id)
                print(f"  电影 {movie_id} (评分: {movie['rating']}): {count} 次")
        
        return movie_counts
    
    def analyze_room_utilization(self):
        """分析放映厅利用率"""
        print("\n=== 放映厅利用率分析 ===")
        
        room_utilization = {}
        
        # 按放映厅分组
        room_schedules = defaultdict(list)
        for schedule in self.schedule_data:
            room_schedules[schedule['room']].append(schedule)
        
        for room, schedules in room_schedules.items():
            # 获取放映厅容量
            cinema = next(c for c in self.cinema_data if c['room'] == room)
            room_capacity = int(cinema['capacity'])
            
            # 计算总放映时长
            total_runtime = 0
            for schedule in schedules:
                movie_id = schedule['id']
                movie = next(m for m in self.movies_data if m['id'] == movie_id)
                runtime = movie['rounded_runtime']
                total_runtime += runtime
            
            # 计算可用时间 (17小时 = 1020分钟)
            available_time = 17 * 60
            
            # 计算利用率
            utilization = total_runtime / available_time * 100
            
            # 计算总观影人数
            total_attendance = sum(int(schedule['attendance']) for schedule in schedules)
            
            # 计算理论最大观影人数
            max_possible_attendance = len(schedules) * room_capacity
            
            # 计算上座率
            attendance_rate = total_attendance / max_possible_attendance * 100
            
            room_utilization[room] = {
                'total_runtime': total_runtime,
                'utilization': utilization,
                'total_attendance': total_attendance,
                'attendance_rate': attendance_rate,
                'show_count': len(schedules)
            }
            
            print(f"放映厅 {room}:")
            print(f"  放映场次: {len(schedules)} 场")
            print(f"  总放映时长: {total_runtime} 分钟")
            print(f"  时间利用率: {utilization:.1f}%")
            print(f"  总观影人数: {total_attendance}")
            print(f"  上座率: {attendance_rate:.1f}%")
        
        return room_utilization
    
    def analyze_prime_time_distribution(self):
        """分析黄金时段分布"""
        print("\n=== 黄金时段分布分析 ===")
        
        prime_time_count = sum(1 for schedule in self.schedule_data if schedule['is_prime_time'])
        non_prime_count = len(self.schedule_data) - prime_time_count
        
        print(f"黄金时段场次: {prime_time_count} 场 ({prime_time_count/len(self.schedule_data)*100:.1f}%)")
        print(f"非黄金时段场次: {non_prime_count} 场 ({non_prime_count/len(self.schedule_data)*100:.1f}%)")
        
        # 统计黄金时段各电影播放情况
        prime_time_movies = defaultdict(int)
        for schedule in self.schedule_data:
            if schedule['is_prime_time']:
                prime_time_movies[schedule['id']] += 1
        
        print("\n黄金时段电影分布:")
        for movie_id, count in prime_time_movies.items():
            movie = next(m for m in self.movies_data if m['id'] == movie_id)
            rating = float(movie['rating'])
            print(f"  电影 {movie_id} (评分: {rating:.2f}): {count} 次")
        
        return prime_time_count, non_prime_count
    
    def analyze_continuous_runtime_constraints(self):
        """分析连续运行时长限制"""
        print("\n=== 连续运行时长限制分析 ===")
        
        violations = []
        
        # 按放映厅分组
        room_schedules = defaultdict(list)
        for schedule in self.schedule_data:
            room_schedules[schedule['room']].append(schedule)
        
        for room, schedules in room_schedules.items():
            # 按开始时间排序
            schedules.sort(key=lambda x: (x['start_hour'], x['start_minute']))
            
            # 检查每连续9小时窗口
            for window_start in range(10, 28):  # 从10:00到次日4:00
                window_end = window_start + 9
                
                # 计算窗口内的总播放时长
                window_runtime = 0
                for schedule in schedules:
                    start_hour = schedule['start_hour']
                    movie_id = schedule['id']
                    movie = next(m for m in self.movies_data if m['id'] == movie_id)
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
        prime_time_count, non_prime_count = self.analyze_prime_time_distribution()
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
        counts = list(movie_counts.values())
        mean_count = sum(counts) / len(counts)
        variance = sum((x - mean_count) ** 2 for x in counts) / len(counts)
        std_count = math.sqrt(variance)
        if std_count / mean_count > 0.5:  # 变异系数大于0.5认为不均衡
            issues.append("电影分布不均衡")
        
        # 检查黄金时段利用
        prime_time_ratio = prime_time_count / len(self.schedule_data)
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
    analyzer = SimpleScheduleAnalyzer(
        r'F:\MathModeling\input_data\df_cinema.csv',
        r'F:\MathModeling\input_data\df_movies_schedule_ours.csv',
        r'F:\MathModeling\Q3\df_result_2_copt_ours_new.csv'
    )
    
    # 生成综合分析报告
    report = analyzer.generate_comprehensive_report()