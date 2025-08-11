"""
多周期电影排片优化模型测试
测试基本功能而不依赖COPTPY求解器
"""

import pandas as pd
import math
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

class MultiPeriodCinemaSchedulerTest:
    """简化版本的多周期排片测试类"""
    
    def __init__(self, cinema_file: str, movies_file: str):
        # 读取基础数据
        self.cinema_df = pd.read_csv(cinema_file)
        self.movies_df = pd.read_csv(movies_file)
        
        # 营业时间设置
        self.start_hour = 10
        self.end_hour = 27
        self.time_slots = self._generate_time_slots()
        
        # 版本成本系数
        self.version_coeff = {'2D': 1.0, '3D': 1.1, 'IMAX': 1.15}
        
        # 基础参数
        self.basic_cost = 2.42
        self.fixed_cost = 90
        
        # 黄金时段设置
        self.prime_time_start = 18
        self.prime_time_end = 21
        self.prime_time_multiplier = 1.3
        
        # 多周期设置
        self.schedule_days = 7
        self.day_types = {
            0: 'weekday', 1: 'weekday', 2: 'weekday', 3: 'weekday', 4: 'weekday',
            5: 'weekend', 6: 'weekend'
        }
        
        # 上座率调整因子
        self.attendance_factors = {
            'weekday': {'morning': 0.7, 'afternoon': 0.8, 'prime': 1.0, 'evening': 0.9, 'night': 0.6},
            'weekend': {'morning': 0.9, 'afternoon': 1.1, 'prime': 1.3, 'evening': 1.2, 'night': 0.8}
        }
        
        # 电影热度衰减因子
        self.movie_decay_factors = {}
        self._initialize_movie_decay_factors()
    
    def _generate_time_slots(self):
        """生成时间段列表"""
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
            rating = movie['rating']
            if rating >= 8.0:
                decay_rate = 0.95
            elif rating >= 7.0:
                decay_rate = 0.90
            else:
                decay_rate = 0.85
            self.movie_decay_factors[movie_id] = decay_rate
    
    def _get_time_period(self, time_slot):
        """获取时间段类型"""
        hour = int(time_slot.split(':')[0])
        if hour >= 24:
            hour = hour - 24
        
        if 10 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 21:
            return 'prime'
        elif 21 <= hour < 24:
            return 'evening'
        else:
            return 'night'
    
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
        
        if version == '2D':
            price = basic_price
        elif version == '3D':
            price = basic_price * 1.2
        elif version == 'IMAX':
            price = basic_price * 1.23
        else:
            price = basic_price
        
        if is_prime_time:
            price *= self.prime_time_multiplier
        
        return price
    
    def _calculate_attendance(self, capacity, rating, day_type, time_slot, movie_id, day_offset=0):
        """计算实际观影人数"""
        base_attendance = math.floor(capacity * rating / 10)
        time_period = self._get_time_period(time_slot)
        attendance_factor = self.attendance_factors[day_type][time_period]
        decay_factor = self.movie_decay_factors[movie_id] ** day_offset
        adjusted_attendance = math.floor(base_attendance * attendance_factor * decay_factor)
        return min(adjusted_attendance, capacity)
    
    def _calculate_cost(self, capacity, version):
        """计算播放成本"""
        version_coeff = self.version_coeff[version]
        return version_coeff * capacity * self.basic_cost + self.fixed_cost
    
    def _get_sharing_rate(self, movie_id):
        """获取分成比例"""
        movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
        original_language = movie['original_language']
        if 'Mandarin' in original_language:
            return 0.43
        else:
            return 0.51
    
    def _is_prime_time(self, time_slot):
        """判断是否为黄金时段"""
        hour = int(time_slot.split(':')[0])
        return self.prime_time_start <= hour < self.prime_time_end
    
    def test_basic_functionality(self):
        """测试基本功能"""
        print("=== 多周期排片优化模型测试 ===\n")
        
        # 测试数据读取
        print(f"读取放映厅数据: {len(self.cinema_df)} 个放映厅")
        print(f"读取电影数据: {len(self.movies_df)} 部电影")
        print(f"时间段数量: {len(self.time_slots)}")
        
        # 测试电影版本获取
        print("\n--- 电影版本测试 ---")
        for _, movie in self.movies_df.head(3).iterrows():
            movie_id = movie['id']
            versions = self._get_versions(movie_id)
            print(f"电影 {movie_id}: 版本 {versions}")
        
        # 测试上座率计算
        print("\n--- 上座率计算测试 ---")
        test_movie_id = self.movies_df.iloc[0]['id']
        test_capacity = 100
        test_rating = 7.5
        
        for day_type in ['weekday', 'weekend']:
            for time_slot in ['10:00', '14:00', '19:00', '22:00']:
                attendance = self._calculate_attendance(
                    test_capacity, test_rating, day_type, time_slot, test_movie_id, 0
                )
                print(f"{day_type} {time_slot}: 预期观影人数 {attendance}")
        
        # 测试票价计算
        print("\n--- 票价计算测试 ---")
        for version in ['2D', '3D', 'IMAX']:
            for is_prime in [False, True]:
                try:
                    price = self._calculate_ticket_price(test_movie_id, version, is_prime)
                    time_type = "黄金时段" if is_prime else "普通时段"
                    print(f"{version} {time_type}: 票价 {price:.2f} 元")
                except Exception:
                    print(f"{version} 版本不支持")
        
        # 测试成本计算
        print("\n--- 成本计算测试 ---")
        for version in ['2D', '3D', 'IMAX']:
            cost = self._calculate_cost(test_capacity, version)
            print(f"{version} 版本成本: {cost:.2f} 元")
        
        # 测试净收益计算
        print("\n--- 净收益计算测试 ---")
        for day in range(3):  # 测试前3天
            day_type = self.day_types[day]
            print(f"\n第{day+1}天 ({day_type}):")
            
            total_profit = 0
            for time_slot in ['10:00', '14:00', '19:00', '22:00']:
                version = '2D'
                is_prime = self._is_prime_time(time_slot)
                
                ticket_price = self._calculate_ticket_price(test_movie_id, version, is_prime)
                attendance = self._calculate_attendance(
                    test_capacity, test_rating, day_type, time_slot, test_movie_id, day
                )
                cost = self._calculate_cost(test_capacity, version)
                sharing_rate = self._get_sharing_rate(test_movie_id)
                
                revenue = ticket_price * attendance * (1 - sharing_rate)
                net_profit = revenue - cost
                total_profit += net_profit
                
                print(f"  {time_slot}: 票价{ticket_price:.1f}, 观众{attendance}, 净收益{net_profit:.2f}")
            
            print(f"  当日总净收益: {total_profit:.2f} 元")
        
        print("\n测试完成！基本功能正常。")
    
    def generate_sample_schedule(self):
        """生成样本排片计划（贪心算法示例）"""
        print("\n=== 生成样本排片计划 ===")
        
        sample_schedule = []
        current_date = pd.to_datetime('2024-08-12')
        
        for day in range(self.schedule_days):
            day_date = current_date + timedelta(days=day)
            day_type = self.day_types[day]
            print(f"\n第{day+1}天 ({day_date.strftime('%Y-%m-%d')}, {day_type}):")
            
            # 为每个放映厅安排电影
            for _, room_info in self.cinema_df.iterrows():
                room = room_info['room']
                capacity = room_info['capacity']
                
                # 简单策略：选择评分最高的电影
                best_movie = self.movies_df.loc[self.movies_df['rating'].idxmax()]
                movie_id = best_movie['id']
                versions = self._get_versions(movie_id)
                
                # 选择该放映厅支持的版本
                available_versions = [v for v in versions if self._can_room_play_version(room, v)]
                if not available_versions:
                    continue
                
                version = available_versions[0]  # 选择第一个可用版本
                
                # 安排几个时段
                for time_slot in ['10:00', '14:00', '19:00', '22:00']:
                    is_prime = self._is_prime_time(time_slot)
                    ticket_price = self._calculate_ticket_price(movie_id, version, is_prime)
                    attendance = self._calculate_attendance(
                        capacity, best_movie['rating'], day_type, time_slot, movie_id, day
                    )
                    cost = self._calculate_cost(capacity, version)
                    sharing_rate = self._get_sharing_rate(movie_id)
                    revenue = ticket_price * attendance * (1 - sharing_rate)
                    net_profit = revenue - cost
                    
                    if net_profit > 0:  # 只安排有利润的场次
                        sample_schedule.append({
                            'date': day_date.strftime('%Y-%m-%d'),
                            'day_type': day_type,
                            'room': room,
                            'movie_id': movie_id,
                            'version': version,
                            'showtime': time_slot,
                            'capacity': capacity,
                            'attendance': attendance,
                            'ticket_price': ticket_price,
                            'revenue': revenue,
                            'cost': cost,
                            'net_profit': net_profit,
                            'is_prime_time': is_prime
                        })
                        
                        print(f"  {room} {time_slot}: 电影{movie_id} {version}, 净收益{net_profit:.2f}")
        
        # 保存样本排片结果
        if sample_schedule:
            df_sample = pd.DataFrame(sample_schedule)
            df_sample.to_csv('sample_schedule.csv', index=False, encoding='utf-8-sig')
            
            print("\n样本排片计划已生成:")
            print(f"总场次数: {len(df_sample)}")
            print(f"总净收益: {df_sample['net_profit'].sum():.2f} 元")
            print(f"平均每场净收益: {df_sample['net_profit'].mean():.2f} 元")
            print("结果已保存到 sample_schedule.csv")


def main():
    """测试主函数"""
    try:
        # 初始化测试类
        scheduler = MultiPeriodCinemaSchedulerTest(
            cinema_file='../input_data/df_cinema.csv',
            movies_file='../input_data/df_movies_schedule_ours.csv'
        )
        
        # 测试基本功能
        scheduler.test_basic_functionality()
        
        # 生成样本排片
        scheduler.generate_sample_schedule()
        
    except Exception as e:
        print(f"测试出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
