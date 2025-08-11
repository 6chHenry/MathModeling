import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, date
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class GridExperimentAnalyzer:
    """
    网格实验分析器
    
    提供全面的网格实验功能，包括：
    - 关键参数（λ, α, ρ）的全面网格实验
    - 一周滚动回测评估时间稳定性
    - 紧急场景响应测试
    """
    
    def __init__(self, cinema_file, movies_file, historical_attendance_file=None):
        """
        初始化网格实验分析器
        
        Args:
            cinema_file: 放映厅信息文件路径
            movies_file: 电影信息文件路径
            historical_attendance_file: 历史上座率数据文件路径（可选）
        """
        self.cinema_file = cinema_file
        self.movies_file = movies_file
        self.historical_attendance_file = historical_attendance_file
        
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
        
        # 关键参数（用于网格实验）
        self.lambda_param = 1.0  # 收益权重
        self.alpha_param = 0.5   # 上座率权重
        self.rho_param = 0.3     # 多样性权重
        
        # 存储实验结果
        self.grid_results = []
        self.backtest_results = []
        self.emergency_results = []
        
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
        return int(((runtime + 29) // 30) * 30)
    
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
            price *= 1.3
        
        return price
    
    def _calculate_attendance(self, capacity, rating, time_slot, movie_id, day_offset=0):
        """计算实际观影人数"""
        # 基础上座率
        base_attendance = int(capacity * rating / 10)
        
        # 获取时间段类型
        time_period = self._get_time_period(time_slot)
        
        # 应用上座率因子（简化版）
        attendance_factors = {
            'morning': 0.8,
            'afternoon': 0.9,
            'prime': 1.0,
            'evening': 0.9,
            'night': 0.7
        }
        
        attendance_factor = attendance_factors.get(time_period, 0.8)
        
        # 应用电影热度衰减（基于天数偏移）
        decay_factor = 0.95 ** day_offset
        
        # 计算最终上座率
        adjusted_attendance = int(base_attendance * attendance_factor * decay_factor)
        
        # 确保不超过容量
        return min(adjusted_attendance, capacity)
    
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
    
    def _is_prime_time(self, time_slot):
        """判断是否为黄金时段"""
        hour = int(time_slot.split(':')[0])
        return 18 <= hour < 21
    
    def _get_sharing_rate(self, movie_id):
        """获取分成比例"""
        movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
        original_language = movie['original_language']
        # 如果包含普通话则为国产电影
        if 'Mandarin' in original_language:
            return 0.43
        else:
            return 0.51
    
    def _calculate_objective_value(self, schedule_results):
        """计算目标函数值"""
        if not schedule_results:
            return 0
        
        df_schedule = pd.DataFrame(schedule_results)
        
        # 计算各项指标
        total_revenue = df_schedule['revenue'].sum()
        total_attendance = df_schedule['attendance'].sum()
        
        # 计算多样性指标（不同电影的数量）
        movie_diversity = df_schedule['movie_id'].nunique()
        
        # 计算加权目标函数
        objective = (
            self.lambda_param * total_revenue +
            self.alpha_param * total_attendance +
            self.rho_param * movie_diversity
        )
        
        return objective
    
    def parameter_grid_experiment(self, lambda_range=[0.5, 1.0, 1.5], 
                                alpha_range=[0.3, 0.5, 0.7], 
                                rho_range=[0.2, 0.3, 0.4]):
        """关键参数网格实验"""
        print("执行关键参数网格实验...")
        
        results = []
        
        # 生成参数组合
        from itertools import product
        param_combinations = list(product(lambda_range, alpha_range, rho_range))
        
        print(f"共 {len(param_combinations)} 种参数组合需要测试")
        
        for i, (lambda_val, alpha_val, rho_val) in enumerate(param_combinations):
            # 设置参数
            self.lambda_param = lambda_val
            self.alpha_param = alpha_val
            self.rho_param = rho_val
            
            # 生成排片方案（简化版）
            schedule_results = self._generate_simplified_schedule()
            
            # 计算目标函数值
            objective_value = self._calculate_objective_value(schedule_results)
            
            # 计算其他指标
            if schedule_results:
                df_schedule = pd.DataFrame(schedule_results)
                total_revenue = df_schedule['revenue'].sum()
                total_attendance = df_schedule['attendance'].sum()
                movie_diversity = df_schedule['movie_id'].nunique()
                avg_profit = df_schedule['net_profit'].mean()
            else:
                total_revenue = 0
                total_attendance = 0
                movie_diversity = 0
                avg_profit = 0
            
            results.append({
                'lambda': lambda_val,
                'alpha': alpha_val,
                'rho': rho_val,
                'objective_value': objective_value,
                'total_revenue': total_revenue,
                'total_attendance': total_attendance,
                'movie_diversity': movie_diversity,
                'avg_profit': avg_profit
            })
            
            if (i + 1) % 5 == 0:
                print(f"已完成 {i + 1}/{len(param_combinations)} 种参数组合测试")
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        self.grid_results = results_df
        
        # 可视化结果
        self._visualize_grid_results(results_df)
        
        # 保存结果
        results_df.to_csv('Q4/parameter_grid_results.csv', index=False)
        
        print("关键参数网格实验完成，结果已保存")
        
        return results_df
    
    def _generate_simplified_schedule(self):
        """生成简化版排片方案"""
        schedule_results = []
        
        # 为每个放映厅生成排