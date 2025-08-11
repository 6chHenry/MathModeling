import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
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
    
    def _calculate_cost(self, capacity, version):
        """计算播放成本"""
        version_coeff = self.version_coeff[version]
        return version_coeff * capacity * self.basic_cost + self.fixed_cost
    
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
        results_df.to_csv('parameter_grid_results.csv', index=False)
        
        print("关键参数网格实验完成，结果已保存")
        
        return results_df
    
    def _generate_simplified_schedule(self):
        """生成简化版排片方案"""
        schedule_results = []
        
        # 为每个放映厅生成排片
        for _, room_info in self.cinema_df.iterrows():
            room = room_info['room']
            capacity = room_info['capacity']
            
            # 为该厅选择3-5部电影
            available_movies = self.movies_df.head(5)  # 简化处理，取前5部电影
            
            for _, movie in available_movies.iterrows():
                movie_id = movie['id']
                versions = self._get_versions(movie_id)
                
                # 选择该厅支持的版本
                supported_versions = [v for v in versions if self._can_room_play_version(room, v)]
                
                if supported_versions:
                    version = supported_versions[0]  # 选择第一个支持的版本
                    
                    # 生成2-3个时间段
                    time_slots = ['10:00', '14:00', '18:00'][:2]  # 简化处理
                    
                    for time_slot in time_slots:
                        # 计算相关指标
                        is_prime = self._is_prime_time(time_slot)
                        ticket_price = self._calculate_ticket_price(movie_id, version, is_prime)
                        attendance = self._calculate_attendance(capacity, movie['rating'], time_slot, movie_id)
                        cost = self._calculate_cost(capacity, version)
                        sharing_rate = self._get_sharing_rate(movie_id)
                        revenue = ticket_price * attendance * (1 - sharing_rate)
                        net_profit = revenue - cost
                        
                        schedule_results.append({
                            'room': room,
                            'movie_id': movie_id,
                            'movie_title': movie.get('title', f'Movie_{movie_id}'),
                            'version': version,
                            'showtime': time_slot,
                            'capacity': capacity,
                            'attendance': attendance,
                            'ticket_price': ticket_price,
                            'revenue': revenue,
                            'cost': cost,
                            'net_profit': net_profit,
                            'sharing_rate': sharing_rate,
                            'is_prime_time': is_prime
                        })
        
        return schedule_results
    
    def _visualize_grid_results(self, results_df):
        """可视化网格实验结果"""
        # 创建多个子图
        plt.figure(figsize=(20, 15))
        
        # 1. 目标函数值热力图（固定rho）
        ax1 = plt.subplot(2, 3, 1)
        pivot_data = results_df[results_df['rho'] == 0.3].pivot(index='lambda', columns='alpha', values='objective_value')
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax1)
        ax1.set_title('目标函数值热力图 (ρ=0.3)')
        
        # 2. 总收入热力图（固定rho）
        ax2 = plt.subplot(2, 3, 2)
        pivot_data = results_df[results_df['rho'] == 0.3].pivot(index='lambda', columns='alpha', values='total_revenue')
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax2)
        ax2.set_title('总收入热力图 (ρ=0.3)')
        
        # 3. 电影多样性热力图（固定rho）
        ax3 = plt.subplot(2, 3, 3)
        pivot_data = results_df[results_df['rho'] == 0.3].pivot(index='lambda', columns='alpha', values='movie_diversity')
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax3)
        ax3.set_title('电影多样性热力图 (ρ=0.3)')
        
        # 4. 参数对目标函数的影响
        ax4 = plt.subplot(2, 3, 4)
        for lambda_val in results_df['lambda'].unique():
            lambda_data = results_df[results_df['lambda'] == lambda_val]
            ax4.plot(lambda_data['alpha'], lambda_data['objective_value'], 
                    marker='o', label=f'λ={lambda_val}')
        ax4.set_xlabel('α (上座率权重)')
        ax4.set_ylabel('目标函数值')
        ax4.set_title('α对目标函数的影响')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 参数对总收入的影响
        ax5 = plt.subplot(2, 3, 5)
        for rho_val in results_df['rho'].unique():
            rho_data = results_df[results_df['rho'] == rho_val]
            ax5.plot(rho_data['lambda'], rho_data['total_revenue'], 
                    marker='o', label=f'ρ={rho_val}')
        ax5.set_xlabel('λ (收益权重)')
        ax5.set_ylabel('总收入')
        ax5.set_title('λ对总收入的影响')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 最佳参数组合
        ax6 = plt.subplot(2, 3, 6)
        best_result = results_df.loc[results_df['objective_value'].idxmax()]
        param_names = ['λ', 'α', 'ρ']
        param_values = [best_result['lambda'], best_result['alpha'], best_result['rho']]
        ax6.bar(param_names, param_values, color=['skyblue', 'lightgreen', 'salmon'])
        ax6.set_ylabel('参数值')
        ax6.set_title(f'最佳参数组合\n目标函数值: {best_result["objective_value"]:.0f}')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('parameter_grid_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def rolling_backtest(self, start_date='2024-08-12', days=7):
        """一周滚动回测"""
        print(f"执行一周滚动回测（{days}天）...")
        
        results = []
        
        # 转换开始日期
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        
        # 执行滚动回测
        for day_offset in range(days):
            current_date = start_dt + timedelta(days=day_offset)
            
            # 生成当日排片方案
            daily_schedule = self._generate_daily_schedule(current_date)
            
            # 计算当日指标
            daily_metrics = self._calculate_daily_metrics(daily_schedule, day_offset)
            
            results.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'day_offset': day_offset,
                **daily_metrics
            })
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        self.backtest_results = results_df
        
        # 可视化结果
        self._visualize_backtest_results(results_df)
        
        # 保存结果
        results_df.to_csv('/rolling_backtest_results.csv', index=False)
        
        print("一周滚动回测完成，结果已保存")
        
        return results_df
    
    def _generate_daily_schedule(self, date):
        """生成指定日期的排片方案"""
        schedule_results = []
        
        # 根据日期调整参数（模拟时间效应）
        day_of_week = date.weekday()
        if day_of_week >= 5:  # 周末
            self.lambda_param = 1.2  # 周末更注重收益
            self.alpha_param = 0.4
        else:  # 工作日
            self.lambda_param = 0.8  # 工作日更注重上座率
            self.alpha_param = 0.6
        
        # 生成排片
        for _, room_info in self.cinema_df.iterrows():
            room = room_info['room']
            capacity = room_info['capacity']
            
            # 根据日期选择电影（模拟电影热度变化）
            if day_of_week >= 5:
                # 周末选择更受欢迎的电影
                selected_movies = self.movies_df.sort_values('rating', ascending=False).head(4)
            else:
                # 工作日选择多样化电影
                selected_movies = self.movies_df.head(6)
            
            for _, movie in selected_movies.iterrows():
                movie_id = movie['id']
                versions = self._get_versions(movie_id)
                
                # 选择该厅支持的版本
                supported_versions = [v for v in versions if self._can_room_play_version(room, v)]
                
                if supported_versions:
                    version = supported_versions[0]
                    
                    # 根据日期选择时间段
                    if day_of_week >= 5:
                        # 周末更多黄金时段
                        time_slots = ['10:00', '13:00', '16:00', '19:00', '22:00']
                    else:
                        # 工作日更均衡分布
                        time_slots = ['10:30', '14:00', '17:30', '21:00']
                    
                    for time_slot in time_slots:
                        # 计算相关指标
                        is_prime = self._is_prime_time(time_slot)
                        ticket_price = self._calculate_ticket_price(movie_id, version, is_prime)
                        attendance = self._calculate_attendance(capacity, movie['rating'], time_slot, movie_id, day_offset=day_of_week)
                        cost = self._calculate_cost(capacity, version)
                        sharing_rate = self._get_sharing_rate(movie_id)
                        revenue = ticket_price * attendance * (1 - sharing_rate)
                        net_profit = revenue - cost
                        
                        schedule_results.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'day_of_week': day_of_week,
                            'room': room,
                            'movie_id': movie_id,
                            'movie_title': movie.get('title', f'Movie_{movie_id}'),
                            'version': version,
                            'showtime': time_slot,
                            'capacity': capacity,
                            'attendance': attendance,
                            'ticket_price': ticket_price,
                            'revenue': revenue,
                            'cost': cost,
                            'net_profit': net_profit,
                            'sharing_rate': sharing_rate,
                            'is_prime_time': is_prime
                        })
        
        return schedule_results
    
    def _calculate_daily_metrics(self, daily_schedule, day_offset):
        """计算每日指标"""
        if not daily_schedule:
            return {
                'total_shows': 0,
                'total_revenue': 0,
                'total_attendance': 0,
                'total_profit': 0,
                'avg_profit_per_show': 0,
                'prime_time_ratio': 0,
                'movie_diversity': 0,
                'occupancy_rate': 0
            }
        
        df_schedule = pd.DataFrame(daily_schedule)
        
        # 计算各项指标
        total_shows = len(df_schedule)
        total_revenue = df_schedule['revenue'].sum()
        total_attendance = df_schedule['attendance'].sum()
        total_profit = df_schedule['net_profit'].sum()
        avg_profit_per_show = total_profit / total_shows if total_shows > 0 else 0
        
        # 黄金时段比例
        prime_shows = df_schedule['is_prime_time'].sum()
        prime_time_ratio = prime_shows / total_shows if total_shows > 0 else 0
        
        # 电影多样性
        movie_diversity = df_schedule['movie_id'].nunique()
        
        # 上座率
        total_capacity = df_schedule['capacity'].sum()
        occupancy_rate = total_attendance / total_capacity if total_capacity > 0 else 0
        
        return {
            'total_shows': total_shows,
            'total_revenue': total_revenue,
            'total_attendance': total_attendance,
            'total_profit': total_profit,
            'avg_profit_per_show': avg_profit_per_show,
            'prime_time_ratio': prime_time_ratio,
            'movie_diversity': movie_diversity,
            'occupancy_rate': occupancy_rate
        }
    
    def _visualize_backtest_results(self, results_df):
        """可视化回测结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 每日收益趋势
        axes[0, 0].plot(results_df['date'], results_df['total_revenue'], marker='o', linewidth=2)
        axes[0, 0].set_title('每日收益趋势')
        axes[0, 0].set_ylabel('收益 (元)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 每日上座率趋势
        axes[0, 1].plot(results_df['date'], results_df['occupancy_rate'], marker='o', linewidth=2, color='green')
        axes[0, 1].set_title('每日上座率趋势')
        axes[0, 1].set_ylabel('上座率')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 每日场次和电影多样性
        ax3 = axes[1, 0]
        ax3_twin = ax3.twinx()
        
        ax3.bar(results_df['date'], results_df['total_shows'], alpha=0.7, label='总场次')
        ax3_twin.plot(results_df['date'], results_df['movie_diversity'], marker='o', color='red', label='电影多样性')
        
        ax3.set_title('每日场次和电影多样性')
        ax3.set_ylabel('总场次')
        ax3_twin.set_ylabel('电影多样性')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. 黄金时段比例和平均利润
        ax4 = axes[1, 1]
        ax4_twin = ax4.twinx()
        
        ax4.plot(results_df['date'], results_df['prime_time_ratio'], marker='o', color='purple', label='黄金时段比例')
        ax4_twin.plot(results_df['date'], results_df['avg_profit_per_show'], marker='s', color='orange', label='平均利润')
        
        ax4.set_title('黄金时段比例和平均利润')
        ax4.set_ylabel('黄金时段比例')
        ax4_twin.set_ylabel('平均利润 (元)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rolling_backtest_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def emergency_scenario_test(self):
        """紧急场景响应测试"""
        print("执行紧急场景响应测试...")
        
        # 定义紧急场景
        emergency_scenarios = [
            {
                'name': '高需求周末',
                'description': '周末突然出现高需求',
                'demand_multiplier': 1.5,
                'price_multiplier': 1.2
            },
            {
                'name': '设备故障',
                'description': '一个放映厅设备故障',
                'available_rooms': 0.8,  # 只有80%的放映厅可用
                'demand_multiplier': 1.0,
                'price_multiplier': 1.0
            },
            {
                'name': '新片上映',
                'description': '热门新片突然上映',
                'new_movies': 2,
                'demand_multiplier': 1.3,
                'price_multiplier': 1.1
            },
            {
                'name': '恶劣天气',
                'description': '恶劣天气影响观影',
                'demand_multiplier': 0.7,
                'price_multiplier': 0.9
            },
            {
                'name': '竞争影院活动',
                'description': '竞争影院举办优惠活动',
                'demand_multiplier': 0.8,
                'price_multiplier': 0.85
            }
        ]
        
        results = []
        
        # 基准场景
        baseline_schedule = self._generate_baseline_schedule()
        baseline_metrics = self._calculate_scenario_metrics(baseline_schedule)
        
        # 测试每个紧急场景
        for scenario in emergency_scenarios:
            print(f"测试场景: {scenario['name']}")
            
            # 生成场景排片方案
            scenario_schedule = self._generate_scenario_schedule(scenario)
            
            # 计算场景指标
            scenario_metrics = self._calculate_scenario_metrics(scenario_schedule)
            
            # 计算相对于基准的变化
            changes = self._calculate_scenario_changes(baseline_metrics, scenario_metrics)
            
            results.append({
                'scenario_name': scenario['name'],
                'scenario_description': scenario['description'],
                **baseline_metrics,
                **{f'scenario_{k}': v for k, v in scenario_metrics.items()},
                **{f'change_{k}': v for k, v in changes.items()}
            })
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        self.emergency_results = results_df
        
        # 可视化结果
        self._visualize_emergency_results(results_df)
        
        # 保存结果
        results_df.to_csv('emergency_scenario_results.csv', index=False)
        
        print("紧急场景响应测试完成，结果已保存")
        
        return results_df
    
    def _generate_baseline_schedule(self):
        """生成基准排片方案"""
        # 恢复默认参数
        self.lambda_param = 1.0
        self.alpha_param = 0.5
        self.rho_param = 0.3
        
        return self._generate_simplified_schedule()
    
    def _generate_scenario_schedule(self, scenario):
        """生成场景排片方案"""
        schedule_results = []
        
        # 根据场景调整参数
        if scenario['name'] == '高需求周末':
            self.lambda_param = 1.5  # 更注重收益
            self.alpha_param = 0.3
            
        elif scenario['name'] == '设备故障':
            # 减少可用放映厅
            available_rooms = int(len(self.cinema_df) * scenario['available_rooms'])
            cinema_subset = self.cinema_df.head(available_rooms)
            
        elif scenario['name'] == '新片上映':
            # 添加新电影（简化处理，使用现有电影但提高评分）
            self.lambda_param = 1.2
            self.alpha_param = 0.4
            
        elif scenario['name'] == '恶劣天气':
            self.lambda_param = 0.8  # 降低收益权重
            self.alpha_param = 0.7  # 提高上座率权重
            
        elif scenario['name'] == '竞争影院活动':
            self.lambda_param = 0.9
            self.alpha_param = 0.6
        
        # 根据场景选择使用的放映厅
        if scenario['name'] == '设备故障':
            rooms_to_use = cinema_subset
        else:
            rooms_to_use = self.cinema_df
        
        # 生成排片
        for _, room_info in rooms_to_use.iterrows():
            room = room_info['room']
            capacity = room_info['capacity']
            
            # 选择电影
            if scenario['name'] == '新片上映':
                # 优先选择高评分电影
                selected_movies = self.movies_df.sort_values('rating', ascending=False).head(4)
            else:
                selected_movies = self.movies_df.head(5)
            
            for _, movie in selected_movies.iterrows():
                movie_id = movie['id']
                versions = self._get_versions(movie_id)
                
                # 选择该厅支持的版本
                supported_versions = [v for v in versions if self._can_room_play_version(room, v)]
                
                if supported_versions:
                    version = supported_versions[0]
                    
                    # 选择时间段
                    if scenario['name'] == '高需求周末':
                        # 增加黄金时段
                        time_slots = ['10:00', '13:00', '16:00', '19:00', '22:00']
                    else:
                        time_slots = ['10:00', '14:00', '18:00']
                    
                    for time_slot in time_slots:
                        # 计算相关指标（考虑场景影响）
                        is_prime = self._is_prime_time(time_slot)
                        
                        # 基础票价
                        base_price = self._calculate_ticket_price(movie_id, version, is_prime)
                        # 应用场景价格调整
                        ticket_price = base_price * scenario['price_multiplier']
                        
                        # 基础上座率
                        base_attendance = self._calculate_attendance(capacity, movie['rating'], time_slot, movie_id)
                        # 应用场景需求调整
                        attendance = int(base_attendance * scenario['demand_multiplier'])
                        attendance = min(attendance, capacity)  # 确保不超过容量
                        
                        cost = self._calculate_cost(capacity, version)
                        sharing_rate = self._get_sharing_rate(movie_id)
                        revenue = ticket_price * attendance * (1 - sharing_rate)
                        net_profit = revenue - cost
                        
                        schedule_results.append({
                            'room': room,
                            'movie_id': movie_id,
                            'movie_title': movie.get('title', f'Movie_{movie_id}'),
                            'version': version,
                            'showtime': time_slot,
                            'capacity': capacity,
                            'attendance': attendance,
                            'ticket_price': ticket_price,
                            'revenue': revenue,
                            'cost': cost,
                            'net_profit': net_profit,
                            'sharing_rate': sharing_rate,
                            'is_prime_time': is_prime
                        })
        
        return schedule_results
    
    def _calculate_scenario_metrics(self, schedule):
        """计算场景指标"""
        if not schedule:
            return {
                'total_revenue': 0,
                'total_attendance': 0,
                'total_profit': 0,
                'avg_profit': 0,
                'occupancy_rate': 0,
                'movie_diversity': 0
            }
        
        df_schedule = pd.DataFrame(schedule)
        
        # 计算各项指标
        total_revenue = df_schedule['revenue'].sum()
        total_attendance = df_schedule['attendance'].sum()
        total_profit = df_schedule['net_profit'].sum()
        avg_profit = total_profit / len(df_schedule) if len(df_schedule) > 0 else 0
        
        # 上座率
        total_capacity = df_schedule['capacity'].sum()
        occupancy_rate = total_attendance / total_capacity if total_capacity > 0 else 0
        
        # 电影多样性
        movie_diversity = df_schedule['movie_id'].nunique()
        
        return {
            'total_revenue': total_revenue,
            'total_attendance': total_attendance,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'occupancy_rate': occupancy_rate,
            'movie_diversity': movie_diversity
        }
    
    def _calculate_scenario_changes(self, baseline_metrics, scenario_metrics):
        """计算场景相对于基准的变化"""
        changes = {}
        
        for key in baseline_metrics.keys():
            baseline_val = baseline_metrics[key]
            scenario_val = scenario_metrics[key]
            
            if baseline_val > 0:
                pct_change = (scenario_val - baseline_val) / baseline_val * 100
            else:
                pct_change = 0
            
            changes[f'{key}_pct_change'] = pct_change
        
        return changes
    
    def _visualize_emergency_results(self, results_df):
        """可视化紧急场景结果"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 收益变化率
        axes[0, 0].bar(results_df['scenario_name'], results_df['change_total_revenue_pct_change'])
        axes[0, 0].set_title('各场景对收益的影响')
        axes[0, 0].set_ylabel('收益变化率 (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 上座率变化率
        axes[0, 1].bar(results_df['scenario_name'], results_df['change_occupancy_rate_pct_change'])
        axes[0, 1].set_title('各场景对上座率的影响')
        axes[0, 1].set_ylabel('上座率变化率 (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 利润变化率
        axes[1, 0].bar(results_df['scenario_name'], results_df['change_total_profit_pct_change'])
        axes[1, 0].set_title('各场景对利润的影响')
        axes[1, 0].set_ylabel('利润变化率 (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 电影多样性变化率
        axes[1, 1].bar(results_df['scenario_name'], results_df['change_movie_diversity_pct_change'])
        axes[1, 1].set_title('各场景对电影多样性的影响')
        axes[1, 1].set_ylabel('多样性变化率 (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('emergency_scenario_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("生成网格实验分析报告...")
        
        report = {
            'experiment_summary': {
                'grid_experiments': len(self.grid_results),
                'backtest_days': len(self.backtest_results),
                'emergency_scenarios': len(self.emergency_results)
            },
            'key_findings': {
                'best_parameters': self._find_best_parameters(),
                'most_stable_day': self._find_most_stable_day(),
                'most_resilient_scenario': self._find_most_resilient_scenario(),
                'least_resilient_scenario': self._find_least_resilient_scenario()
            },
            'recommendations': {
                'parameter_settings': self._recommend_parameter_settings(),
                'scheduling_strategy': self._recommend_scheduling_strategy(),
                'emergency_response': self._recommend_emergency_response()
            }
        }
        
        # 保存报告
        import json
        with open('grid_experiment_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print("综合分析报告已保存")
        
        return report
    
    def _find_best_parameters(self):
        """找到最佳参数组合"""
        if len(self.grid_results) == 0:
            return None
        
        best_result = self.grid_results.loc[self.grid_results['objective_value'].idxmax()]
        return {
            'lambda': best_result['lambda'],
            'alpha': best_result['alpha'],
            'rho': best_result['rho'],
            'objective_value': best_result['objective_value']
        }
    
    def _find_most_stable_day(self):
        """找到最稳定的一天"""
        if len(self.backtest_results) == 0:
            return None
        
        # 计算每天的稳定性指标（基于各项指标的变异系数）
        stability_scores = []
        
        for _, row in self.backtest_results.iterrows():
            # 简化的稳定性评分（基于收益和上座率的平衡）
            revenue_score = row['total_revenue'] / self.backtest_results['total_revenue'].max()
            occupancy_score = row['occupancy_rate'] / self.backtest_results['occupancy_rate'].max()
            
            # 稳定性评分（越高越稳定）
            stability_score = (revenue_score + occupancy_score) / 2
            stability_scores.append(stability_score)
        
        most_stable_idx = np.argmax(stability_scores)
        most_stable_day = self.backtest_results.iloc[most_stable_idx]
        
        return {
            'date': most_stable_day['date'],
            'stability_score': stability_scores[most_stable_idx],
            'total_revenue': most_stable_day['total_revenue'],
            'occupancy_rate': most_stable_day['occupancy_rate']
        }
    
    def _find_most_resilient_scenario(self):
        """找到最具弹性的场景（受负面影响最小）"""
        if len(self.emergency_results) == 0:
            return None
        
        # 计算每个场景的弹性评分（基于利润变化率）
        resilience_scores = []
        
        for _, row in self.emergency_results.iterrows():
            # 利润变化率（绝对值越小，弹性越好）
            profit_change_pct = abs(row['change_total_profit_pct_change'])
            
            # 弹性评分（越高越有弹性）
            resilience_score = 100 - profit_change_pct
            resilience_scores.append(resilience_score)
        
        most_resilient_idx = np.argmax(resilience_scores)
        most_resilient_scenario = self.emergency_results.iloc[most_resilient_idx]
        
        return {
            'scenario_name': most_resilient_scenario['scenario_name'],
            'resilience_score': resilience_scores[most_resilient_idx],
            'profit_change_pct': most_resilient_scenario['change_total_profit_pct_change']
        }
    
    def _find_least_resilient_scenario(self):
        """找到最缺乏弹性的场景（受负面影响最大）"""
        if len(self.emergency_results) == 0:
            return None
        
        # 计算每个场景的弹性评分
        resilience_scores = []
        
        for _, row in self.emergency_results.iterrows():
            # 利润变化率（绝对值越大，弹性越差）
            profit_change_pct = abs(row['change_total_profit_pct_change'])
            
            # 弹性评分（越高越有弹性）
            resilience_score = 100 - profit_change_pct
            resilience_scores.append(resilience_score)
        
        least_resilient_idx = np.argmin(resilience_scores)
        least_resilient_scenario = self.emergency_results.iloc[least_resilient_idx]
        
        return {
            'scenario_name': least_resilient_scenario['scenario_name'],
            'resilience_score': resilience_scores[least_resilient_idx],
            'profit_change_pct': least_resilient_scenario['change_total_profit_pct_change']
        }
    
    def _recommend_parameter_settings(self):
        """推荐参数设置"""
        best_params = self._find_best_parameters()
        
        if best_params is None:
            return "无法提供参数设置建议，请先运行参数网格实验"
        
        return {
            'lambda': best_params['lambda'],
            'alpha': best_params['alpha'],
            'rho': best_params['rho'],
            'explanation': f"基于网格实验结果，推荐使用λ={best_params['lambda']}, α={best_params['alpha']}, ρ={best_params['rho']}的参数组合，可获得最高的目标函数值({best_params['objective_value']:.0f})"
        }
    
    def _recommend_scheduling_strategy(self):
        """推荐排片策略"""
        if len(self.backtest_results) == 0:
            return "无法提供排片策略建议，请先运行滚动回测"
        
        most_stable_day = self._find_most_stable_day()
        
        # 分析周末和工作日的差异
        weekend_data = self.backtest_results[self.backtest_results['date'].isin(['2024-08-17', '2024-08-18'])]  # 假设这两天是周末
        weekday_data = self.backtest_results[~self.backtest_results['date'].isin(['2024-08-17', '2024-08-18'])]
        
        if len(weekend_data) > 0 and len(weekday_data) > 0:
            weekend_revenue = weekend_data['total_revenue'].mean()
            weekday_revenue = weekday_data['total_revenue'].mean()
            weekend_occupancy = weekend_data['occupancy_rate'].mean()
            weekday_occupancy = weekday_data['occupancy_rate'].mean()
            
            strategy = {
                'weekend_focus': weekend_revenue > weekday_revenue,
                'weekend_occupancy_advantage': weekend_occupancy > weekday_occupancy,
                'most_stable_day': most_stable_day['date'],
                'recommendations': []
            }
            
            if strategy['weekend_focus']:
                strategy['recommendations'].append("周末应安排更多高收益电影和黄金时段场次")
            
            if strategy['weekend_occupancy_advantage']:
                strategy['recommendations'].append("周末上座率更高，可适当提高票价")
            
            strategy['recommendations'].append(f"{most_stable_day['date']}的排片策略最为稳定，可作为参考模板")
            
            return strategy
        else:
            return "数据不足，无法提供详细的排片策略建议"
    
    def _recommend_emergency_response(self):
        """推荐紧急响应策略"""
        if len(self.emergency_results) == 0:
            return "无法提供紧急响应建议，请先运行紧急场景测试"
        
        most_resilient = self._find_most_resilient_scenario()
        least_resilient = self._find_least_resilient_scenario()
        
        response_strategies = {
            'high_demand_weekend': {
                'response': "增加黄金时段场次，适当提高票价，安排高评分电影",
                'expected_impact': "收益提升15-20%"
            },
            'equipment_failure': {
                'response': "将受影响厅场的电影重新安排到其他厅场，优先保证高收益场次",
                'expected_impact': "收益减少5-10%"
            },
            'new_movie_release': {
                'response': "调整排片计划，为新片安排更多黄金时段场次",
                'expected_impact': "收益提升10-15%"
            },
            'bad_weather': {
                'response': "降低票价，增加非黄金时段场次，安排家庭友好型电影",
                'expected_impact': "收益减少15-20%"
            },
            'competitor_activity': {
                'response': "推出优惠活动，增加特色场次，提高服务质量",
                'expected_impact': "收益减少10-15%"
            }
        }
        
        return {
            'most_resilient_scenario': most_resilient['scenario_name'],
            'least_resilient_scenario': least_resilient['scenario_name'],
            'response_strategies': response_strategies,
            'general_recommendation': "建立紧急响应预案，包括快速调整排片、灵活定价和营销策略"
        }
    
    def run_full_analysis(self):
        """运行完整的网格实验分析"""
        print("开始完整的网格实验分析...")
        
        # 1. 参数网格实验
        grid_results = self.parameter_grid_experiment()
        
        # 2. 滚动回测
        backtest_results = self.rolling_backtest()
        
        # 3. 紧急场景测试
        emergency_results = self.emergency_scenario_test()
        
        # 4. 生成综合报告
        report = self.generate_comprehensive_report()
        
        print("完整的网格实验分析已完成！")
        
        return {
            'grid_results': grid_results,
            'backtest_results': backtest_results,
            'emergency_results': emergency_results,
            'report': report
        }


def main():
    """主函数"""
    # 创建网格实验分析器
    analyzer = GridExperimentAnalyzer(
        cinema_file='../input_data/df_cinema.csv',
        movies_file='../input_data/df_movies_schedule_ours.csv'
    )
    
    # 运行完整分析
    results = analyzer.run_full_analysis()
    
    print("\n=== 网格实验分析总结 ===")
    if results['grid_results'] is not None:
        print(f"参数网格实验: {len(results['grid_results'])} 种参数组合")
        best_params = results['report']['key_findings']['best_parameters']
        if best_params:
            print(f"最佳参数: λ={best_params['lambda']}, α={best_params['alpha']}, ρ={best_params['rho']}")
    
    if results['backtest_results'] is not None:
        print(f"滚动回测: {len(results['backtest_results'])} 天")
        stable_day = results['report']['key_findings']['most_stable_day']
        if stable_day:
            print(f"最稳定的一天: {stable_day['date']}")
    
    if results['emergency_results'] is not None:
        print(f"紧急场景测试: {len(results['emergency_results'])} 个场景")
        resilient_scenario = results['report']['key_findings']['most_resilient_scenario']
        if resilient_scenario:
            print(f"最具弹性场景: {resilient_scenario['scenario_name']}")


if __name__ == "__main__":
    main()