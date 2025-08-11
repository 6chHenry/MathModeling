import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ParameterPerturbationAnalyzer:
    """
    参数扰动分析器
    
    提供全面的参数扰动测试功能，包括：
    - 参数扰动测试（±变化）
    - 约束边界测试（±20%场景）
    - 随机返回噪声重采样30次评估解的稳定性
    - Gap指标和参数变化率分析
    """
    
    def __init__(self, schedule_file, movies_file, rules_file=None):
        """
        初始化参数扰动分析器
        
        Args:
            schedule_file: 排片文件路径
            movies_file: 电影文件路径
            rules_file: 规则文件路径（可选）
        """
        self.schedule_file = schedule_file
        self.movies_file = movies_file
        self.schedule_df = None
        self.movies_df = None
        
        # 默认规则
        self.rules = {
            "open_time": "10:00",
            "close_time": "03:00",
            "min_gap": 15,  # minutes
            "golden_start": "18:00",
            "golden_end": "21:00",  # inclusive
            "version_coeff": {"2D": 1.0, "3D": 1.1, "IMAX": 1.15},
            "version_total_caps": {"3D": 1200, "IMAX": 1500},  # minutes
            "genre_caps": {"Animation": (1, 5), "Horror": (0, 3), "Action": (2, 6), "Drama": (1, 6)},
            "genre_time_limits": {
                "Animation": (None, "19:00"),
                "Family": (None, "19:00"),
                "Horror": ("21:00", None),
                "Thriller": ("21:00", None),
            },
        }
        
        # 如果有规则文件，加载规则
        if rules_file:
            self.load_rules(rules_file)
        
        # 存储分析结果
        self.perturbation_results = []
        self.constraint_results = []
        self.noise_resampling_results = []
        self.gap_metrics = []
        
    def load_rules(self, rules_file):
        """加载规则文件"""
        try:
            # 这里假设规则文件是JSON格式
            import json
            with open(rules_file, 'r', encoding='utf-8') as f:
                loaded_rules = json.load(f)
            self.rules.update(loaded_rules)
            print(f"已从 {rules_file} 加载规则")
        except Exception as e:
            print(f"加载规则文件失败: {e}")
    
    def load_data(self):
        """加载排片和电影数据"""
        print("加载数据...")
        self.schedule_df = pd.read_csv(self.schedule_file)
        self.movies_df = pd.read_csv(self.movies_file)
        
        # 预处理电影数据
        self._preprocess_movies_data()
        
        print(f"数据加载完成: {len(self.schedule_df)} 个场次, {len(self.movies_df)} 部电影")
    
    def _preprocess_movies_data(self):
        """预处理电影数据"""
        # 计算圆整后的运行时间
        self.movies_df['rounded_runtime'] = self.movies_df['runtime'].apply(self._ceil_to_30)
        
        # 处理题材列表
        self.movies_df['genres_list'] = self.movies_df['genres'].apply(
            lambda x: [g.strip() for g in str(x).split(',')]
        )
    
    def _ceil_to_30(self, runtime):
        """将运行时间向上取整到30分钟倍数"""
        return int(((runtime + 29) // 30) * 30)
    
    def _parse_hhmm(self, hhmm):
        """解析HH:MM格式时间"""
        return datetime.strptime(hhmm, "%H:%M")
    
    def _to_minutes(self, td):
        """将时间差转换为分钟"""
        return td.days * 24 * 60 + td.seconds // 60
    
    def _in_quarter(self, hhmm):
        """检查时间是否在15分钟刻度上"""
        m = int(hhmm.split(":")[1])
        return m in (0, 15, 30, 45)
    
    def parameter_perturbation_test(self, perturbation_levels=[0.8, 0.9, 1.1, 1.2]):
        """参数扰动测试"""
        print("执行参数扰动测试...")
        
        # 基础参数
        base_params = {
            'min_gap': self.rules['min_gap'],
            'version_caps': self.rules['version_total_caps'].copy(),
            'genre_caps': self.rules['genre_caps'].copy()
        }
        
        # 计算基础排片的性能指标
        base_metrics = self._calculate_schedule_metrics(self.schedule_df)
        
        results = []
        
        # 对每个参数进行扰动测试
        for param_name, param_value in base_params.items():
            if param_name == 'min_gap':
                # 对最小间隔时间进行扰动
                for perturbation in perturbation_levels:
                    new_gap = int(param_value * perturbation)
                    
                    # 临时修改规则
                    original_gap = self.rules['min_gap']
                    self.rules['min_gap'] = new_gap
                    
                    # 重新计算指标
                    metrics = self._calculate_schedule_metrics(self.schedule_df)
                    
                    # 计算变化
                    changes = self._calculate_metric_changes(base_metrics, metrics)
                    
                    results.append({
                        'parameter': param_name,
                        'perturbation': perturbation,
                        'original_value': param_value,
                        'new_value': new_gap,
                        **changes
                    })
                    
                    # 恢复原始值
                    self.rules['min_gap'] = original_gap
                    
            elif param_name == 'version_caps':
                # 对版本时长限制进行扰动
                for version, cap in param_value.items():
                    for perturbation in perturbation_levels:
                        new_cap = int(cap * perturbation)
                        
                        # 临时修改规则
                        original_cap = self.rules['version_total_caps'][version]
                        self.rules['version_total_caps'][version] = new_cap
                        
                        # 重新计算指标
                        metrics = self._calculate_schedule_metrics(self.schedule_df)
                        
                        # 计算变化
                        changes = self._calculate_metric_changes(base_metrics, metrics)
                        
                        results.append({
                            'parameter': f'{param_name}_{version}',
                            'perturbation': perturbation,
                            'original_value': cap,
                            'new_value': new_cap,
                            **changes
                        })
                        
                        # 恢复原始值
                        self.rules['version_total_caps'][version] = original_cap
                        
            elif param_name == 'genre_caps':
                # 对题材播放次数限制进行扰动
                for genre, (min_cap, max_cap) in param_value.items():
                    for perturbation in perturbation_levels:
                        new_min = int(min_cap * perturbation)
                        new_max = int(max_cap * perturbation)
                        
                        # 临时修改规则
                        original_min, original_max = self.rules['genre_caps'][genre]
                        self.rules['genre_caps'][genre] = (new_min, new_max)
                        
                        # 重新计算指标
                        metrics = self._calculate_schedule_metrics(self.schedule_df)
                        
                        # 计算变化
                        changes = self._calculate_metric_changes(base_metrics, metrics)
                        
                        results.append({
                            'parameter': f'{param_name}_{genre}',
                            'perturbation': perturbation,
                            'original_value': f'{min_cap}-{max_cap}',
                            'new_value': f'{new_min}-{new_max}',
                            **changes
                        })
                        
                        # 恢复原始值
                        self.rules['genre_caps'][genre] = (original_min, original_max)
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        self.perturbation_results = results_df
        
        # 可视化结果
        self._visualize_perturbation_results(results_df)
        
        # 保存结果
        results_df.to_csv('parameter_perturbation_results.csv', index=False)
        
        print("参数扰动测试完成，结果已保存")
        
        return results_df
    
    def constraint_boundary_test(self, boundary_percentages=[0.8, 1.2]):
        """约束边界测试（±20%场景）"""
        print("执行约束边界测试...")
        
        # 基础约束
        base_constraints = {
            'version_total_caps': self.rules['version_total_caps'].copy(),
            'genre_caps': self.rules['genre_caps'].copy()
        }
        
        # 计算基础排片的性能指标
        base_metrics = self._calculate_schedule_metrics(self.schedule_df)
        
        results = []
        
        # 对每个约束进行边界测试
        for constraint_name, constraint_value in base_constraints.items():
            if constraint_name == 'version_total_caps':
                # 对版本时长限制进行边界测试
                for version, cap in constraint_value.items():
                    for boundary_pct in boundary_percentages:
                        new_cap = int(cap * boundary_pct)
                        
                        # 临时修改约束
                        original_cap = self.rules['version_total_caps'][version]
                        self.rules['version_total_caps'][version] = new_cap
                        
                        # 重新计算指标
                        metrics = self._calculate_schedule_metrics(self.schedule_df)
                        
                        # 计算变化
                        changes = self._calculate_metric_changes(base_metrics, metrics)
                        
                        # 计算约束违反情况
                        violations = self._calculate_constraint_violations()
                        
                        results.append({
                            'constraint': f'{constraint_name}_{version}',
                            'boundary_pct': boundary_pct,
                            'original_value': cap,
                            'new_value': new_cap,
                            'violations_count': violations['total_violations'],
                            **changes
                        })
                        
                        # 恢复原始值
                        self.rules['version_total_caps'][version] = original_cap
                        
            elif constraint_name == 'genre_caps':
                # 对题材播放次数限制进行边界测试
                for genre, (min_cap, max_cap) in constraint_value.items():
                    for boundary_pct in boundary_percentages:
                        new_min = int(min_cap * boundary_pct)
                        new_max = int(max_cap * boundary_pct)
                        
                        # 临时修改约束
                        original_min, original_max = self.rules['genre_caps'][genre]
                        self.rules['genre_caps'][genre] = (new_min, new_max)
                        
                        # 重新计算指标
                        metrics = self._calculate_schedule_metrics(self.schedule_df)
                        
                        # 计算变化
                        changes = self._calculate_metric_changes(base_metrics, metrics)
                        
                        # 计算约束违反情况
                        violations = self._calculate_constraint_violations()
                        
                        results.append({
                            'constraint': f'{constraint_name}_{genre}',
                            'boundary_pct': boundary_pct,
                            'original_value': f'{min_cap}-{max_cap}',
                            'new_value': f'{new_min}-{new_max}',
                            'violations_count': violations['total_violations'],
                            **changes
                        })
                        
                        # 恢复原始值
                        self.rules['genre_caps'][genre] = (original_min, original_max)
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        self.constraint_results = results_df
        
        # 可视化结果
        self._visualize_constraint_results(results_df)
        
        # 保存结果
        results_df.to_csv('constraint_boundary_results.csv', index=False)
        
        print("约束边界测试完成，结果已保存")
        
        return results_df
    
    def noise_resampling_test(self, n_samples=30, noise_level=0.1):
        """随机返回噪声重采样测试"""
        print(f"执行随机返回噪声重采样测试（{n_samples}次）...")
        
        # 基础排片的性能指标
        base_metrics = self._calculate_schedule_metrics(self.schedule_df)
        
        results = []
        
        for i in range(n_samples):
            # 添加噪声到排片数据
            noisy_schedule = self._add_noise_to_schedule(noise_level)
            
            # 计算噪声排片的性能指标
            noisy_metrics = self._calculate_schedule_metrics(noisy_schedule)
            
            # 计算变化
            changes = self._calculate_metric_changes(base_metrics, noisy_metrics)
            
            # 计算Gap指标
            gap_metrics = self._calculate_gap_metrics(base_metrics, noisy_metrics)
            
            results.append({
                'sample_id': i + 1,
                'noise_level': noise_level,
                **changes,
                **gap_metrics
            })
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        self.noise_resampling_results = results_df
        
        # 计算统计指标
        stats = self._calculate_noise_statistics(results_df)
        
        # 可视化结果
        self._visualize_noise_results(results_df, stats)
        
        # 保存结果
        results_df.to_csv('noise_resampling_results.csv', index=False)
        
        print("随机返回噪声重采样测试完成，结果已保存")
        
        return results_df, stats
    
    def _add_noise_to_schedule(self, noise_level):
        """给排片数据添加噪声"""
        noisy_schedule = self.schedule_df.copy()
        
        # 对放映时间添加噪声（±15分钟）
        time_noise = np.random.uniform(-15, 15, len(noisy_schedule))
        noisy_schedule['showtime_noise'] = noisy_schedule['showtime'].copy()
        
        # 对版本添加随机变化
        versions = ['2D', '3D', 'IMAX']
        for idx in range(len(noisy_schedule)):
            if np.random.random() < noise_level:
                noisy_schedule.loc[idx, 'version'] = np.random.choice(versions)
        
        # 对放映厅添加随机变化
        rooms = noisy_schedule['room'].unique()
        for idx in range(len(noisy_schedule)):
            if np.random.random() < noise_level:
                noisy_schedule.loc[idx, 'room'] = np.random.choice(rooms)
        
        return noisy_schedule
    
    def _calculate_schedule_metrics(self, schedule_df):
        """计算排片性能指标"""
        metrics = {}
        
        # 计算版本总时长
        version_minutes = defaultdict(int)
        for _, row in schedule_df.iterrows():
            movie_id = row['id']
            version = row['version']
            movie_info = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
            runtime = movie_info['rounded_runtime']
            
            if version in ('3D', 'IMAX'):
                version_minutes[version] += runtime
        
        metrics['version_minutes'] = dict(version_minutes)
        
        # 计算题材播放次数
        genre_counts = defaultdict(int)
        for _, row in schedule_df.iterrows():
            movie_id = row['id']
            movie_info = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
            genres = movie_info['genres_list']
            
            for genre in genres:
                genre_counts[genre] += 1
        
        metrics['genre_counts'] = dict(genre_counts)
        
        # 计算总场次
        metrics['total_shows'] = len(schedule_df)
        
        # 计算黄金时段场次
        golden_shows = 0
        for _, row in schedule_df.iterrows():
            showtime = row['showtime']
            if self._is_golden_time(showtime):
                golden_shows += 1
        
        metrics['golden_shows'] = golden_shows
        metrics['golden_ratio'] = golden_shows / len(schedule_df) if len(schedule_df) > 0 else 0
        
        # 计算平均间隔时间
        if len(schedule_df) > 1:
            intervals = self._calculate_intervals(schedule_df)
            metrics['avg_interval'] = np.mean(intervals)
        else:
            metrics['avg_interval'] = 0
        
        return metrics
    
    def _is_golden_time(self, showtime):
        """判断是否为黄金时段"""
        hour = int(showtime.split(':')[0])
        golden_start_hour = int(self.rules['golden_start'].split(':')[0])
        golden_end_hour = int(self.rules['golden_end'].split(':')[0])
        return golden_start_hour <= hour <= golden_end_hour
    
    def _calculate_intervals(self, schedule_df):
        """计算场次间隔时间"""
        intervals = []
        
        # 按放映厅分组
        for room, group in schedule_df.groupby('room'):
            # 按时间排序
            shows = group.sort_values('showtime')
            
            # 计算间隔
            for i in range(len(shows) - 1):
                current_time = self._parse_hhmm(shows.iloc[i]['showtime'])
                next_time = self._parse_hhmm(shows.iloc[i+1]['showtime'])
                
                # 计算当前场次的结束时间
                current_movie_id = shows.iloc[i]['id']
                current_movie = self.movies_df[self.movies_df['id'] == current_movie_id].iloc[0]
                current_runtime = int(current_movie['rounded_runtime'])  # 转换为 int 类型
                current_end = current_time + timedelta(minutes=current_runtime)
                
                # 计算间隔
                interval = self._to_minutes(next_time - current_end)
                intervals.append(interval)
        
        return intervals
    
    def _calculate_metric_changes(self, base_metrics, new_metrics):
        """计算指标变化"""
        changes = {}
        
        # 版本时长变化
        for version in base_metrics['version_minutes']:
            base_val = base_metrics['version_minutes'][version]
            new_val = new_metrics['version_minutes'].get(version, 0)
            changes[f'{version}_minutes_change'] = new_val - base_val
            changes[f'{version}_minutes_pct_change'] = (new_val - base_val) / base_val * 100 if base_val > 0 else 0
        
        # 题材次数变化
        for genre in base_metrics['genre_counts']:
            base_val = base_metrics['genre_counts'][genre]
            new_val = new_metrics['genre_counts'].get(genre, 0)
            changes[f'{genre}_count_change'] = new_val - base_val
            changes[f'{genre}_count_pct_change'] = (new_val - base_val) / base_val * 100 if base_val > 0 else 0
        
        # 总场次变化
        changes['total_shows_change'] = new_metrics['total_shows'] - base_metrics['total_shows']
        changes['total_shows_pct_change'] = (new_metrics['total_shows'] - base_metrics['total_shows']) / base_metrics['total_shows'] * 100 if base_metrics['total_shows'] > 0 else 0
        
        # 黄金时段变化
        changes['golden_shows_change'] = new_metrics['golden_shows'] - base_metrics['golden_shows']
        changes['golden_shows_pct_change'] = (new_metrics['golden_shows'] - base_metrics['golden_shows']) / base_metrics['golden_shows'] * 100 if base_metrics['golden_shows'] > 0 else 0
        
        # 平均间隔变化
        changes['avg_interval_change'] = new_metrics['avg_interval'] - base_metrics['avg_interval']
        changes['avg_interval_pct_change'] = (new_metrics['avg_interval'] - base_metrics['avg_interval']) / base_metrics['avg_interval'] * 100 if base_metrics['avg_interval'] > 0 else 0
        
        return changes
    
    def _calculate_constraint_violations(self):
        """计算约束违反情况"""
        violations = {
            'version_violations': {},
            'genre_violations': {},
            'total_violations': 0
        }
        
        # 检查版本时长约束
        version_minutes = defaultdict(int)
        for _, row in self.schedule_df.iterrows():
            movie_id = row['id']
            version = row['version']
            movie_info = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
            runtime = movie_info['rounded_runtime']
            
            if version in ('3D', 'IMAX'):
                version_minutes[version] += runtime
        
        for version, cap in self.rules['version_total_caps'].items():
            if version_minutes[version] > cap:
                violations['version_violations'][version] = version_minutes[version] - cap
                violations['total_violations'] += 1
        
        # 检查题材次数约束
        genre_counts = defaultdict(int)
        for _, row in self.schedule_df.iterrows():
            movie_id = row['id']
            movie_info = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
            genres = movie_info['genres_list']
            
            for genre in genres:
                genre_counts[genre] += 1
        
        for genre, (min_cap, max_cap) in self.rules['genre_caps'].items():
            count = genre_counts.get(genre, 0)
            if count < min_cap or count > max_cap:
                violations['genre_violations'][genre] = (min_cap, max_cap, count)
                violations['total_violations'] += 1
        
        return violations
    
    def _calculate_gap_metrics(self, base_metrics, new_metrics):
        """计算Gap指标"""
        gap_metrics = {}
        
        # 计算各个指标的Gap
        gap_metrics['total_shows_gap'] = abs(new_metrics['total_shows'] - base_metrics['total_shows'])
        gap_metrics['golden_shows_gap'] = abs(new_metrics['golden_shows'] - base_metrics['golden_shows'])
        gap_metrics['avg_interval_gap'] = abs(new_metrics['avg_interval'] - base_metrics['avg_interval'])
        
        # 计算版本时长Gap
        version_gap = 0
        for version in base_metrics['version_minutes']:
            base_val = base_metrics['version_minutes'][version]
            new_val = new_metrics['version_minutes'].get(version, 0)
            version_gap += abs(new_val - base_val)
        gap_metrics['version_gap'] = version_gap
        
        # 计算题材次数Gap
        genre_gap = 0
        for genre in base_metrics['genre_counts']:
            base_val = base_metrics['genre_counts'][genre]
            new_val = new_metrics['genre_counts'].get(genre, 0)
            genre_gap += abs(new_val - base_val)
        gap_metrics['genre_gap'] = genre_gap
        
        # 计算总Gap
        gap_metrics['total_gap'] = (
            gap_metrics['total_shows_gap'] +
            gap_metrics['golden_shows_gap'] +
            gap_metrics['avg_interval_gap'] +
            gap_metrics['version_gap'] +
            gap_metrics['genre_gap']
        )
        
        return gap_metrics
    
    def _calculate_noise_statistics(self, results_df):
        """计算噪声测试的统计指标"""
        stats = {}
        
        # 计算各个指标的统计量
        gap_columns = [col for col in results_df.columns if col.endswith('_gap')]
        for col in gap_columns:
            stats[col] = {
                'mean': results_df[col].mean(),
                'std': results_df[col].std(),
                'min': results_df[col].min(),
                'max': results_df[col].max(),
                'cv': results_df[col].std() / results_df[col].mean() if results_df[col].mean() > 0 else 0
            }
        
        return stats
    
    def _visualize_perturbation_results(self, results_df):
        """可视化参数扰动结果"""
        # 选择关键指标进行可视化
        key_metrics = ['total_shows_pct_change', 'golden_shows_pct_change', 'avg_interval_pct_change']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(key_metrics):
            # 按参数分组
            param_groups = results_df.groupby('parameter')
            
            # 为每个参数绘制扰动曲线
            for param_name, group in param_groups:
                if len(group) > 1:
                    axes[i].plot(group['perturbation'], group[metric], 
                               marker='o', label=param_name, alpha=0.7)
            
            axes[i].set_xlabel('扰动水平')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].set_title(f'参数扰动对{metric.replace("_", " ").replace("pct change", "变化率")}的影响')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('parameter_perturbation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _visualize_constraint_results(self, results_df):
        """可视化约束边界结果"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 约束违反情况
        violation_data = results_df.groupby('constraint')['violations_count'].mean().sort_values(ascending=False)
        axes[0].bar(range(len(violation_data)), violation_data.values)
        axes[0].set_xticks(range(len(violation_data)))
        axes[0].set_xticklabels(violation_data.index, rotation=45, ha='right')
        axes[0].set_ylabel('平均违反次数')
        axes[0].set_title('各约束边界下的平均违反次数')
        axes[0].grid(True, alpha=0.3)
        
        # 2. 边界百分比对性能的影响
        boundary_performance = results_df.groupby('boundary_pct')['total_shows_pct_change'].agg(['mean', 'std']).reset_index()
        axes[1].errorbar(boundary_performance['boundary_pct'], boundary_performance['mean'], 
                        yerr=boundary_performance['std'], fmt='-o', capsize=5)
        axes[1].set_xlabel('边界百分比')
        axes[1].set_ylabel('总场次变化率 (%)')
        axes[1].set_title('边界百分比对总场次的影响')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('constraint_boundary_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _visualize_noise_results(self, results_df, stats):
        """可视化噪声重采样结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 总Gap分布
        axes[0, 0].hist(results_df['total_gap'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('总Gap')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].set_title('总Gap分布')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 各指标Gap的箱线图
        gap_columns = [col for col in results_df.columns if col.endswith('_gap') and col != 'total_gap']
        gap_data = results_df[gap_columns]
        axes[0, 1].boxplot(gap_data.values)
        axes[0, 1].set_xticklabels([col.replace('_gap', '') for col in gap_columns], rotation=45)
        axes[0, 1].set_ylabel('Gap值')
        axes[0, 1].set_title('各指标Gap分布')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 总场次变化率分布
        axes[1, 0].hist(results_df['total_shows_pct_change'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('总场次变化率 (%)')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].set_title('总场次变化率分布')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 黄金时段比例变化率分布
        axes[1, 1].hist(results_df['golden_shows_pct_change'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('黄金时段场次变化率 (%)')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].set_title('黄金时段场次变化率分布')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('noise_resampling_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("生成参数扰动分析报告...")
        
        report = {
            'test_summary': {
                'parameter_perturbation_tests': len(self.perturbation_results),
                'constraint_boundary_tests': len(self.constraint_results),
                'noise_resampling_tests': len(self.noise_resampling_results)
            },
            'key_findings': {
                'most_sensitive_parameter': self._find_most_sensitive_parameter(),
                'most_violated_constraint': self._find_most_violated_constraint(),
                'average_total_gap': self.noise_resampling_results['total_gap'].mean() if len(self.noise_resampling_results) > 0 else 0,
                'solution_stability': self._assess_solution_stability()
            }
        }
        
        # 保存报告
        import json
        with open('parameter_perturbation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print("综合分析报告已保存")
        
        return report
    
    def _find_most_sensitive_parameter(self):
        """找到最敏感的参数"""
        if len(self.perturbation_results) == 0:
            return None
        
        # 计算每个参数的平均绝对变化率
        param_sensitivity = {}
        pct_change_columns = [col for col in self.perturbation_results.columns if col.endswith('_pct_change')]
        
        for param in self.perturbation_results['parameter'].unique():
            param_data = self.perturbation_results[self.perturbation_results['parameter'] == param]
            avg_change = param_data[pct_change_columns].abs().mean().mean()
            param_sensitivity[param] = avg_change
        
        return max(param_sensitivity, key=param_sensitivity.get)
    
    def _find_most_violated_constraint(self):
        """找到最常被违反的约束"""
        if len(self.constraint_results) == 0:
            return None
        
        # 计算每个约束的平均违反次数
        constraint_violations = self.constraint_results.groupby('constraint')['violations_count'].mean()
        
        return constraint_violations.idxmax()
    
    def _assess_solution_stability(self):
        """评估解的稳定性"""
        if len(self.noise_resampling_results) == 0:
            return None
        
        # 计算总Gap的变异系数
        total_gap_cv = self.noise_resampling_results['total_gap'].std() / self.noise_resampling_results['total_gap'].mean()
        
        # 根据变异系数评估稳定性
        if total_gap_cv < 0.1:
            return "高稳定性"
        elif total_gap_cv < 0.3:
            return "中等稳定性"
        else:
            return "低稳定性"
    
    def run_full_analysis(self):
        """运行完整的参数扰动分析"""
        print("开始完整的参数扰动分析...")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 参数扰动测试
        perturbation_results = self.parameter_perturbation_test()
        
        # 3. 约束边界测试
        constraint_results = self.constraint_boundary_test()
        
        # 4. 噪声重采样测试
        noise_results, noise_stats = self.noise_resampling_test()
        
        # 5. 生成综合报告
        report = self.generate_comprehensive_report()
        
        print("完整的参数扰动分析已完成！")
        
        return {
            'perturbation_results': perturbation_results,
            'constraint_results': constraint_results,
            'noise_results': noise_results,
            'noise_stats': noise_stats,
            'report': report
        }


def main():
    """主函数"""
    # 创建参数扰动分析器
    analyzer = ParameterPerturbationAnalyzer(
        schedule_file='df_result_2_copt_ours_new.csv',
        movies_file='df_movies_schedule_ours_new.csv'
    )
    
    # 运行完整分析
    results = analyzer.run_full_analysis()
    
    print("\n=== 参数扰动分析总结 ===")
    if results['perturbation_results'] is not None:
        print(f"参数扰动测试: {len(results['perturbation_results'])} 个测试场景")
    
    if results['constraint_results'] is not None:
        print(f"约束边界测试: {len(results['constraint_results'])} 个测试场景")
    
    if results['noise_results'] is not None:
        print(f"噪声重采样测试: {len(results['noise_results'])} 次重采样")
        print(f"平均总Gap: {results['noise_results']['total_gap'].mean():.2f}")
    
    if results['report'] is not None:
        print(f"最敏感参数: {results['report']['key_findings']['most_sensitive_parameter']}")
        print(f"解稳定性: {results['report']['key_findings']['solution_stability']}")


if __name__ == "__main__":
    main()