import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class CinemaSchedulingVisualizer:
    def __init__(self, optimizer, schedule_results):
        """
        初始化可视化器

        Args:
            optimizer: CinemaSchedulingOptimizer实例
            schedule_results: 优化结果列表
        """
        self.optimizer = optimizer
        self.schedule = schedule_results
        self.colors = plt.cm.Set3(np.linspace(0, 1, len(optimizer.movies_df)))

        # 为每部电影分配颜色
        self.movie_colors = {}
        for i, movie_id in enumerate(optimizer.movies_df['id']):
            self.movie_colors[movie_id] = self.colors[i]

    def plot_schedule_gantt(self, save_path=None):
        """绘制甘特图显示排片时间表"""
        fig, ax = plt.subplots(figsize=(16, 10))

        rooms = sorted(self.optimizer.cinema_df['room'].unique())
        y_positions = {room: i for i, room in enumerate(rooms)}

        for item in self.schedule:
            room = item['room']
            movie_id = item['id']
            showtime = item['showtime']
            version = item['version']

            # 获取电影信息
            movie = self.optimizer.movies_df[self.optimizer.movies_df['id'] == movie_id].iloc[0]
            runtime = self.optimizer._round_up_to_30(movie['runtime'])

            # 转换时间
            hour, minute = map(int, showtime.split(':'))
            start_time = hour + minute / 60.0
            end_time = start_time + runtime / 60.0

            # 绘制时间条
            y_pos = y_positions[room]
            rect = Rectangle((start_time, y_pos - 0.4), runtime / 60.0, 0.8,
                             facecolor=self.movie_colors[movie_id],
                             edgecolor='black', alpha=0.7)
            ax.add_patch(rect)

            # 添加文本标签
            ax.text(start_time + runtime / 120.0, y_pos,
                    f'电影{movie_id}\n{version}',
                    ha='center', va='center', fontsize=8, fontweight='bold')

        # 设置坐标轴
        ax.set_xlim(10, 27)
        ax.set_ylim(-0.5, len(rooms) - 0.5)
        ax.set_xlabel('时间 (小时)', fontsize=12)
        ax.set_ylabel('放映厅', fontsize=12)
        ax.set_title('影院排片甘特图', fontsize=16, fontweight='bold')

        # 设置y轴刻度
        ax.set_yticks(range(len(rooms)))
        ax.set_yticklabels(rooms)

        # 设置x轴刻度
        x_ticks = list(range(10, 28))
        x_labels = [f'{h:02d}:00' if h < 24 else f'{h - 24:02d}:00' for h in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45)

        # 添加网格
        ax.grid(True, alpha=0.3)

        # 添加图例
        legend_elements = []
        for movie_id in self.optimizer.movies_df['id']:
            movie_title = f'电影{movie_id}'
            legend_elements.append(plt.Rectangle((0, 0), 1, 1,
                                                 facecolor=self.movie_colors[movie_id],
                                                 label=movie_title))
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_revenue_analysis(self, save_path=None):
        """分析收益情况"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 计算每场次的收益
        revenues = []
        costs = []
        profits = []
        movie_revenues = {}
        room_revenues = {}
        version_revenues = {'2D': 0, '3D': 0, 'IMAX': 0}
        time_revenues = {}

        for item in self.schedule:
            room = item['room']
            movie_id = item['id']
            version = item['version']
            showtime = item['showtime']
            attendance = item['attendance']

            # 获取容量和电影信息
            room_capacity = self.optimizer.cinema_df[
                self.optimizer.cinema_df['room'] == room]['capacity'].iloc[0]
            movie = self.optimizer.movies_df[self.optimizer.movies_df['id'] == movie_id].iloc[0]

            # 计算收益和成本
            is_prime = self.optimizer._is_prime_time(showtime)
            ticket_price = self.optimizer._calculate_ticket_price(movie_id, version, is_prime)
            sharing_rate = self.optimizer._get_sharing_rate(movie_id)

            revenue = ticket_price * attendance * (1 - sharing_rate)
            cost = self.optimizer._calculate_cost(room_capacity, version)
            profit = revenue - cost

            revenues.append(revenue)
            costs.append(cost)
            profits.append(profit)

            # 按电影统计
            if movie_id not in movie_revenues:
                movie_revenues[movie_id] = 0
            movie_revenues[movie_id] += profit

            # 按房间统计
            if room not in room_revenues:
                room_revenues[room] = 0
            room_revenues[room] += profit

            # 按版本统计
            version_revenues[version] += profit

            # 按时间段统计
            hour = int(showtime.split(':')[0])
            time_key = f'{hour:02d}:00'
            if time_key not in time_revenues:
                time_revenues[time_key] = 0
            time_revenues[time_key] += profit

        # 1. 各电影收益对比
        movies = list(movie_revenues.keys())
        movie_profits = list(movie_revenues.values())
        colors = [self.movie_colors[m] for m in movies]

        bars1 = ax1.bar(movies, movie_profits, color=colors, alpha=0.7)
        ax1.set_title('各电影净收益对比', fontsize=14, fontweight='bold')
        ax1.set_xlabel('电影ID')
        ax1.set_ylabel('净收益 (元)')
        ax1.tick_params(axis='x', rotation=45)

        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.0f}', ha='center', va='bottom')

        # 2. 各房间收益对比
        rooms = list(room_revenues.keys())
        room_profits = list(room_revenues.values())

        bars2 = ax2.bar(rooms, room_profits, color='skyblue', alpha=0.7)
        ax2.set_title('各放映厅净收益对比', fontsize=14, fontweight='bold')
        ax2.set_xlabel('放映厅')
        ax2.set_ylabel('净收益 (元)')

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.0f}', ha='center', va='bottom')

        # 3. 版本收益分析
        versions = list(version_revenues.keys())
        version_profits = list(version_revenues.values())

        wedges, texts, autotexts = ax3.pie(version_profits, labels=versions, autopct='%1.1f%%',
                                           colors=['lightcoral', 'lightblue', 'lightgreen'])
        ax3.set_title('各版本收益占比', fontsize=14, fontweight='bold')

        # 4. 时间段收益分析
        sorted_times = sorted(time_revenues.items())
        times = [t[0] for t in sorted_times]
        time_profits = [t[1] for t in sorted_times]

        ax4.plot(times, time_profits, marker='o', linewidth=2, markersize=6)
        ax4.set_title('各时间段净收益趋势', fontsize=14, fontweight='bold')
        ax4.set_xlabel('时间')
        ax4.set_ylabel('净收益 (元)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_constraint_analysis(self, save_path=None):
        """分析约束满足情况"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 题材播放次数分析
        genre_count = {}
        for item in self.schedule:
            movie = self.optimizer.movies_df[self.optimizer.movies_df['id'] == item['id']].iloc[0]
            movie_genres = [g.strip() for g in movie['genres'].split(',')]
            for genre in movie_genres:
                genre_count[genre] = genre_count.get(genre, 0) + 1

        genres = list(self.optimizer.genre_limits.keys())
        counts = [genre_count.get(g, 0) for g in genres]
        min_limits = [self.optimizer.genre_limits[g].get('min', 0) for g in genres]
        max_limits = [self.optimizer.genre_limits[g].get('max', 0) for g in genres]

        x = np.arange(len(genres))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, counts, width, label='实际播放次数', color='skyblue', alpha=0.7)
        bars2 = ax1.bar(x + width / 2, max_limits, width, label='最大限制', color='lightcoral', alpha=0.7)

        ax1.set_title('题材播放次数约束满足情况', fontsize=14, fontweight='bold')
        ax1.set_xlabel('题材')
        ax1.set_ylabel('播放次数')
        ax1.set_xticks(x)
        ax1.set_xticklabels(genres, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 添加最小限制线
        for i, min_limit in enumerate(min_limits):
            if min_limit > 0:
                ax1.axhline(y=min_limit, xmin=(i - 0.4) / len(genres), xmax=(i + 0.4) / len(genres),
                            color='red', linestyle='--', linewidth=2)

        # 2. 版本播放时长分析
        version_duration = {'3D': 0, 'IMAX': 0}
        for item in self.schedule:
            if item['version'] in version_duration:
                movie = self.optimizer.movies_df[self.optimizer.movies_df['id'] == item['id']].iloc[0]
                runtime = self.optimizer._round_up_to_30(movie['runtime'])
                version_duration[item['version']] += runtime

        versions = list(version_duration.keys())
        durations = list(version_duration.values())
        limits = [self.optimizer.version_limits[v]['max'] for v in versions]

        x2 = np.arange(len(versions))
        bars3 = ax2.bar(x2 - width / 2, durations, width, label='实际播放时长', color='lightgreen', alpha=0.7)
        bars4 = ax2.bar(x2 + width / 2, limits, width, label='最大限制', color='orange', alpha=0.7)

        ax2.set_title('版本播放时长约束满足情况', fontsize=14, fontweight='bold')
        ax2.set_xlabel('版本')
        ax2.set_ylabel('播放时长 (分钟)')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(versions)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 各电影播放次数分析
        movie_count = {}
        for item in self.schedule:
            movie_count[item['id']] = movie_count.get(item['id'], 0) + 1

        movie_ids = list(movie_count.keys())
        movie_counts = list(movie_count.values())
        colors = [self.movie_colors[m] for m in movie_ids]

        bars5 = ax3.bar(movie_ids, movie_counts, color=colors, alpha=0.7)
        ax3.set_title('各电影播放次数分布', fontsize=14, fontweight='bold')
        ax3.set_xlabel('电影ID')
        ax3.set_ylabel('播放次数')
        ax3.axhline(y=3, color='red', linestyle='--', label='最大限制(3次)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 时间分布分析
        hour_count = {}
        for item in self.schedule:
            hour = int(item['showtime'].split(':')[0])
            hour_count[hour] = hour_count.get(hour, 0) + 1

        hours = sorted(hour_count.keys())
        counts = [hour_count[h] for h in hours]
        hour_labels = [f'{h:02d}:00' if h < 24 else f'{h - 24:02d}:00' for h in hours]

        # 标记黄金时段
        prime_hours = list(range(18, 21))
        colors = ['gold' if h in prime_hours else 'lightblue' for h in hours]

        bars6 = ax4.bar(range(len(hours)), counts, color=colors, alpha=0.7)
        ax4.set_title('各时间段排片数量分布', fontsize=14, fontweight='bold')
        ax4.set_xlabel('时间')
        ax4.set_ylabel('排片数量')
        ax4.set_xticks(range(len(hours)))
        ax4.set_xticklabels(hour_labels, rotation=45)

        # 添加黄金时段标识
        ax4.axvspan(8, 11, alpha=0.2, color='gold', label='黄金时段(18-21点)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_utilization_analysis(self, save_path=None):
        """分析资源利用率"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 各放映厅利用率
        room_utilization = {}
        total_operating_hours = (self.optimizer.end_hour - self.optimizer.start_hour)

        for room in self.optimizer.cinema_df['room']:
            used_time = 0
            for item in self.schedule:
                if item['room'] == room:
                    movie = self.optimizer.movies_df[self.optimizer.movies_df['id'] == item['id']].iloc[0]
                    runtime = self.optimizer._round_up_to_30(movie['runtime'])
                    used_time += runtime + 15  # 包含清理时间

            utilization = min(used_time / (total_operating_hours * 60), 1.0)
            room_utilization[room] = utilization

        rooms = list(room_utilization.keys())
        utilizations = list(room_utilization.values())

        bars1 = ax1.bar(rooms, utilizations, color='lightgreen', alpha=0.7)
        ax1.set_title('各放映厅利用率', fontsize=14, fontweight='bold')
        ax1.set_xlabel('放映厅')
        ax1.set_ylabel('利用率')
        ax1.set_ylim(0, 1)
        ax1.axhline(y=0.8, color='red', linestyle='--', label='目标利用率(80%)')
        ax1.legend()

        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1%}', ha='center', va='bottom')

        # 2. 座位利用率分析
        room_capacity_utilization = {}
        for room in self.optimizer.cinema_df['room']:
            room_capacity = self.optimizer.cinema_df[
                self.optimizer.cinema_df['room'] == room]['capacity'].iloc[0]
            total_attendance = sum(item['attendance'] for item in self.schedule if item['room'] == room)
            total_capacity = room_capacity * len([item for item in self.schedule if item['room'] == room])

            if total_capacity > 0:
                room_capacity_utilization[room] = total_attendance / total_capacity
            else:
                room_capacity_utilization[room] = 0

        rooms2 = list(room_capacity_utilization.keys())
        capacity_utils = list(room_capacity_utilization.values())

        bars2 = ax2.bar(rooms2, capacity_utils, color='lightcoral', alpha=0.7)
        ax2.set_title('各放映厅座位利用率', fontsize=14, fontweight='bold')
        ax2.set_xlabel('放映厅')
        ax2.set_ylabel('座位利用率')
        ax2.set_ylim(0, 1)

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1%}', ha='center', va='bottom')

        # 3. 版本分布
        version_count = {}
        for item in self.schedule:
            version = item['version']
            version_count[version] = version_count.get(version, 0) + 1

        versions = list(version_count.keys())
        counts = list(version_count.values())

        wedges, texts, autotexts = ax3.pie(counts, labels=versions, autopct='%1.1f%%',
                                           colors=['lightblue', 'lightgreen', 'lightyellow'])
        ax3.set_title('放映版本分布', fontsize=14, fontweight='bold')

        # 4. 热力图：时间 vs 放映厅
        # 创建时间-房间矩阵
        time_room_matrix = np.zeros((len(self.optimizer.time_slots), len(self.optimizer.cinema_df)))

        for i, time_slot in enumerate(self.optimizer.time_slots):
            for j, room in enumerate(self.optimizer.cinema_df['room']):
                # 检查该时间段该房间是否有排片
                for item in self.schedule:
                    if item['room'] == room:
                        movie = self.optimizer.movies_df[self.optimizer.movies_df['id'] == item['id']].iloc[0]
                        runtime = self.optimizer._round_up_to_30(movie['runtime'])
                        start_minutes = self.optimizer._time_slot_to_minutes(item['showtime'])
                        end_minutes = start_minutes + runtime
                        current_minutes = self.optimizer._time_slot_to_minutes(time_slot)

                        if start_minutes <= current_minutes < end_minutes:
                            time_room_matrix[i, j] = item['id']

        # 只显示部分时间段以提高可读性
        step = max(1, len(self.optimizer.time_slots) // 20)
        time_indices = range(0, len(self.optimizer.time_slots), step)
        time_labels = [self.optimizer.time_slots[i] for i in time_indices]

        im = ax4.imshow(time_room_matrix[::step, :], aspect='auto', cmap='tab20')
        ax4.set_title('时间-放映厅占用热力图', fontsize=14, fontweight='bold')
        ax4.set_xlabel('放映厅')
        ax4.set_ylabel('时间')
        ax4.set_xticks(range(len(self.optimizer.cinema_df)))
        ax4.set_xticklabels(self.optimizer.cinema_df['room'])
        ax4.set_yticks(range(len(time_labels)))
        ax4.set_yticklabels(time_labels)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_interactive_dashboard(self, save_path=None):
        """创建交互式仪表盘"""
        # 准备数据
        df_schedule = pd.DataFrame(self.schedule)

        # 计算收益数据
        revenue_data = []
        for item in self.schedule:
            room = item['room']
            movie_id = item['id']
            version = item['version']
            showtime = item['showtime']
            attendance = item['attendance']

            room_capacity = self.optimizer.cinema_df[
                self.optimizer.cinema_df['room'] == room]['capacity'].iloc[0]
            movie = self.optimizer.movies_df[self.optimizer.movies_df['id'] == movie_id].iloc[0]

            is_prime = self.optimizer._is_prime_time(showtime)
            ticket_price = self.optimizer._calculate_ticket_price(movie_id, version, is_prime)
            sharing_rate = self.optimizer._get_sharing_rate(movie_id)

            revenue = ticket_price * attendance * (1 - sharing_rate)
            cost = self.optimizer._calculate_cost(room_capacity, version)
            profit = revenue - cost

            revenue_data.append({
                'room': room,
                'movie_id': movie_id,
                'version': version,
                'showtime': showtime,
                'attendance': attendance,
                'revenue': revenue,
                'cost': cost,
                'profit': profit,
                'is_prime': is_prime
            })

        df_revenue = pd.DataFrame(revenue_data)

        # 创建子图 - 修正饼图的specs
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('排片时间表', '收益分析', '版本分布', '时间分布'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "pie"}, {"secondary_y": False}]]  # 指定第三个位置为饼图
        )

        # 1. 甘特图风格的排片表
        rooms = sorted(list(set([item['room'] for item in self.schedule])))
        room_y_mapping = {room: i for i, room in enumerate(rooms)}

        for i, item in enumerate(self.schedule):
            movie = self.optimizer.movies_df[self.optimizer.movies_df['id'] == item['id']].iloc[0]
            runtime = self.optimizer._round_up_to_30(movie['runtime'])

            hour, minute = map(int, item['showtime'].split(':'))
            start_time = hour + minute / 60.0
            end_time = start_time + runtime / 60.0

            fig.add_trace(
                go.Scatter(
                    x=[start_time, end_time],
                    y=[room_y_mapping[item['room']], room_y_mapping[item['room']]],
                    mode='lines',
                    line=dict(width=20, color=f'rgba({(i * 50) % 255}, {(i * 80) % 255}, {(i * 120) % 255}, 0.7)'),
                    name=f'电影{item["id"]}',
                    showlegend=False,
                    hovertemplate=f'电影{item["id"]}<br>房间: {item["room"]}<br>时间: {item["showtime"]}<br>版本: {item["version"]}<extra></extra>'
                ),
                row=1, col=1
            )

        # 更新第一个子图的y轴
        fig.update_yaxes(
            tickmode='array',
            tickvals=list(range(len(rooms))),
            ticktext=rooms,
            row=1, col=1
        )

        # 2. 收益柱状图
        movie_profits = df_revenue.groupby('movie_id')['profit'].sum().reset_index()
        fig.add_trace(
            go.Bar(
                x=movie_profits['movie_id'],
                y=movie_profits['profit'],
                name='净收益',
                showlegend=False,
                marker_color='lightblue',
                hovertemplate='电影%{x}<br>净收益: ¥%{y:.2f}<extra></extra>'
            ),
            row=1, col=2
        )

        # 3. 版本饼图
        version_counts = df_schedule['version'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=version_counts.index,
                values=version_counts.values,
                name="版本分布",
                showlegend=False,
                hovertemplate='%{label}<br>数量: %{value}<br>占比: %{percent}<extra></extra>'
            ),
            row=2, col=1
        )

        # 4. 时间分布
        df_schedule['hour'] = df_schedule['showtime'].str.split(':').str[0].astype(int)
        hour_counts = df_schedule['hour'].value_counts().sort_index()

        # 转换小时显示格式
        hour_labels = []
        for h in hour_counts.index:
            if h < 24:
                hour_labels.append(f'{h:02d}:00')
            else:
                hour_labels.append(f'{(h - 24):02d}:00')

        fig.add_trace(
            go.Scatter(
                x=hour_labels,
                y=hour_counts.values,
                mode='lines+markers',
                name='排片数量',
                showlegend=False,
                line=dict(color='green', width=3),
                marker=dict(size=8),
                hovertemplate='时间: %{x}<br>排片数量: %{y}<extra></extra>'
            ),
            row=2, col=2
        )

        # 更新布局
        fig.update_layout(
            title_text="影院排片优化可视化仪表盘",
            title_font_size=20,
            height=800,
            showlegend=False
        )

        # 更新各子图的标题和标签
        fig.update_xaxes(title_text="时间(小时)", row=1, col=1)
        fig.update_yaxes(title_text="放映厅", row=1, col=1)

        fig.update_xaxes(title_text="电影ID", row=1, col=2)
        fig.update_yaxes(title_text="净收益(元)", row=1, col=2)

        fig.update_xaxes(title_text="时间", row=2, col=2)
        fig.update_yaxes(title_text="排片数量", row=2, col=2)

        # 保存或显示
        if save_path:
            pyo.plot(fig, filename=save_path, auto_open=True)
        else:
            fig.show()

    def generate_summary_report(self, save_path=None):
        """生成总结报告"""
        # 计算关键指标
        total_shows = len(self.schedule)

        # 收益计算
        total_revenue = 0
        total_cost = 0
        for item in self.schedule:
            room = item['room']
            movie_id = item['id']
            version = item['version']
            showtime = item['showtime']
            attendance = item['attendance']

            room_capacity = self.optimizer.cinema_df[
                self.optimizer.cinema_df['room'] == room]['capacity'].iloc[0]
            movie = self.optimizer.movies_df[self.optimizer.movies_df['id'] == movie_id].iloc[0]

            is_prime = self.optimizer._is_prime_time(showtime)
            ticket_price = self.optimizer._calculate_ticket_price(movie_id, version, is_prime)
            sharing_rate = self.optimizer._get_sharing_rate(movie_id)

            revenue = ticket_price * attendance * (1 - sharing_rate)
            cost = self.optimizer._calculate_cost(room_capacity, version)

            total_revenue += revenue
            total_cost += cost

        total_profit = total_revenue - total_cost

        # 生成报告
        report = f"""
=== 影院排片优化结果总结报告 ===

📊 基本统计:
• 总排片场次: {total_shows} 场
• 总收入: ¥{total_revenue:,.2f}
• 总成本: ¥{total_cost:,.2f}
• 净利润: ¥{total_profit:,.2f}
• 利润率: {total_profit / total_revenue * 100:.1f}%

电影统计:
"""

        # 各电影统计
        movie_stats = {}
        for item in self.schedule:
            movie_id = item['id']
            if movie_id not in movie_stats:
                movie_stats[movie_id] = {'count': 0, 'attendance': 0}
            movie_stats[movie_id]['count'] += 1
            movie_stats[movie_id]['attendance'] += item['attendance']

        for movie_id, stats in movie_stats.items():
            movie = self.optimizer.movies_df[self.optimizer.movies_df['id'] == movie_id].iloc[0]
            report += f"• 电影{movie_id}: {stats['count']}场, 总观众{stats['attendance']}人\n"

        report += f"""
放映厅统计:
"""
        # 各房间统计
        room_stats = {}
        for item in self.schedule:
            room = item['room']
            if room not in room_stats:
                room_stats[room] = 0
            room_stats[room] += 1

        for room, count in room_stats.items():
            report += f"• {room}: {count}场\n"

        report += f"""
版本分布:
"""
        # 版本统计
        version_stats = {}
        for item in self.schedule:
            version = item['version']
            version_stats[version] = version_stats.get(version, 0) + 1

        for version, count in version_stats.items():
            percentage = count / total_shows * 100
            report += f"• {version}: {count}场 ({percentage:.1f}%)\n"

        report += f"""
时间分布:
"""
        # 时间分布统计
        prime_time_shows = sum(1 for item in self.schedule
                               if self.optimizer._is_prime_time(item['showtime']))
        prime_percentage = prime_time_shows / total_shows * 100
        report += f"• 黄金时段排片: {prime_time_shows}场 ({prime_percentage:.1f}%)\n"
        report += f"• 非黄金时段排片: {total_shows - prime_time_shows}场 ({100 - prime_percentage:.1f}%)\n"

        # 约束满足情况
        report += f"""
约束满足情况:
"""

        # 题材约束
        genre_count = {}
        for item in self.schedule:
            movie = self.optimizer.movies_df[self.optimizer.movies_df['id'] == item['id']].iloc[0]
            movie_genres = [g.strip() for g in movie['genres'].split(',')]
            for genre in movie_genres:
                genre_count[genre] = genre_count.get(genre, 0) + 1

        for genre, limits in self.optimizer.genre_limits.items():
            count = genre_count.get(genre, 0)
            min_limit = limits.get('min', 0)
            max_limit = limits.get('max', '∞')
            status = "✅" if count >= min_limit and (max_limit == '∞' or count <= max_limit) else "❌"
            report += f"• {genre}题材: {count}场 (要求: {min_limit}-{max_limit}) {status}\n"

        # 版本时长约束
        version_duration = {'3D': 0, 'IMAX': 0}
        for item in self.schedule:
            if item['version'] in version_duration:
                movie = self.optimizer.movies_df[self.optimizer.movies_df['id'] == item['id']].iloc[0]
                runtime = self.optimizer._round_up_to_30(movie['runtime'])
                version_duration[item['version']] += runtime

        for version in ['3D', 'IMAX']:
            duration = version_duration[version]
            max_limit = self.optimizer.version_limits[version]['max']
            status = "✅" if duration <= max_limit else "❌"
            report += f"• {version}版本总时长: {duration}分钟 (上限: {max_limit}分钟) {status}\n"

        report += f"""
优化建议:
"""

        # 生成建议
        suggestions = []

        # 利润率建议
        if total_profit / total_revenue < 0.3:
            suggestions.append("• 考虑调整票价策略或降低运营成本以提高利润率")

        # 黄金时段建议
        if prime_percentage < 60:
            suggestions.append("• 可以增加黄金时段(18-21点)的排片以提高收益")

        # 房间利用率建议
        min_room_shows = min(room_stats.values())
        max_room_shows = max(room_stats.values())
        if max_room_shows - min_room_shows > 2:
            suggestions.append("• 各放映厅排片数量差异较大，建议优化资源配置")

        # 版本分布建议
        if version_stats.get('IMAX', 0) / total_shows > 0.3:
            suggestions.append("• IMAX排片较多，注意控制特殊版本成本")

        if not suggestions:
            suggestions.append("• 当前排片方案整体表现良好，建议保持")

        for suggestion in suggestions:
            report += f"{suggestion}\n"

        report += "\n=== 报告生成完成 ===\n"

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"报告已保存到: {save_path}")

        return report

    def plot_all_visualizations(self, output_dir='visualizations'):
        """生成所有可视化图表"""
        import os

        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("正在生成可视化图表...")

        # 1. 甘特图
        print("1. 生成排片甘特图...")
        self.plot_schedule_gantt(f"{output_dir}/schedule_gantt.png")

        # 2. 收益分析
        print("2. 生成收益分析图...")
        self.plot_revenue_analysis(f"{output_dir}/revenue_analysis.png")

        # 3. 约束分析
        print("3. 生成约束分析图...")
        self.plot_constraint_analysis(f"{output_dir}/constraint_analysis.png")

        # 4. 资源利用率分析
        print("4. 生成资源利用率分析图...")
        self.plot_utilization_analysis(f"{output_dir}/utilization_analysis.png")

        # 5. 交互式仪表盘
        print("5. 生成交互式仪表盘...")
        self.create_interactive_dashboard(f"{output_dir}/interactive_dashboard.html")

        # 6. 总结报告
        print("6. 生成总结报告...")
        report = self.generate_summary_report(f"{output_dir}/summary_report.txt")
        print("\n" + "=" * 50)
        print(report)
        print("=" * 50)

        print(f"\n所有可视化文件已保存到: {output_dir}/")
        print("包含文件:")
        print("- schedule_gantt.png: 排片甘特图")
        print("- revenue_analysis.png: 收益分析")
        print("- constraint_analysis.png: 约束分析")
        print("- utilization_analysis.png: 资源利用率分析")
        print("- interactive_dashboard.html: 交互式仪表盘")
        print("- summary_report.txt: 详细报告")


# 修改原始main函数以集成可视化
def main_with_visualization():
    """带可视化的主函数"""
    from milp_copt import CinemaSchedulingOptimizer  # 导入您的优化器

    # 创建优化器实例
    optimizer = CinemaSchedulingOptimizer(
        'D:\PythonProjects\MCM\input_data\df_cinema.csv',
        'D:\PythonProjects\MCM\input_data\df_movies_schedule.csv'
    )

    # 执行优化
    print("开始优化排片计划...")
    schedule, status, objective_value = optimizer.optimize_schedule()

    if status == 1:
        print(f"优化成功！最大净收益: {objective_value:.2f} 元")

        # 创建可视化器
        visualizer = CinemaSchedulingVisualizer(optimizer, schedule)

        # 生成所有可视化图表
        visualizer.plot_all_visualizations('cinema_optimization_results')

        # 单独展示关键图表
        print("\n正在显示关键可视化图表...")

        # 显示甘特图
        print("显示排片甘特图...")
        visualizer.plot_schedule_gantt()

        # 显示收益分析
        print("显示收益分析...")
        visualizer.plot_revenue_analysis()

        # 输出优化结果到CSV
        result_df = pd.DataFrame(schedule)
        result_df.to_csv('D:\PythonProjects\MCM\output_result\df_result_2.csv', index=False)

        print(f"\n排片计划已保存到 df_result_2.csv")
        print(f"总共安排了 {len(schedule)} 场放映")

    else:
        print(f"优化失败，状态码: {status}")


if __name__ == "__main__":
    # 如果您想要运行带可视化的版本，请使用：
    main_with_visualization()

    # 或者如果您已经有了schedule结果，可以直接创建可视化：
    # visualizer = CinemaSchedulingVisualizer(optimizer, schedule)
    # visualizer.plot_all_visualizations()