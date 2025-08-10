import json
import pandas as pd
from collections import defaultdict
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

class CollaborationAnalyzer:
    def __init__(self, movies_data_path, stats_path):
        """
        初始化分析器
        
        Args:
            movies_data_path: 电影数据CSV文件路径
            stats_path: 统计数据JSON文件路径
        """
        self.movies_df = pd.read_csv(movies_data_path)
        with open(stats_path, 'r', encoding='utf-8') as f:
            self.stats = json.load(f)
        
        # 存储协作组合的统计信息
        self.collaborations = defaultdict(list)
        self.genre_collaborations = defaultdict(list)
        
    def parse_person_list(self, person_string):
        """解析人员字符串，返回人员列表"""
        if pd.isna(person_string) or person_string == "":
            return []
        return [person.strip() for person in person_string.split(',')]
    
    def analyze_collaborations(self, min_collaborations=3, min_rating=7.0):
        """
        分析各种类型的协作组合
        
        Args:
            min_collaborations: 最少合作次数
            min_rating: 最低平均评分阈值
        """
        print("正在分析协作组合...")
        
        # 分析不同角色之间的协作
        for idx, row in self.movies_df.iterrows():
            if pd.isna(row['rating']) or row['rating'] < min_rating:
                continue
                
            rating = row['rating']
            genres = self.parse_person_list(row['genres'])
            actors = self.parse_person_list(row['cast'])
            directors = self.parse_person_list(row['director'])
            writers = self.parse_person_list(row['writers'])
            producers = self.parse_person_list(row['producers'])
            
            # 分析演员-导演组合
            for actor in actors:
                for director in directors:
                    combo_key = f"Actor-Director: {actor} & {director}"
                    self.collaborations[combo_key].append(rating)
                    
                    # 按类型分析
                    for genre in genres:
                        genre_key = f"{genre}|Actor-Director: {actor} & {director}"
                        self.genre_collaborations[genre_key].append(rating)
            
            # 分析演员-编剧组合
            for actor in actors:
                for writer in writers:
                    combo_key = f"Actor-Writer: {actor} & {writer}"
                    self.collaborations[combo_key].append(rating)
                    
                    for genre in genres:
                        genre_key = f"{genre}|Actor-Writer: {actor} & {writer}"
                        self.genre_collaborations[genre_key].append(rating)
            
            # 分析导演-编剧组合
            for director in directors:
                for writer in writers:
                    combo_key = f"Director-Writer: {director} & {writer}"
                    self.collaborations[combo_key].append(rating)
                    
                    for genre in genres:
                        genre_key = f"{genre}|Director-Writer: {director} & {writer}"
                        self.genre_collaborations[genre_key].append(rating)
            
            # 分析导演-制片人组合
            for director in directors:
                for producer in producers:
                    combo_key = f"Director-Producer: {director} & {producer}"
                    self.collaborations[combo_key].append(rating)
                    
                    for genre in genres:
                        genre_key = f"{genre}|Director-Producer: {director} & {producer}"
                        self.genre_collaborations[genre_key].append(rating)
            
            # 分析演员组合（2人组合）
            for actor1, actor2 in combinations(actors, 2):
                combo_key = f"Actor-Actor: {actor1} & {actor2}"
                self.collaborations[combo_key].append(rating)
                
                for genre in genres:
                    genre_key = f"{genre}|Actor-Actor: {actor1} & {actor2}"
                    self.genre_collaborations[genre_key].append(rating)
        
        # 筛选符合条件的组合
        self.golden_combinations = self._filter_golden_combinations(min_collaborations, min_rating)
        self.genre_golden_combinations = self._filter_genre_golden_combinations(min_collaborations, min_rating)
    
    def _filter_golden_combinations(self, min_collaborations, min_rating):
        """筛选黄金组合（总体）"""
        golden_combos = {}
        
        for combo, ratings in self.collaborations.items():
            if len(ratings) >= min_collaborations:
                avg_rating = np.mean(ratings)
                if avg_rating >= min_rating:
                    golden_combos[combo] = {
                        'count': len(ratings),
                        'average_rating': avg_rating,
                        'ratings': ratings,
                        'min_rating': min(ratings),
                        'max_rating': max(ratings)
                    }
        
        return dict(sorted(golden_combos.items(), 
                          key=lambda x: (x[1]['average_rating'], x[1]['count']), 
                          reverse=True))
    
    def _filter_genre_golden_combinations(self, min_collaborations, min_rating):
        """筛选特定类型的黄金组合"""
        genre_golden_combos = defaultdict(dict)
        
        for combo_key, ratings in self.genre_collaborations.items():
            if len(ratings) >= min_collaborations:
                avg_rating = np.mean(ratings)
                if avg_rating >= min_rating:
                    genre, combo = combo_key.split('|', 1)
                    genre_golden_combos[genre][combo] = {
                        'count': len(ratings),
                        'average_rating': avg_rating,
                        'ratings': ratings,
                        'min_rating': min(ratings),
                        'max_rating': max(ratings)
                    }
        
        # 对每个类型的组合进行排序
        for genre in genre_golden_combos:
            genre_golden_combos[genre] = dict(sorted(
                genre_golden_combos[genre].items(),
                key=lambda x: (x[1]['average_rating'], x[1]['count']),
                reverse=True
            ))
        
        return dict(genre_golden_combos)
    
    def print_golden_combinations(self, top_n=20):
        """打印黄金组合"""
        print(f"\n{'='*80}")
        print(f"🏆 Overall Gold Group TOP {top_n}")
        print(f"{'='*80}")
        
        for i, (combo, stats) in enumerate(list(self.golden_combinations.items())[:top_n], 1):
            print(f"\n{i}. {combo}")
            print(f"   Collaboration Count: {stats['count']}")
            print(f"   Average Rating : {stats['average_rating']:.2f}")
            print(f"   Count Range: {stats['min_rating']:.1f} - {stats['max_rating']:.1f}")
    
    def print_genre_golden_combinations(self, top_n_per_genre=10):
        """打印各类型的黄金组合"""
        print(f"\n{'='*80}")
        print(f"🎬 Golden Group of All Types TOP {top_n_per_genre}")
        print(f"{'='*80}")
        
        for genre, combos in self.genre_golden_combinations.items():
            if not combos:
                continue
                
            print(f"\n📽️ {genre} genre:")
            print("-" * 60)
            
            for i, (combo, stats) in enumerate(list(combos.items())[:top_n_per_genre], 1):
                print(f"  {i}. {combo}")
                print(f"     cooperate {stats['count']} times | average rating {stats['average_rating']:.2f} | "
                      f"rating range {stats['min_rating']:.1f}-{stats['max_rating']:.1f}")

    def visualize_collaborations(self):
        """可视化分析结果"""
        
        # 设置中文字体支持
        import platform
        
        # 根据操作系统设置字体
        if platform.system() == 'Windows':
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
        elif platform.system() == 'Darwin':  # macOS
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans GB', 'STFangsong']
        else:  # Linux
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
        
        plt.rcParams['axes.unicode_minus'] = False
        
        # 备用方案：如果字体不可用，使用英文标签
        try:
            # 测试中文字体是否可用
            fig_test, ax_test = plt.subplots(figsize=(1, 1))
            ax_test.text(0.5, 0.5, '测试中文', fontsize=10)
            plt.close(fig_test)
            
            # 中文标签
            title_main = '演职人员协作"绝配"组合分析'
            title_rating_dist = 'Top 12 黄金组合评分分布'
            title_collab_scatter = '合作次数 vs 平均评分关系'
            title_genre_count = '各类型黄金组合数量'
            title_type_dist = '协作类型分布'
            label_rating = '评分'
            label_combination = '组合'
            label_collab_count = '合作次数'
            label_avg_rating = '平均评分'
            label_golden_count = '黄金组合数量'
            
        except Exception:
            print("Warning: 中文字体不可用，将使用英文标签")
            # 英文标签
            title_main = 'Cast & Crew "Perfect Match" Collaboration Analysis'
            title_rating_dist = 'Top 12 Golden Combo Rating Distribution'
            title_collab_scatter = 'Collaboration Count vs Average Rating'
            title_genre_count = 'Golden Combo Count by Genre'
            title_type_dist = 'Collaboration Type Distribution'
            label_rating = 'Rating'
            label_combination = 'Combination'
            label_collab_count = 'Collaboration Count'
            label_avg_rating = 'Average Rating'
            label_golden_count = 'Golden Combo Count'
        
        # 创建更大的图形，增加子图间距
        fig, axes = plt.subplots(2, 2, figsize=(24, 18))
        fig.suptitle(title_main, fontsize=20, fontweight='bold', y=0.95)
        
        # 黄金组合评分分布
        ratings_data = []
        combo_names = []
        for combo, stats in list(self.golden_combinations.items())[:10]:  # 进一步减少数量
            ratings_data.extend(stats['ratings'])
            # 进一步缩短名称并处理长名字
            short_name = combo.split(': ')[1][:20]  # 缩短到20个字符
            combo_names.extend([short_name] * len(stats['ratings']))
        
        if ratings_data:
            df_plot = pd.DataFrame({'Combination': combo_names, 'Rating': ratings_data})
            sns.boxplot(data=df_plot, y='Combination', x='Rating', ax=axes[0,0])
            axes[0,0].set_title(title_rating_dist, fontsize=12, fontweight='bold', pad=15)
            axes[0,0].set_xlabel(label_rating, fontsize=10)
            axes[0,0].set_ylabel(label_combination, fontsize=10)
            axes[0,0].tick_params(axis='y', labelsize=7)  # 进一步减小字体
            axes[0,0].tick_params(axis='x', labelsize=9)
            # 确保y轴标签完全可见
            axes[0,0].yaxis.set_tick_params(pad=2)
            plt.setp(axes[0,0].get_yticklabels(), ha='right')
        
        # 合作次数 vs 平均评分散点图
        counts = [stats['count'] for stats in self.golden_combinations.values()]
        avg_ratings = [stats['average_rating'] for stats in self.golden_combinations.values()]
        
        scatter = axes[0,1].scatter(counts, avg_ratings, alpha=0.7, s=50, c=avg_ratings, cmap='viridis')
        axes[0,1].set_xlabel(label_collab_count, fontsize=10)
        axes[0,1].set_ylabel(label_avg_rating, fontsize=10)
        axes[0,1].set_title(title_collab_scatter, fontsize=12, fontweight='bold', pad=15)
        axes[0,1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0,1], label=label_avg_rating)

        # 各类型黄金组合数量
        genre_counts = {genre: len(combos) for genre, combos in self.genre_golden_combinations.items()}
        if genre_counts:
            sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:12]  # 减少数量
            genres, counts = zip(*sorted_genres)
            
            bars = axes[1,0].bar(range(len(genres)), counts, color='skyblue', alpha=0.8)
            axes[1,0].set_xticks(range(len(genres)))
            axes[1,0].set_xticklabels(genres, rotation=45, ha='right', fontsize=9)
            axes[1,0].set_title(title_genre_count, fontsize=12, fontweight='bold', pad=15)
            axes[1,0].set_ylabel(label_golden_count, fontsize=10)
            axes[1,0].grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                              f'{int(height)}', ha='center', va='bottom', fontsize=8)

        # 协作类型分布
        combo_types = defaultdict(int)
        for combo in self.golden_combinations.keys():
            combo_type = combo.split(':')[0]
            combo_types[combo_type] += 1
        
        if combo_types:
            types, counts = zip(*combo_types.items())
            # 创建饼图，不显示图例在饼图内部
            wedges, texts, autotexts = axes[1,1].pie(counts, labels=None, autopct='%1.1f%%', 
                                                    startangle=90, pctdistance=0.85)
            axes[1,1].set_title(title_type_dist, fontsize=12, fontweight='bold', pad=15)
            
            # 调整百分比文本
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_color('black')
                autotext.set_weight('bold')
            
            # 在饼图外部创建图例，避免重叠
            axes[1,1].legend(wedges, types, title="Collaboration Types", 
                           loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                           fontsize=9, title_fontsize=10)

        # 调整子图间距，给左边留更多空间给y轴标签
        plt.subplots_adjust(left=0.12, bottom=0.08, right=0.88, top=0.92, 
                           wspace=0.35, hspace=0.35)
        
        plt.savefig('collaboration_analysis.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
    
    def export_results(self, filename='golden_combinations.json'):
        """导出分析结果"""
        results = {
            'overall_golden_combinations': self.golden_combinations,
            'genre_golden_combinations': self.genre_golden_combinations,
            'analysis_summary': {
                'total_golden_combinations': len(self.golden_combinations),
                'total_genres_analyzed': len(self.genre_golden_combinations),
                'top_combination': list(self.golden_combinations.keys())[0] if self.golden_combinations else None
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n分析结果已导出到: {filename}")

def main():
    # 创建分析器
    analyzer = CollaborationAnalyzer(
        movies_data_path='input_data/df_movies_train.csv',
        stats_path='movie_stats.json'
    )
    
    # 执行分析
    print("开始分析演职人员协作组合...")
    analyzer.analyze_collaborations(min_collaborations=3, min_rating=7.0)
    
    # 显示结果
    analyzer.print_golden_combinations(top_n=20)
    analyzer.print_genre_golden_combinations(top_n_per_genre=10)
    
    # 可视化
    print("\n正在生成可视化图表...")
    analyzer.visualize_collaborations()
    
    # 导出结果
    analyzer.export_results('golden_combinations.json')
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()
