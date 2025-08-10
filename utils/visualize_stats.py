import json
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict


# 加载统计文件
def load_stats(file_path='../movie_stats.json'):
    with open(file_path, 'r') as f:
        return json.load(f)


# 提取每个类别前20名的数据（其余归为Others）
def get_top_items(stats, category, top_n=20):
    items = []
    for name, data in stats[category].items():
        if data['count'] > 0:  # 确保有有效数据
            items.append({
                'name': name,
                'average_rating': data['average_rating'],
                'count': data['count']
            })
    # 按平均评分降序排序
    items_sorted = sorted(items, key=lambda x: x['average_rating'], reverse=True)

    # 分离前top_n和Others
    top_items = items_sorted[:top_n]
    others_avg = sum(item['average_rating'] for item in items_sorted[top_n:]) / max(1, len(items_sorted[top_n:]))
    others_count = sum(item['count'] for item in items_sorted[top_n:])

    if others_count > 0:
        top_items.append({
            'name': 'Others',
            'average_rating': others_avg,
            'count': others_count
        })
    return pd.DataFrame(top_items)


# 可视化函数
def plot_rating_distribution(df, category, color='skyblue'):
    plt.figure(figsize=(12, 8))
    bars = plt.barh(
        df['name'], df['average_rating'],
        color=color,
        edgecolor='black'
    )

    # 添加数据标签
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width - 0.3, bar.get_y() + bar.get_height() / 2,
            f'{width:.2f}',
            ha='right', va='center',
            color='white', fontweight='bold'
        )

    plt.title(f'{category.capitalize()} Average Rating (Top 20 + Others)', fontsize=14)
    plt.xlabel('Average Rating', fontsize=12)
    plt.ylabel(category.capitalize(), fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{category}_ratings.png', dpi=300, bbox_inches='tight')
    plt.show()


# 主流程
if __name__ == '__main__':
    stats = load_stats()

    # 定义颜色方案
    colors = {
        'actors': '#1f77b4',
        'directors': '#ff7f0e',
        'writers': '#2ca02c',
        'producers': '#d62728',
        'genres': '#9467bd'
    }

    # 逐个类别可视化
    for category in ['actors', 'directors', 'writers', 'producers', 'genres']:
        df = get_top_items(stats, category)
        plot_rating_distribution(df, category, color=colors[category])