import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据
df = pd.read_csv('input_data/df_movies_train.csv')

# 2. 基础信息概览
print("数据维度：", df.shape)
print("\n前5行：")
print(df.head())

print("\n字段类型：")
print(df.dtypes)

print("\n缺失值统计：")
print(df.isnull().sum())

# 3. 初步清洗
# 3.1 填充缺失值
# 文本类字段用 'Unknown'
text_cols = ['genres', 'cast', 'director', 'writers', 'production_companies', 'producers']
for col in text_cols:
    df[col] = df[col].fillna('Unknown')

# 数值类字段 runtime 用中位数填充
runtime_median = df['runtime'].median()
df['runtime'] = df['runtime'].fillna(runtime_median)

# 3.2 创建辅助特征
# 3.2.1 主要类型（取 genres 的第一段）
df['main_genre'] = df['genres'].str.split(',').str[0].str.strip()
df['main_genre'] = df['main_genre'].replace('', 'Unknown')

# 3.2.2 语言编码（保留原值即可）
df['lang'] = df['original_language'].fillna('Unknown')

# 3.2.3 cast 和 writers 的数量
df['cast_count'] = df['cast'].apply(lambda x: len(str(x).split(',')) if x != 'Unknown' else 0)
df['writers_count'] = df['writers'].apply(lambda x: len(str(x).split(',')) if x != 'Unknown' else 0)

# 3.2.4 是否有已知导演（1/0）
df['director_known'] = (df['director'] != 'Unknown').astype(int)

# 4. 再次查看清洗后数据
print("\n清洗后缺失值：")
print(df.isnull().sum())

print("\n新增字段示例：")
print(df[['main_genre', 'lang', 'cast_count', 'writers_count', 'director_known']].head())

# 5. 保存为csv
df.to_csv('df_movies_cleaned.csv', index=False)

# 6.可视化
def overall_distribution(df):
    plt.figure(figsize=(8, 4))
    sns.histplot(df['rating'], bins=50, kde=True, color='#4c72b0')
    plt.title('Distribution of Movie Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig('overall_distribution.png')
    plt.show()

def language_vs_rating(df):
    lang_stats = (
        df.groupby('lang')['rating']
        .agg(['mean', 'count'])
        .sort_values('mean', ascending=False)
    )

    # 只看样本量>=5 的语言
    lang_stats = lang_stats[lang_stats['count'] >= 5]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=lang_stats.index, y='mean', data=lang_stats.reset_index(), palette='viridis')
    plt.title('Average Rating by Language (n≥5)')
    plt.xticks(rotation=45)
    plt.ylabel('Mean Rating')
    plt.savefig('language_vs_rating.png')
    plt.show()

def main_genre_vs_rating(df):
    genre_stats = (
        df.groupby('main_genre')['rating']
        .agg(['mean', 'count'])
        .sort_values('mean', ascending=False)
    )

    # 只看样本量>=10 的类型
    genre_stats = genre_stats[genre_stats['count'] >= 10]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=genre_stats.index, y='mean', data=genre_stats.reset_index(), palette='magma')
    plt.title('Average Rating by Main Genre (n≥10)')
    plt.xticks(rotation=45)
    plt.ylabel('Mean Rating')
    plt.savefig('main_genre_vs_rating.png')
    plt.show()

def runtime_vs_rating(df):
    plt.figure(figsize=(8, 5))
    sns.regplot(x='runtime', y='rating', data=df, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    plt.title('Rating vs Runtime')
    plt.xlabel('Runtime (min)')
    plt.ylabel('Rating')
    plt.show()
    plt.savefig('runtime_vs_rating.png')
    # 计算相关系数
    print('Pearson r =', df['runtime'].corr(df['rating']))

def genre_runtime_vs_rating(df):
    heatmap_data = df.groupby(['main_genre', 'runtime_bin'])['rating'].mean().unstack()
    fig, (ax1, ax2) = plt.subplots(
    ncols=2,
    figsize=(18, 6),
    gridspec_kw={'width_ratios': [1, 2]}
    )

    # 左侧：热力图
    sns.heatmap(
        heatmap_data,
        cmap='YlOrRd',
        annot=True,
        fmt='.1f',
        ax=ax1,
        cbar=False
    )
    ax1.set_title('Average Rating by Genre & Runtime', fontsize=12)
    ax1.set_xlabel('')
    ax1.set_ylabel('Genre')

    # 右侧：箱线图
    sns.boxplot(
        data=df,
        x='main_genre',
        y='rating',
        hue='runtime_bin',
        palette='Blues',
        ax=ax2,
        showfliers=False  # 不显示异常值
    )
    ax2.set_title('Rating Distribution by Genre and Runtime', fontsize=12)
    ax2.legend(title='Runtime', bbox_to_anchor=(1, 1))
    ax2.set_xlabel('')
    ax2.set_ylabel('')

    plt.tight_layout()
    plt.savefig('combined_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def main():
    df = pd.read_csv('df_movies_cleaned.csv')
    overall_distribution(df)
    language_vs_rating(df)
    main_genre_vs_rating(df)
    runtime_vs_rating(df)

if __name__ == '__main__':
    main()