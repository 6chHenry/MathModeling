import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def overall_distribution(df):
    plt.figure(figsize=(8, 4))
    sns.histplot(df['rating'], bins=50, kde=True, color='#4c72b0')
    plt.title('Distribution of Movie Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
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
    plt.show()

def runtime_vs_rating(df):
    plt.figure(figsize=(8, 5))
    sns.regplot(x='runtime', y='rating', data=df, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    plt.title('Rating vs Runtime')
    plt.xlabel('Runtime (min)')
    plt.ylabel('Rating')
    plt.show()

    # 计算相关系数
    print('Pearson r =', df['runtime'].corr(df['rating']))
if __name__ == '__main__':
    df = pd.read_csv('df_movies_cleaned.csv')
    overall_distribution(df)
    language_vs_rating(df)
    main_genre_vs_rating(df)
    runtime_vs_rating(df)