# 0. 环境准备
import pandas as pd
# import numpy as np

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

# 1. 保存（体积小、读写快、保留类型信息）
df.to_csv('df_movies_cleaned.csv', index=False)

# 2. 下次读取
# df = pd.read_parquet('df_movies_cleaned.parquet')
# plt.figure(figsize=(8,4))
# sns.histplot(df['rating'], bins=50, kde=True, color='#4c72b0')
# plt.title('Distribution of Movie Ratings')
# plt.xlabel('Rating')
# plt.ylabel('Count')
# plt.show()
#
# # 基本统计量
# print(df['rating'].describe())
