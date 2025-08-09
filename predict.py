import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载数据
try:
    df = pd.read_csv('df_movies_cleaned.csv')
    print("数据加载成功！")

    # # 2. 显示数据前几行
    # print("\n数据前5行：")
    # print(df.head())
    #
    # # 3. 数据基本信息
    # print("\n数据基本信息：")
    # print(df.info())
    #
    # # 4. 描述性统计
    # print("\n数值列的描述性统计：")
    # print(df.describe())
    #
    # # 5. 检查缺失值
    # print("\n缺失值统计：")
    # print(df.isnull().sum())

    # # 6. 初步可视化 - 评分分布
    # if 'rating' in df.columns:
    #     plt.figure(figsize=(10, 6))
    #     sns.histplot(df['rating'], bins=20, kde=True)
    #     plt.title('电影评分分布')
    #     plt.xlabel('评分')
    #     plt.ylabel('数量')
    #     plt.show()
    # else:
    #     print("\n数据中没有'rating'列，请确认评分列的名称")
    #
    # # 7. 可能的关联分析 - 如果有其他数值列
    # numeric_cols = df.select_dtypes(include=[np.number]).columns
    # if len(numeric_cols) > 1:
    #     print("\n数值特征间的相关性：")
    #     corr_matrix = df[numeric_cols].corr()
    #     print(corr_matrix)
    #
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    #     plt.title('特征相关性热图')
    #     plt.show()

except FileNotFoundError:
    print("文件未找到，请确认文件路径和名称是否正确。")
except Exception as e:
    print(f"发生错误: {e}")

from sklearn.preprocessing import LabelEncoder

# 预处理分类变量（简单编码）
df_analysis = df.copy()
for col in ['main_genre', 'lang']:
    le = LabelEncoder()
    df_analysis[col] = le.fit_transform(df_analysis[col])

# 选择特征和目标变量
features = ['main_genre', 'lang', 'runtime', 'cast_count', 'writers_count', 'director_known']
X = df_analysis[features]
y = df_analysis['rating']

# 训练线性模型
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

# 输出系数权重
coef_df = pd.DataFrame({'feature': features, 'coefficient': model.coef_})
coef_df = coef_df.sort_values('coefficient', key=abs, ascending=False)
print("\n线性回归系数（绝对值越大影响越强）：")
print(coef_df)