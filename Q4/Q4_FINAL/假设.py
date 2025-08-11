import pandas as pd
import numpy as np

# === Step 1: 读取影片预测评分 ===
movies = pd.read_csv(r"C:\Users\47797\Desktop\predicted_movies.csv")  # 假设有 'movie_id', 'predicted_rating'
movies = movies[['id', 'predicted_rating']]

# === Step 2: 设置初始参数 ===
k = 0.05   # 评分影响系数（人为设定）
c = 0.3    # 基础热度系数（人为设定）
lambda_decay = 0.05  # 衰减系数 λ（人为设定）

# α_{t,w}：不同时间段 & 星期类型修正系数（人为设定）
alpha_table = {
    (0, 0): 0.30,  # 上午-工作日
    (0, 1): 0.50,  # 上午-周末
    (1, 0): 0.50,  # 下午-工作日
    (1, 1): 0.75,  # 下午-周末
    (2, 0): 0.40,  # 晚上-工作日
    (2, 1): 0.65   # 晚上-周末
}

# === Step 3: 模拟一周数据 ===
days = 7
records = []
np.random.seed(42)  # 固定随机种子，方便复现

for _, row in movies.iterrows():
    movie_id = row['id']
    s_i = row['predicted_rating']
    beta_i = k * s_i + c  # 冷启动阶段 φ_i(t) = 1

    for d in range(days):
        w = 1 if d in [5, 6] else 0  # 周末：星期六(日)
        gamma_d = np.exp(-lambda_decay * d)

        for t in [0, 1, 2]:  # 时间段: 上午, 下午, 晚上
            alpha_tw = alpha_table[(t, w)]

            # 基于模型的 baseline
            baseline = min(1, alpha_tw * beta_i * gamma_d)

            # Step 4: 加入噪声
            noise = np.random.normal(0, 0.05)  # 均值0，标准差0.05
            simulated = np.clip(baseline + noise, 0, 1)

            records.append({
                "movie_id": movie_id,
                "day": d + 1,
                "time_slot": t,
                "weekday_flag": w,
                "baseline_attendance": round(baseline, 3),
                "simulated_attendance": round(simulated, 3)
            })

# === Step 5: 保存完整数据 ===
df_simulated = pd.DataFrame(records)
df_simulated.to_csv(r"C:\Users\47797\Desktop\sample_simulated_table.csv", index=False)

print("模拟数据已保存到 C:\\Users\\47797\\Desktop\\sample_simulated_table.csv")
