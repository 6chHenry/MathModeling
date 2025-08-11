# train_model.py
import time
import numpy as np
import xgboost as xgb
import joblib

MODEL_FILE = "xgb_model.pkl"

if __name__ == "__main__":
    # 1. 加载预处理后的训练数据
    data = np.load("features_and_labels.npz")
    X, y = data["X"], data["y"]

    # 2. 创建并训练 XGBoost 模型
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=750,
        learning_rate=0.01,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.6,
    )

    print("开始训练模型...")
    train_start = time.time()
    model.fit(X, y)
    train_end = time.time()
    print(f"训练完成，耗时 {train_end - train_start:.2f} 秒")

    # 3. 保存模型
    joblib.dump(model, MODEL_FILE)
    print(f"模型已保存到 {MODEL_FILE}")
