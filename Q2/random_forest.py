from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# 1. 特征 & 目标
X = df[['main_genre', 'lang', 'runtime', 'cast_count', 'writers_count', 'director_known']]
y = df['rating']

# 2. 预处理：类别变量 one-hot，数值变量保留
cat_cols = ['main_genre', 'lang']
num_cols = ['runtime', 'cast_count', 'writers_count', 'director_known']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ])

# 3. 建模
model = Pipeline(steps=[
    ('prep', preprocessor),
    ('rf', RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1))
])

# 4. 训练 & 评估（快速 80/20 分割）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('R² on test set:', r2_score(y_test, y_pred))

# 5. 特征重要性可视化
importances = model.named_steps['rf'].feature_importances_
feature_names = (model.named_steps['prep']
                 .get_feature_names_out())

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# 只看前 15 个
plt.figure(figsize=(8,6))
sns.barplot(x=feat_imp.head(15), y=feat_imp.head(15).index, palette='viridis')
plt.title('Top 15 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# 表格形式
display(feat_imp.head(15))