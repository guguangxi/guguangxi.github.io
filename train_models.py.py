import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭多余日志
os.environ['CUDA_VISIBLE_DEVICES'] = ''   # 强制只用CPU，避免GPU冲突
# train_models.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

print("开始训练模型...")

# ---------- 1. 生成模拟数据（如果你有真实数据，替换这部分）----------
np.random.seed(42)
n_samples = 5000

# 生成特征
data = {
    'mix_ratio': np.random.uniform(0, 20, n_samples),        # 掺烧比例
    'moisture': np.random.uniform(15, 55, n_samples),         # 含水率
    'heat_value_in': np.random.uniform(1800, 4200, n_samples), # 入炉热值
    'volatile': np.random.uniform(20, 60, n_samples),         # 挥发分
    'ash': np.random.uniform(15, 50, n_samples),              # 灰分
    'temp_furnace': np.random.uniform(800, 950, n_samples),   # 炉膛温度
    'heat_value_out': np.zeros(n_samples),                     # 输出热值（待生成）
    'nox_emission': np.zeros(n_samples)                         # NOx排放（待生成）
}

df = pd.DataFrame(data)

# 生成输出值（模拟真实关系）
df['heat_value_out'] = (0.3 * df['heat_value_in'] + 
                        200 * df['mix_ratio'] - 
                        15 * df['moisture'] + 
                        5 * df['volatile'] + 
                        np.random.normal(0, 50, n_samples))

df['nox_emission'] = (120 + 
                      2.5 * df['mix_ratio'] + 
                      0.7 * (df['moisture'] - 30) - 
                      0.03 * (df['heat_value_in'] - 2800) + 
                      0.2 * df['temp_furnace'] + 
                      np.random.normal(0, 10, n_samples))

print("模拟数据生成完成")
# ---------- 模拟数据生成结束 ----------

# ---------- 2. 训练随机森林模型（预测NOx）----------
print("\n训练随机森林模型...")
feature_cols = ['mix_ratio', 'moisture', 'heat_value_in', 'volatile', 'ash', 'temp_furnace']
X = df[feature_cols]
y_nox = df['nox_emission']

# 划分训练集和测试集
X_train, X_test, y_train_nox, y_test_nox = train_test_split(X, y_nox, test_size=0.2, random_state=42)

# 训练随机森林
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train_nox)

# 评估
rf_score = rf_model.score(X_test, y_test_nox)
print(f"随机森林R²分数: {rf_score:.3f}")

# 保存模型
joblib.dump(rf_model, 'rf_nox_model.pkl')
print("随机森林模型已保存为 rf_nox_model.pkl")

# ---------- 3. 训练LSTM模型（预测热值）----------
print("\n训练LSTM模型...")
# LSTM需要时间序列格式，这里简化：用前N个时刻预测当前
# 构造序列数据
sequence_length = 10
X_lstm = []
y_lstm = []

for i in range(len(df) - sequence_length):
    X_lstm.append(df[['mix_ratio', 'moisture', 'heat_value_in', 'volatile', 'ash']].iloc[i:i+sequence_length].values)
    y_lstm.append(df['heat_value_out'].iloc[i+sequence_length])

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

# 划分
split = int(0.8 * len(X_lstm))
X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]

# 构建LSTM模型
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, 5)),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
lstm_model.summary()

# 训练
history = lstm_model.fit(
    X_train_lstm, y_train_lstm,
    epochs=20,
    batch_size=16,
    validation_split=0.1,
    verbose=2
)

# 评估
loss, mae = lstm_model.evaluate(X_test_lstm, y_test_lstm, verbose=0)
print(f"LSTM测试集MAE: {mae:.2f}")

# 保存模型
lstm_model.save('lstm_heat_model.h5')
print("LSTM模型已保存为 lstm_heat_model.h5")

# ---------- 4. 保存Scaler（用于数据标准化）----------
scaler = StandardScaler()
scaler.fit(X)
joblib.dump(scaler, 'scaler.pkl')
print("标准化器已保存为 scaler.pkl")

print("\n✅ 所有模型训练完成！")
