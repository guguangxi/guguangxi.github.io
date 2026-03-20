import os
# 🔴 必须放在最开头：修复中文路径编码问题
os.environ["JOBLIB_TEMP_FOLDER"] = "C:/temp"  # 改用纯英文临时路径
os.environ["PYTHONUTF8"] = "1"                # 强制 UTF-8 编码
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'      # 关闭冗余日志
os.environ['CUDA_VISIBLE_DEVICES'] = ''       # 强制 CPU 训练，避免 GPU 冲突

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

print("开始训练模型...")

# ---------- 1. 生成模拟数据 ----------
np.random.seed(42)
n_samples = 5000

data = {
    'mix_ratio': np.random.uniform(0, 20, n_samples),
    'moisture': np.random.uniform(15, 55, n_samples),
    'heat_value_in': np.random.uniform(1800, 4200, n_samples),
    'volatile': np.random.uniform(20, 60, n_samples),
    'ash': np.random.uniform(15, 50, n_samples),
    'temp_furnace': np.random.uniform(800, 950, n_samples),
    'heat_value_out': np.zeros(n_samples),
    'nox_emission': np.zeros(n_samples)
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

X_train, X_test, y_train_nox, y_test_nox = train_test_split(X, y_nox, test_size=0.2, random_state=42)

# 🔴 修复中文路径：强制单线程训练
rf_model = RandomForestRegressor(
    n_estimators=100, 
    max_depth=10, 
    random_state=42,
    n_jobs=1  # 关键：关闭多进程并行
)
rf_model.fit(X_train, y_train_nox)

rf_score = rf_model.score(X_test, y_test_nox)
print(f"随机森林R²分数: {rf_score:.3f}")

joblib.dump(rf_model, 'rf_nox_model.pkl')
print("随机森林模型已保存为 rf_nox_model.pkl")

# ---------- 3. 训练LSTM模型（预测热值）✅ 稳定版 ----------
print("\n训练LSTM模型...")
sequence_length = 10
X_lstm = []
y_lstm = []

for i in range(len(df) - sequence_length):
    X_lstm.append(df[['mix_ratio', 'moisture', 'heat_value_in', 'volatile', 'ash']].iloc[i:i+sequence_length].values)
    y_lstm.append(df['heat_value_out'].iloc[i+sequence_length])

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

# 🔴 对LSTM输入做标准化
scaler_lstm = StandardScaler()
n_samples_lstm, seq_len, n_features = X_lstm.shape
X_lstm_reshaped = X_lstm.reshape(-1, n_features)
X_lstm_scaled = scaler_lstm.fit_transform(X_lstm_reshaped).reshape(n_samples_lstm, seq_len, n_features)

# 🔴 对目标变量y做标准化
scaler_y = StandardScaler()
y_lstm_scaled = scaler_y.fit_transform(y_lstm.reshape(-1, 1)).flatten()

split = int(0.8 * len(X_lstm_scaled))
X_train_lstm, X_test_lstm = X_lstm_scaled[:split], X_lstm_scaled[split:]
y_train_lstm, y_test_lstm = y_lstm_scaled[:split], y_lstm_scaled[split:]

# 🔴 构建LSTM模型（消除Keras警告）
lstm_model = Sequential([
    Input(shape=(sequence_length, n_features)),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1)
])

lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# 🔴 早停法防止过拟合
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=2
)

history = lstm_model.fit(
    X_train_lstm, y_train_lstm,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=2
)

# 🔴 关键：反标准化后计算真实指标
y_pred_lstm_scaled = lstm_model.predict(X_test_lstm, verbose=0)

# 还原到原始数值尺度
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled).flatten()
y_test_lstm_real = scaler_y.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten()

# 计算真实MAE和R²
lstm_r2 = r2_score(y_test_lstm_real, y_pred_lstm)
mae_real = np.mean(np.abs(y_pred_lstm - y_test_lstm_real))

print(f"\nLSTM测试集MAE: {mae_real:.2f}")
print(f"LSTM测试集R²: {lstm_r2:.3f}")

# 🔴 用新格式保存模型，消除警告
lstm_model.save('lstm_heat_model.keras')
joblib.dump(scaler_lstm, 'scaler_lstm.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("LSTM模型已保存为 lstm_heat_model.keras")

# ---------- 4. 保存通用Scaler ----------
scaler = StandardScaler()
scaler.fit(X)
joblib.dump(scaler, 'scaler.pkl')
print("标准化器已保存为 scaler.pkl")

print("\n✅ 所有模型训练完成！")

