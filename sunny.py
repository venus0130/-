import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from itertools import combinations
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, GaussianNoise

# 讀取數據
data = pd.read_csv('re_predictors15.csv')

# 選取前七個重要特徵
selected_features = ['VAR169', 'VAR178', 'VAR165', 'VAR166', 'SEASON', 'VAR164', 'HOUR']
X = data[selected_features]
y = data['POWER']

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 使用均值填充缺失值
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# 特徵標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 初始化變量來儲存最佳結果
best_mse = float('inf')
best_combo = None
best_model = None

# 遍歷所有可能的特徵組合，從2個到7個特徵
for r in range(2, 8):
    for combo in combinations(selected_features, r):
        # 根據特徵組合提取數據
        X_train_subset = X_train[:, [selected_features.index(feature) for feature in combo]]
        X_test_subset = X_test[:, [selected_features.index(feature) for feature in combo]]
        
        # 使用Lasso回歸進行特徵選擇和訓練
        model = Lasso(alpha=0.01)
        model.fit(X_train_subset, y_train)
        
        # 預測並計算MSE
        y_pred = model.predict(X_test_subset)
        mse = mean_squared_error(y_test, y_pred)
        
        # 如果當前組合的模型表現最好，則更新最佳結果
        if mse < best_mse:
            best_mse = mse
            best_combo = combo
            best_model = model

print(f"Best combination: {best_combo}, Best MSE: {best_mse}")

# 選擇最優特徵組合
selected_features_final = np.array(best_combo)[best_model.coef_ != 0]
print("Selected features in final model:", selected_features_final)

# 使用選出的最佳特徵組合來構建RBF-DNN模型
def build_rbf_dnn_model(input_dim):
    model = Sequential()
    model.add(GaussianNoise(0.1, input_shape=(input_dim,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

# 使用最佳特徵組合來訓練RBF-DNN模型
X_train_final = X_train[:, [selected_features.index(feature) for feature in selected_features_final]]
X_test_final = X_test[:, [selected_features.index(feature) for feature in selected_features_final]]

rbf_dnn_model = build_rbf_dnn_model(X_train_final.shape[1])
rbf_dnn_model.fit(X_train_final, y_train, epochs=100, validation_split=0.2, verbose=1)

# 評估最終RBF-DNN模型
final_y_pred = rbf_dnn_model.predict(X_test_final)
final_mse = mean_squared_error(y_test, final_y_pred)
print(f"Final RBF-DNN Model MSE: {final_mse}", selected_features_final )
