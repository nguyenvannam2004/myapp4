import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Bước 1: Đọc dữ liệu bệnh tim từ file CSV 
data = pd.read_csv('./mynewdata.csv')

# Giả sử các đặc trưng (feature) của bạn là các cột từ 0 đến -1 và cột đích là 'target'
X = data.drop(columns=['target', 'STT']).values  # Các đặc trưng đầu vào
y = data['target'].values  # Nhãn mục tiêu

# Chuẩn hóa dữ liệu
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Chia dữ liệu: 80% cho huấn luyện, 20% cho xác thực và kiểm tra
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Chia 20% còn lại thành 50% cho xác thực và 50% cho kiểm tra
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Sử dụng SMOTE để cân bằng dữ liệu
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Khởi tạo trọng số ngẫu nhiên
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)  
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X_train, y_train, epochs, learning_rate, lambda_reg):
        for epoch in range(epochs):
            # Giai đoạn lan truyền tiến
            hidden_layer_input = np.dot(X_train, self.weights_input_hidden)
            hidden_layer_output = self.sigmoid(hidden_layer_input)
            
            final_input = np.dot(hidden_layer_output, self.weights_hidden_output)
            final_output = self.sigmoid(final_input)

            # Tính toán lỗi
            error = y_train.reshape(-1, 1) - final_output

            # Lan truyền ngược
            d_final_output = error * self.sigmoid_derivative(final_output)
            error_hidden_layer = d_final_output.dot(self.weights_hidden_output.T)
            d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(hidden_layer_output)

            # Cập nhật trọng số với regularization
            self.weights_hidden_output += hidden_layer_output.T.dot(d_final_output) * learning_rate - lambda_reg * self.weights_hidden_output
            self.weights_input_hidden += X_train.T.dot(d_hidden_layer) * learning_rate - lambda_reg * self.weights_input_hidden

    def predict(self, X_new):
        # Chuẩn hóa dữ liệu mới sử dụng các giá trị từ tập huấn luyện
        X_new = (X_new - X_mean) / X_std

        # Giai đoạn lan truyền tiến
        hidden_layer_input = np.dot(X_new, self.weights_input_hidden)
        hidden_layer_output = self.sigmoid(hidden_layer_input)

        final_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        final_output = self.sigmoid(final_input)

        # Chuyển đổi đầu ra thành nhãn dự đoán
        y_pred = (final_output > 0.5).astype(int)
        return y_pred

# Khởi tạo mô hình
input_size = X_train.shape[1]
hidden_size = 3
output_size = 1
model = NeuralNetwork(input_size, hidden_size, output_size)

# Huấn luyện mô hình
model.train(X_train_resampled, y_train_resampled, epochs=2000, learning_rate=0.01, lambda_reg=0.02)

# Lưu mô hình vào file
joblib.dump(model, "neuralnetwork_model.pkl")
