import rospy
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from geometry_msgs.msg import Twist
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

class LstmModel:
    def __init__(self):
        rospy.init_node('LSTM_AI_Model')
        self.target_velocity_sub = rospy.Subscriber('/target_velocity', Twist, self.target_velocity_callback)
        self.vx_data = []  # 여전히 리스트로 유지
        self.vy_data = []
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # 데이터 정규화를 위한 스케일러
        self.plot_done = False  # 플롯이 한번만 실행되게 제어하는 플래그
        self.rate = rospy.Rate(10)  # 적절한 빈도로 설정
        while not rospy.is_shutdown():
            self.rate.sleep()

    def target_velocity_callback(self, data):
        target_vx = data.linear.x
        target_vy = data.linear.y
        self.vx_data.append(target_vx)
        self.vy_data.append(target_vy)
        print(f"Data length: {len(self.vx_data)}")
        if len(self.vx_data) >= 100:
            if self.plot_done == False:
                self.make_lstm()
            self.run_lstm()
            self.plot_done = True  # 플롯이 실행되었음을 표시

    def make_lstm(self):
        print("Running LSTM model...")
        
        vx_data_array = np.array(self.vx_data).reshape(-1, 1)
        self.vx_data_scaled = self.scaler.fit_transform(vx_data_array)

        # LSTM 입력 데이터 준비 (시퀀스 길이 10)
        X, y = [], []
        for i in range(10, len(self.vx_data_scaled)):
            X.append(self.vx_data_scaled[i-10:i, 0])
            y.append(self.vx_data_scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # LSTM 모델 생성
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        self.model.add(LSTM(units=50))
        self.model.add(Dense(1))
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        # 모델 학습
        self.model.fit(X, y, epochs=20, batch_size=32, verbose=1)

    def run_lstm(self):
        # LSTM 입력 데이터 준비 (마지막 10개의 데이터 사용)
        inputs = np.array(self.vx_data[-10:]).reshape(-1, 1)
        inputs = self.scaler.transform(inputs)
        inputs = inputs.reshape((1, inputs.shape[0], 1))

        # 예측
        self.forecast = []
        for _ in range(20):  # 20 스텝 예측
            predicted_value = self.model.predict(inputs)
            predicted_value = np.reshape(predicted_value, (1, 1, 1))  # 차원 맞추기
            self.forecast.append(predicted_value[0, 0, 0])
            inputs = np.append(inputs[:, 1:, :], predicted_value, axis=1)

        self.forecast = self.scaler.inverse_transform(np.array(self.forecast).reshape(-1, 1)).flatten()
        print("Forecast: ", self.forecast)

        # 실제 값과 예측 값을 비교하여 정확성 평가
        if len(self.vx_data) >= 110:  # 최소 110개의 데이터가 필요함 (예측 데이터 + 비교할 실제 데이터)
            actual = np.array(self.vx_data[-10:])  # 예측 시점 이후의 실제 데이터
            self.evaluate_forecast(actual, self.forecast)

    def evaluate_forecast(self, actual, forecast):
        mae = mean_absolute_error(actual, forecast)
        mse = mean_squared_error(actual, forecast)
