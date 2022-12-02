from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# <editor-fold desc="step1：原始数据导入">
"""step1：原始数据导入"""
data_path = 'F:\\PycharmProjects\\DATA\\'
result_path = "../../TestProject/Result"
features = pd.read_csv(data_path + 'data0.csv', index_col=['Time'])
features.head()

print("The selected parameters are:Dis(m),P(dBm),TL(dB),Ph(deg),FSPL(dB),MTA(sec)")
df = features.loc[:, ["P(dBm)"]]  # 信道增益
df.head()
# </editor-fold>

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
dataset = min_max_scaler.fit_transform(df)
dataset = pd.DataFrame(dataset)
dataset.index = range(1, int(len(df)) + 1, 1)

plt.figure()
plt.plot(dataset, label='sample data')
plt.legend()
plt.show()

look_back = 20  # 数据窗口大小

train_size = int(len(dataset) * 0.7)
train_data = dataset[:train_size]
test_data = dataset[train_size - look_back:]


def create_dataset(data_set, time_step):
    # 这里的look_back与timestep相同
    data_X = []
    data_Y = []
    for i in range(len(data_set) - time_step):
        print(i)
        a = data_set.iloc[i:(i + time_step)]
        data_X.append(a)
        data_Y.append(data_set.iloc[i + time_step])
    return data_X, data_Y


# 训练数据太少 look_back并不能过大

trainX, trainY = create_dataset(train_data, look_back)
trainX = np.array(trainX)
trainY = np.array(trainY)
testX, testY = create_dataset(test_data, look_back)
testX = np.array(testX)
testY = np.array(testY)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# create and fit the LSTM network
model = Sequential()
model.add(Input(shape=(trainX.shape[1], trainX.shape[-1])))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
train_t0 = time()
model.fit(trainX, trainY, epochs=1, batch_size=100, verbose=1)
train_t1 = time()
train_time = train_t1-train_t0
model.summary()

# make predictions

# model = load_model(os.path.join("DATA","Test" + ".h5"))
trainPredict = model.predict(trainX)
pre_t0 = time()
testPredict = model.predict(testX)
pre_t1 = time()
pre_time = pre_t1-pre_t0


# 反归一化
trainPredict = min_max_scaler.inverse_transform(trainPredict)
trainY = min_max_scaler.inverse_transform(trainY)
testPredict = min_max_scaler.inverse_transform(testPredict)
testY = min_max_scaler.inverse_transform(testY)

plt.figure()
plt.plot(trainY, label="train_real_value", color='red')
plt.plot(trainPredict, label="train_pre_value", color='blue')
plt.legend()
plt.show()

plt.figure()
plt.plot(testY, label="test_real_value", color='red')
plt.plot(testPredict, label="test_pre_value", color='blue')
plt.legend()
plt.show()

testPredict = pd.DataFrame(testPredict)
testPredict.index = range(int(len(df) * 0.7) + 1, int(len(df)) + 1, 1)
testY = pd.DataFrame(testY)
testY.index = range(int(len(df) * 0.7) + 1, int(len(df)) + 1, 1)

"保存结果"
result_index = range(int(len(df) * 0.7) + 1, int(len(df)) + 1, 1)

np.savetxt('rnn predict result.csv',
           np.column_stack((result_index,
                            testY,
                            testPredict)),
           fmt='%f',
           delimiter=',')  # 保存样本内预测结果

"指标计算MSE、MAE、MAPE、可靠性"
rnn_mse = np.zeros([len(testPredict), 1], dtype=float)  # 均方误差
rnn_rmse = np.zeros([len(testPredict), 1], dtype=float)  # 均方根误差
rnn_mae = np.zeros([len(testPredict), 1], dtype=float)  # 平均绝对误差
rnn_mape = np.zeros([len(testPredict), 1], dtype=float)  # 平均绝对百分误差
rnn_r2_score = np.zeros([len(testPredict), 1], dtype=float)  # 决定系数,越大于好，最大为1

for i in range(int(len(df) * 0.7) + 1, int(len(df)) + 1, 1):
    print(i)
    rnn_mse[i - 1187] = mean_squared_error(testY.loc[int(len(df) * 0.7) + 1:i],
                                           testPredict.loc[int(len(df) * 0.7) + 1:i])  # 均方误差

    rnn_rmse[i - 1187] = sqrt(mean_squared_error(testY.loc[int(len(df) * 0.7) + 1:i],
                                                 testPredict.loc[int(len(df) * 0.7) + 1:i]))  # 均方根误差

    rnn_mae[i - 1187] = mean_absolute_error(testY.loc[int(len(df) * 0.7) + 1:i],
                                            testPredict.loc[int(len(df) * 0.7) + 1:i])  # 平均绝对误差

    rnn_mape[i - 1187] = mean_absolute_percentage_error(testY.loc[int(len(df) * 0.7) + 1:i],
                                                        testPredict.loc[int(len(df) * 0.7) + 1:i])  # 平均绝对百分误差

    rnn_r2_score[i - 1187] = r2_score(testY.loc[int(len(df) * 0.7) + 1:i],
                                      testPredict.loc[int(len(df) * 0.7) + 1:i])  # 决定系数

"保存结果"
np.savetxt('rnn predict performance.csv',
           np.column_stack((result_index,
                            rnn_mse,
                            rnn_rmse,
                            rnn_mae,
                            rnn_mape,
                            rnn_r2_score)),
           fmt='%f',
           delimiter=',')  # 保存样本内预测结果

"结果可视化"
plt.figure()
plt.title("rnn_mse")
plt.plot(rnn_mse[0:30], label="rnn_mse")
plt.legend()
plt.show()

plt.figure()
plt.title("rnn_rmse")
plt.plot(rnn_rmse[0:30], label="rnn_rmse")
plt.legend()
plt.show()

plt.figure()
plt.title("rnn_mae")
plt.plot(rnn_mae[0:30], label="rnn_mae")
plt.legend()
plt.show()

plt.figure()
plt.title("rnn_mape")
plt.plot(rnn_mape[0:30], label="rnn_mape")
plt.legend()
plt.show()

plt.figure()
plt.title("rnn_r2_score")
plt.plot(rnn_r2_score[0:30], label="rnn_r2_score")
plt.legend()
plt.show()
