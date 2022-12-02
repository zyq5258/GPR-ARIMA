import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# <editor-fold desc="step1：原始数据导入">
"""step1：原始数据导入"""
path = 'F:\\PycharmProjects\\DATA\\'
features = pd.read_csv(path + 'data0.csv', index_col=['Time'])
features.head()

print("The selected parameters are:Dis(m),P(dBm),TL(dB),Ph(deg),FSPL(dB),MTA(sec)")
df = features.loc[:, ["P(dBm)"]]
df.head()

# </editor-fold>

# <editor-fold desc="step2：数据延时处理">
"""step2：数据划分：训练集、验证集、测试集，比例70%:20%:10%"""
time_lag = 1  # 时延
df_lag = df.shift(time_lag)  # 训练集70%

plt.figure(1)
plt.title("Time delay CSI")
plt.plot(df, label="The receiver samples data in real time")
plt.plot(df_lag, label="The feedback data obtained by the transmitter")
plt.legend()
plt.show()

df_valid = df_lag[int(len(df) * 0.7):]  # 用于指标计算

bl_mse = np.zeros([len(df_valid), 1], dtype=float)  # 均方误差
bl_rmse = np.zeros([len(df_valid), 1], dtype=float)  # 均方根误差
bl_mae = np.zeros([len(df_valid), 1], dtype=float)  # 平均绝对误差
bl_mape = np.zeros([len(df_valid), 1], dtype=float)  # 平均绝对百分误差
bl_r2_score = np.zeros([len(df_valid), 1], dtype=float)  # 决定系数,越大于好，最大为1


for i in range(len(df_valid)):
    print(i)
    start_point = int(len(df) * 0.7) + 1  # 预测开始时间点
    end_point = start_point + i  # 预测结束时间点
    df_real = df.loc[start_point:end_point]
    df_feedback = df.loc[start_point - 1:end_point - 1]
    # print(df_feedback)

    "计算指标"
    bl_mse[i] = mean_squared_error(df_real,
                                   df_feedback)  # 均方误差

    bl_rmse[i] = sqrt(mean_squared_error(df_real,
                                         df_feedback))  # 均方根误差

    bl_mae[i] = mean_absolute_error(df_real,
                                    df_feedback)  # 平均绝对误差

    bl_mape[i] = mean_absolute_percentage_error(df_real,
                                                df_feedback)  # 平均绝对百分误差

    bl_r2_score[i] = r2_score(df_real,
                              df_feedback)  # 决定系数

sample_index = range(1, int(len(df))+1, 1)
result_index = range(int(len(df) * 0.7)+1, int(len(df))+1, 1)
"保存结果"
np.savetxt('base line sample data and feedback data all.csv',
           np.column_stack((sample_index,
                            df,
                            df_lag)),
           fmt='%f',
           delimiter=',')  # 保存实际数据与延时反馈数据
np.savetxt('base line sample data and feedback data part.csv',
           np.column_stack((result_index,
                            df_real,
                            df_feedback)),
           fmt='%f',
           delimiter=',')  # 保存实际数据与延时反馈数据
np.savetxt('base line performance.csv',
           np.column_stack((result_index,
                            bl_mse,
                            bl_rmse,
                            bl_mae,
                            bl_mape,
                            bl_r2_score)),
           fmt='%f',
           delimiter=',')  # 保存样本内预测性能指标
"绘图"
plt.figure(2)
plt.title("bl_mse")
plt.plot(bl_mse[0:30], label="bl_mse")
plt.legend()
plt.show()

plt.figure(3)
plt.title("bl_rmse")
plt.plot(bl_rmse, label="bl_rmse")
plt.legend()
plt.show()

plt.figure(4)
plt.title("bl_mae")
plt.plot(bl_mae, label="bl_mae")
plt.legend()
plt.show()

plt.figure(5)
plt.title("bl_mape")
plt.plot(bl_mape, label="bl_mape")
plt.legend()
plt.show()

plt.figure(6)
plt.title("bl_r2_score")
plt.plot(bl_r2_score, label="bl_r2_score")
plt.legend()
plt.show()
# </editor-fold>

