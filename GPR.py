import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from time import time
from tensorflow import keras
# from sklearn.model_selection._search import BaseSearchCV
from skopt import BayesSearchCV
from sklearn import preprocessing
from statsmodels.tsa.stattools import adfuller  # ADF检验
from statsmodels.stats.diagnostic import acorr_ljungbox  # 白噪声检验
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# <editor-fold desc="step1：原始数据导入">
"""step1：原始数据导入"""
path = 'F:\\PycharmProjects\\DATA\\'
features = pd.read_csv(path + 'data0.csv', index_col=['Time'])
features.head()

print("The selected parameters are:Dis(m),P(dBm),TL(dB),Ph(deg),FSPL(dB),MTA(sec)")
df = features.loc[:, ["P(dBm)"]]
df_div = df.div(-160, axis=0)  # -160代表接收灵敏度
df_div.head()
# </editor-fold>

# <editor-fold desc="step2：数据划分：训练集、验证集、测试集，比例70%:20%:10%">
"""step2：数据划分：训练集、验证集、测试集，比例70%:20%:10%"""
column_indices = {name: i for i, name in enumerate(features.columns)}
window = 20  # 时序滑窗大小
n = len(df_div)
# train_df = features[0:int(n * 0.7)]  # 训练集70%
# val_df = features[int(n * 0.7)-window:]  # 验证集30%

train_df = df_div[0:int(n * 0.7)]  # 训练集70%
val_df = df_div[int(n * 0.7) - window:]  # 验证集30%

# </editor-fold>

# <editor-fold desc="step3：数据归一化">
"""step3：数据归一化"""
# min_max_scaler = preprocessing.MinMaxScaler()
# train_df_minmax = min_max_scaler.fit_transform(train_df)
# val_df_minmax = min_max_scaler.fit_transform(val_df)  # fit_transform转换为array
#
# plt.figure(1)
# plt.plot(np.arange(0, len(train_df_minmax), 1),
#          train_df_minmax, 'b')
# plt.plot(np.arange(len(train_df_minmax)-window+1, len(train_df_minmax)+len(val_df_minmax)-window+1, 1),
#          val_df_minmax, 'g')
#
# plt.show()
#
# # 转换为DataFrame
# train_df_minmax = pd.DataFrame(train_df_minmax)
# val_df_minmax = pd.DataFrame(val_df_minmax)
# </editor-fold>

# <editor-fold desc="step4：数据检验">
# 时间序列查分
# train_df_diff1 = train_df_minmax.diff(1).dropna()  # DataFrame使用diff
# plt.figure(2)
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.title('训练数据集一阶差分')
# plt.plot(train_df_diff1)
# plt.show()
#
# val_df_diff1 = val_df_minmax.diff(1).dropna()  # 验证数据归一化
# plt.figure(3)
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.title('测试数据集一阶差分')
# plt.plot(val_df_diff1)
# plt.show()
#
# # ADF检验
# adf_test = adfuller(train_df_diff1.values, autolag='AIC')
# print(adf_test)
# # 纯随机性检验（白噪声检验）
# p_value = acorr_ljungbox(train_df_diff1, lags=1)
# print(p_value)
# </editor-fold>

# <editor-fold desc="step5：构造样本数据和测试数据">
"""step5：构造样本数据和测试数据"""
# batch_size_train = len(train_df_minmax)-window
# train_data = train_df_minmax.iloc[:-1]
# train_targets = train_df_minmax.iloc[window:]
#
# batch_size_val = len(val_df_minmax)-window
# val_data = val_df_minmax.iloc[:-1]
# val_targets = val_df_minmax.iloc[window:]

batch_size_train = len(train_df) - window
train_data = train_df.iloc[:-1]
train_targets = train_df.iloc[window:]

batch_size_val = len(val_df) - window
val_data = val_df.iloc[:-1]
val_targets = val_df.iloc[window:]

in_sample_train_dataset = keras.preprocessing.timeseries_dataset_from_array(
    data=train_data,
    targets=None,
    sequence_length=window,  # 窗口大小,过去的时间
    sampling_rate=1,  # 每1步采集一次batch_size=batch_size_train
    batch_size=batch_size_train)
for batch in in_sample_train_dataset.take(1):
    in_sample_train_data = batch

in_sample_targets_dataset = keras.preprocessing.timeseries_dataset_from_array(
    data=train_targets,
    targets=None,
    sequence_length=1,  # 窗口大小,过去的时间
    sampling_rate=1,  # 每1步采集一次batch_size=batch_size_train
    batch_size=batch_size_train)
for batch in in_sample_targets_dataset.take(1):
    in_sample_targets_data = batch

out_sample_train_dataset = keras.preprocessing.timeseries_dataset_from_array(
    data=val_data,
    targets=None,
    sequence_length=window,  # 窗口大小,过去的时间
    sampling_rate=1,  # 每1步采集一次batch_size=batch_size_train
    batch_size=batch_size_val)
for batch in out_sample_train_dataset.take(1):
    out_sample_train_data = batch

out_sample_targets_dataset = keras.preprocessing.timeseries_dataset_from_array(
    data=val_targets,
    targets=None,
    sequence_length=1,  # 窗口大小,过去的时间
    sampling_rate=1,  # 每1步采集一次batch_size=batch_size_train
    batch_size=batch_size_val)
for batch in out_sample_targets_dataset.take(1):
    out_sample_targets_data = batch
# </editor-fold>

# <editor-fold desc="step6：建立高斯过程回归模型">
# 建立高斯过程回归模型
# ConstantKernel(constant_value=1.0,constant_value_bounds=(1e-05, 100000.0))
# GaussianProcessRegressor(kernel=None,指定GP的协方差函数的核
#                          *,
#                          alpha=1e-10,拟合时核矩阵对角线上增加的值
#                          optimizer='fmin_l_bfgs_b',
#                          n_restarts_optimizer=0,为了找到使对数边际可能性最大化的内核参数，优化器重新启动的次数。
#                          normalize_y=False,是否通过去除均值并缩放到单位方差来规范化目标值y。
#                          copy_X_train=True,
#                          random_state=None)确定用于初始化中心的随机数生成
# 该常数值定义协方差: k(x1, x2)=constant_value,默认是1
# 常量值的下界和上界。如果设置为“fixed”，则在超参数调优过程中不能更改constant_value。
test_data = np.reshape(in_sample_train_data, (-1, window))
test_label = np.reshape(out_sample_targets_data, (-1, 1))
test_label_recover = -160 * test_label
print(type(test_label))

# kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=0.5)  # Matern内核

kernel1 = Matern(1.0, (1e-3, 1e3), 0.5)  # Matern内核
kernel2 = RationalQuadratic(length_scale=1.0, alpha=1.5)
kernel3 = ConstantKernel(1.0, (1e-3, 1e3))
kernel = kernel3 + kernel1 + kernel2 + DotProduct() + WhiteKernel()

gpr = GaussianProcessRegressor(kernel=kernel,
                               optimizer="fmin_l_bfgs_b",
                               n_restarts_optimizer=9)
train_t0 = time()  # 训练开始时间
gpr.fit(np.reshape(in_sample_train_data, (-1, window)),
        np.reshape(in_sample_targets_data, (-1, 1)))
train_t1 = time()  # 训练结束时间
train_time = train_t1-train_t0   # 训练总时间

pre_t0 = time()
gpr_pred, gpr_sigmas = gpr.predict(np.reshape(out_sample_train_data, (-1, window)), return_std=True)
pre_t1 = time()
pre_time = pre_t1-pre_t0

gpr_pred_recover = -160 * gpr_pred
plt.figure()
plt.title("gpr predict result")
plt.plot(gpr_pred_recover, label="gpr predict value")
plt.plot(test_label_recover, label="real value")
# plt.plot(np.reshape(out_sample_targets_data, (-1, 1)), label="real value")
plt.legend()
plt.show()
# </editor-fold>

# <editor-fold desc="step7：高斯过程回归模型预测与结果分析">
gpr_mse = np.zeros([len(test_label), 1], dtype=float)  # 均方误差
gpr_rmse = np.zeros([len(test_label), 1], dtype=float)  # 均方根误差
gpr_mae = np.zeros([len(test_label), 1], dtype=float)  # 平均绝对误差
gpr_mape = np.zeros([len(test_label), 1], dtype=float)  # 平均绝对百分误差
gpr_r2_score = np.zeros([len(test_label), 1], dtype=float)  # 决定系数,越大于好，最大为1
gpr_means = np.zeros([len(test_label), 1], dtype=float)  # 预测结果，均值
gpr_sigmas = np.zeros([len(test_label), 1], dtype=float)  # 预测结果，方差

for i in range(1, len(test_label) + 1, 1):
    print(i)
    "预测"
    gpr_means, gpr_sigmas = gpr.predict(np.reshape(out_sample_train_data[0:i], (-1, window)), return_std=True)
    gpr_means_recover = -160 * gpr_means
    "计算指标"
    gpr_mse[i - 1] = mean_squared_error(test_label_recover[0:i],
                                        gpr_means_recover[0:i])  # 均方误差

    gpr_rmse[i - 1] = sqrt(mean_squared_error(test_label_recover[0:i],
                                              gpr_means_recover[0:i]))  # 均方根误差

    gpr_mae[i - 1] = mean_absolute_error(test_label_recover[0:i],
                                         gpr_means_recover[0:i])  # 平均绝对误差

    gpr_mape[i - 1] = mean_absolute_percentage_error(test_label_recover[0:i],
                                                     gpr_means_recover[0:i])  # 平均绝对百分误差

    gpr_r2_score[i - 1] = r2_score(test_label_recover[0:i],
                                   gpr_means_recover[0:i])  # 决定系数


"保存结果"
result_index = range(int(len(df) * 0.7)+1, int(len(df))+1, 1)

np.savetxt('gpr predict result.csv',
           np.column_stack((result_index,
                            test_label_recover,
                            gpr_means_recover)),
           fmt='%f',
           delimiter=',')  # 保存样本内预测结果

np.savetxt('gpr predict performance.csv',
           np.column_stack((result_index,
                            gpr_mse,
                            gpr_rmse,
                            gpr_mae,
                            gpr_mape,
                            gpr_r2_score)),
           fmt='%f',
           delimiter=',')  # 保存样本内预测性能指标
"绘图"
plt.figure(4)
plt.title("gpr_means")
plt.plot(gpr_means, label="gpr_means")
plt.legend()
plt.show()

plt.figure(5)
plt.title("gpr_sigmas")
plt.plot(gpr_sigmas, label="gpr_sigmas")
plt.legend()
plt.show()

plt.figure(6)
plt.title("gpr_mse")
plt.plot(gpr_mse, label="gpr_mse")
plt.legend()
plt.show()

plt.figure(7)
plt.title("gpr_rmse")
plt.plot(gpr_rmse, label="gpr_rmse")
plt.legend()
plt.show()

plt.figure(8)
plt.title("gpr_mae")
plt.plot(gpr_mae, label="gpr_mae")
plt.legend()
plt.show()

plt.figure(9)
plt.title("gpr_mape")
plt.plot(gpr_mape, label="gpr_mape")
plt.legend()
plt.show()

plt.figure(10)
plt.title("gpr_r2_score")
plt.plot(gpr_r2_score, label="gpr_r2_score")
plt.legend()
plt.show()
# </editor-fold>

# <editor-fold desc="step：高斯过程回归模型预测在线训练预测">
# means = np.zeros([len(targets), 1], dtype=float)
# sigmas = np.zeros([len(targets), 1], dtype=float)
#
# for i in range(len(targets)):
#     train_dat = np.concatenate((inputs[i:, :], val_inputs[0:i, :]), axis=0)
#     train_lab = np.concatenate((targets[i:], val_targets[0:i]), axis=0)
#
#     # Fit the model to the data (optimize hyper parameters)
#     GPR_model.fit(train_dat, train_lab)
#     # 预测
#     means[i], sigmas[i] = GPR_model.predict(np.reshape(val_inputs[i], (1, -1)), return_std=True)
#     # np.shape(mean)
#     # means = np.concatenate((means, mean), axis=0)
#     # sigmas = np.concatenate((sigmas, sigma), axis=0)

# </editor-fold>
