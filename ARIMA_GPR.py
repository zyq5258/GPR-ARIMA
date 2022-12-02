from time import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tensorflow import keras
from math import sqrt
from sklearn import preprocessing
from scipy.stats import shapiro, anderson
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.arima_model import ARMA, ARIMA  # 模型
from statsmodels.tsa.stattools import adfuller, kpss  # ADF检验
from statsmodels.stats.diagnostic import acorr_ljungbox  # 白噪声检验
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# <editor-fold desc="step1：原始数据导入">
PATH = 'F:\\PycharmProjects\\DATA\\'
data = pd.read_csv(PATH + 'data0.csv', index_col=['Time'])
data.head()

print("The selected parameters are:Dis(m),P(dBm),TL(dB),Ph(deg),FSPL(dB),MTA(sec)")
df = data.loc[:, ["P(dBm)"]]
df_abs = abs(data.loc[:, ["P(dBm)"]])
df_log = np.log10(df_abs)
df_log = pd.DataFrame(df_log)  # 转换为DataFrame

# </editor-fold>


# <editor-fold desc="step2：数据预处理">
"""时间序列平稳性检测"""
'''
  ADF单位根检测：零假设为序列有单位根,是非平稳的,
  P-Value如果小于显著级别(0.05),则可以拒绝零假设.
'''


def Stationarity_ADF_Test(input_ts):
    adf_test = adfuller(input_ts)
    adf_output = pd.Series(adf_test[0:4],
                           index=['ADF Test Statistic', 'pvalue', '#Lags Used', 'Number of Observations Used'])
    for key, value in adf_test[4].items():
        adf_output['Critical Value (%s)' % key] = value
    return adf_output


'''
KPSS检测：零假设为序列是*稳的,通过指定regression='ct'参数
来让kps把“确定性趋势(deterministic trend)”的序列认为是平稳的.
'''


def Stationarity_KPSS_Test(input_ts):
    kpss_test = kpss(input_ts, regression='c')
    kpss_output = pd.Series(kpss_test[0:3],
                            index=['KPSS Test Statistic', 'pvalue', '#Lags Used'])
    for key, value in kpss_test[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    return kpss_output


"""白噪声检验(若序列为纯白噪声则没有分析的意义)"""
"""acorr_ljungbox(x, lags=None, boxpierce=False)函数检验无自相关
    lags为延迟期数，如果为整数，则是包含在内的延迟期数，如果是一个列表或数组，那么所有时滞都包含在列表中最大的时滞中
    boxpierce为True时表示除开返回LB统计量还会返回Box和Pierce的Q统计量
    返回值：
        lbvalue:测试的统计量
        pvalue:基于卡方分布的p统计量
        bpvalue:((optionsal), float or array) – test statistic for Box-Pierce test
        bppvalue:((optional), float or array) – p-value based for Box-Pierce test on chi-square distribution
"""


def Ljung_Box_Test(input_ts):
    lb_test = acorr_ljungbox(input_ts, lags=1, boxpierce=True)
    print('\nLB Test Statistic:')
    lb_output = pd.Series(lb_test[0:4],
                          index=['lbvalue', 'pvalue', 'bpvalue', 'bppvalue'])
    return lb_output


"""原始时间序列平稳性检测"""
# df_adf_test = Stationarity_ADF_Test(df_log)  # 若p值＞0.05，则可以拒绝原假设，认为该序列为平稳
# df_kpss_test = Stationarity_KPSS_Test(df_log)  # 若p值<0.05，则可以拒绝原假设，认为该序列为平稳
# df_lb_test = Ljung_Box_Test(df_log)  # 若p值<0.05，则可以拒绝原假设，认为该序列不是白噪声序列

"""时间序列差分转换为平稳时间序列"""
'差分'
df_log_diff1 = df_log.diff(1)
df_log_diff1 = df_log_diff1.dropna()  # 一阶差分
# np.savetxt('df_log_diff1.csv', df_log_diff1, fmt='%f', delimiter=',')  # 保存一阶差分训练数据

'平稳检测'
df_log_diff1_adf_test = Stationarity_ADF_Test(df_log_diff1)
print(df_log_diff1_adf_test)

'对差分得到的平稳时间序列进行划分，训练集70%、测试集30%'
train_df = df_log_diff1[0:int(len(df_log_diff1) * 0.7)]  # 训练集70%
val_df = df_log_diff1[int(len(df_log_diff1) * 0.7):]  # 验证集30%
# </editor-fold>

# <editor-fold desc="step3：ARIMA建模">
"模型定阶"
# order = st.arma_order_select_ic(train_df,max_ar=19,max_ma=19,ic=['aic','bic'])
# order.bic_min_order

"构建预测模型p=10,d=1,q=11"
model = ARMA(train_df, order=(10, 11))  # 直接对差分后的数据进行拟合，只需用ARMA模型
arma_model = model.fit(disp=-1, method='css')
# qqplot(arma_model.resid, line='q', fit=True)
# plt.show()
# anderson (arma_model.resid, dist ='norm' )

# stat, p = shapiro(arma_model.resid)
# print('stat=%.3f, p=%.3f' % (stat, p))
# if p > 0.05:
#     print('Probably Gaussian')
# else:
#     print('Probably not Gaussian')
arma_model.summary2()

"样本内预测"
in_sample_diff_pred = arma_model.predict()  # 样本内预测

"数据还原：一阶差分、对数"
in_sample_diff_pred_df = pd.DataFrame(in_sample_diff_pred)
in_sample_diff_train_df_shift = df_log.loc[1:int(len(df_log_diff1) * 0.7)].shift(1)

in_sample_diff_pred_df_recover = in_sample_diff_pred_df.add(in_sample_diff_train_df_shift['P(dBm)'], axis=0)  # 差分还原
in_sample_pred_recover = -1 * pow(10, in_sample_diff_pred_df_recover)  # 对数还原

# np.savetxt('in_sample_pred_recover.csv',
#            in_sample_pred_recover,
#            fmt='%f',
#            delimiter=',')  # 保存样本内预测结果
"绘图"
in_sample_df = df.loc[1:int(len(df_log_diff1) * 0.7)]

plt.figure(3)
plt.title("in sample pred")
plt.plot(in_sample_df, label="real_value")
plt.plot(in_sample_pred_recover, label="predict_value")
plt.legend()
plt.show()

"分析残差ACF和PACF"
# fig = plt.figure(figsize=(12, 8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(arma_model.resid.values.squeeze(), lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(arma_model.resid, lags=40, ax=ax2)
# plt.show()
# </editor-fold>


# <editor-fold desc="step4：预测结果与指标分析">
"样本外预测"
in_sample_start_point = 10  # 采样样本起点
start_point = int(len(df) * 0.7) - in_sample_start_point  # 预测开始时间点
end_point = len(df)  # 预测结束时间点
out_sample_df = df.loc[start_point:end_point]
"预测"
t1 = time()
out_sample_diff_pred = arma_model.predict(start=start_point,
                                          end=end_point,
                                          dynamic=True)  # 预测未来时刻
t2 = time()
dt1 = t2-t1
print("Spend time: %s" % dt1)

"还原"
out_sample_diff_pred_df = pd.DataFrame(out_sample_diff_pred)
out_sample_diff_train_df_shift = df_log.loc[start_point - 1:end_point].shift(1).dropna()
out_sample_diff_pred_df_recover = out_sample_diff_pred_df.add(out_sample_diff_train_df_shift['P(dBm)'],
                                                              axis=0)  # 差分还原
out_sample_pred_recover = -1 * pow(10, out_sample_diff_pred_df_recover)  # 对数还原

plt.figure(4)
plt.title("out sample pred")
plt.plot(out_sample_df, label="real_value")
plt.plot(out_sample_pred_recover, label="predict_value")
plt.legend()
plt.show()

in_sample_pred_error = in_sample_pred_recover.sub(in_sample_df['P(dBm)'], axis=0)
plt.figure(5)
plt.title("in sample pred error")
plt.plot(in_sample_pred_error, label="in sample predict error")
plt.legend()
plt.show()

out_sample_pred_error = out_sample_pred_recover.sub(out_sample_df['P(dBm)'], axis=0)
plt.figure(6)
plt.title("arima out sample pred error")
plt.plot(out_sample_pred_error, label="out sample predict error")
plt.legend()
plt.show()

np.savetxt('arima in sample predict error.csv',
           in_sample_pred_error,
           fmt='%f',
           delimiter=',')  # 保存样本内预测性能指标
np.savetxt('arima out sample predict error.csv',
           out_sample_pred_error,
           fmt='%f',
           delimiter=',')  # 保存样本内预测性能指标
# plt.figure(7)
# plt.title("pred error")
# plt.plot(in_sample_pred_error.loc[1176:1186], label="in sample predict error")
# plt.plot(out_sample_pred_error.loc[1176:1186], label="out sample predict error")
# plt.legend()
# plt.show()

error1_lb_test = Ljung_Box_Test(in_sample_pred_error.dropna())  # 若p值<0.05，则可以拒绝原假设，认为该序列不是白噪声序列
error2_lb_test = Ljung_Box_Test(out_sample_pred_error)  # 若p值<0.05，则可以拒绝原假设，认为该序列不是白噪声序列

# <editor-fold desc="step3：GPR建模残差预测">
"""step3：GPR建模残差预测"""
"构造样本和标签"
window = 20  # 时序滑窗大小
error_data = pd.concat([in_sample_pred_error.loc[1185 - window + 1:1185], out_sample_pred_error.loc[1186:]], axis=0)
# error_data = out_sample_pred_error.add(20, axis=0)

gpr_data = error_data.iloc[:-1]
gpr_targets = error_data.iloc[window:]
new_dat = gpr_data.iloc[1:1 + window]
"""
ConstantKernel(constant_value=1.0,constant_value_bounds=(1e-05, 100000.0))
GaussianProcessRegressor(kernel=None,指定GP的协方差函数的核
                         *,
                         alpha=1e-10,拟合时核矩阵对角线上增加的值
                         optimizer='fmin_l_bfgs_b',
                         n_restarts_optimizer=0,为了找到使对数边际可能性最大化的内核参数，优化器重新启动的次数。
                         normalize_y=False,是否通过去除均值并缩放到单位方差来规范化目标值y。
                         copy_X_train=True,
                         random_state=None)确定用于初始化中心的随机数生成
该常数值定义协方差: k(x1, x2)=constant_value,默认是1
常量值的下界和上界。如果设置为“fixed”，则在超参数调优过程中不能更改constant_value。
"""
# kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-3, 1e3), 0.5)  # Matern内核
kernel1 = Matern(1.0, (1e-3, 1e3), 0.5)  # Matern内核
kernel2 = RationalQuadratic(length_scale=1.0, alpha=1.5)
kernel3 = ConstantKernel(1.0, (1e-3, 1e3))
kernel = kernel3 + kernel1 + kernel2 + DotProduct() + WhiteKernel()

GPR_model = GaussianProcessRegressor(kernel=kernel,
                                     optimizer="fmin_l_bfgs_b",
                                     alpha=1e-10,
                                     n_restarts_optimizer=9
                                     )
# </editor-fold>

# <editor-fold desc="step3：GPR预测">
means = np.zeros([len(gpr_targets), 1], dtype=float)
sigmas = np.zeros([len(gpr_targets), 1], dtype=float)
pre_time = np.zeros([len(gpr_targets), 1], dtype=float)

for i in range(len(gpr_targets) - 1):
    print(i)
    train_dat = gpr_data.iloc[i:i + window]
    train_lab = gpr_targets.iloc[i]

    new_dat = gpr_data.iloc[i + 1:i + 1 + window]
    new_dat = np.reshape(np.array(new_dat), (-1, window))
    # Fit the model to the data (optimize hyper parameters)
    t3 = time()
    GPR_model.fit(np.reshape(np.array(train_dat), (-1, window)),
                  np.array(train_lab))
    # 预测
    means[i], sigmas[i] = GPR_model.predict(new_dat, return_std=True)
    t4 = time()
    pre_time[i] = t4-t3

avg_time = np.mean(pre_time)


error_pred_temp = pd.DataFrame(means)
error_pred_temp.index = range(1186, 1696, 1)
error_pred = error_pred_temp
plt.figure(7)
plt.title("gpr error predict result")
plt.plot(gpr_targets, 'b-', label="arima predict error")
plt.plot(error_pred, 'r-', label="gpr predict value")
plt.legend()
plt.show()

error_pred_index = range(1186, 1696, 1)
np.savetxt('gpr error predict result.csv',
           np.column_stack((error_pred_index, gpr_targets, error_pred)),
           fmt='%f',
           delimiter=',')  # 保存最终预测结果(ARIMA+GPR)

# </editor-fold>


# <editor-fold desc="step6:合并ARIMA预测结果与GPR误差预测结果">
out_sample_pred = out_sample_pred_recover.loc[1186:].add(error_pred.values, axis=0)

plt.figure(8)
plt.title("out sample pred")
plt.plot(out_sample_df, label="real_value")
plt.plot(out_sample_pred, label="predict_value")
plt.legend()
plt.show()

final_pred_index = range(1187, 1696, 1)
np.savetxt('arima gpr predict result.csv',
           np.column_stack((final_pred_index,
                            out_sample_df,
                            out_sample_pred.loc[1187:])),
           fmt='%f',
           delimiter=',')  # 保存最终预测结果(ARIMA+GPR)

# </editor-fold>

# <editor-fold desc="step7：性能分析，指标计算MSE、MAE、MAPE、可靠性">
out_sample_df = out_sample_df.loc[1187:]
final_mse = np.zeros([len(out_sample_df), 1], dtype=float)  # 均方误差
final_rmse = np.zeros([len(out_sample_df), 1], dtype=float)  # 均方根误差
final_mae = np.zeros([len(out_sample_df), 1], dtype=float)  # 平均绝对误差
final_mape = np.zeros([len(out_sample_df), 1], dtype=float)  # 平均绝对百分误差
final_r2_score = np.zeros([len(out_sample_df), 1], dtype=float)  # 决定系数,越大于好，最大为1

for i in range(1187, 1696, 1):
    print(i)
    final_mse[i - 1187] = mean_squared_error(out_sample_df.loc[1187:i],
                                             out_sample_pred.loc[1187:i])  # 均方误差

    final_rmse[i - 1187] = sqrt(mean_squared_error(out_sample_df.loc[1187:i],
                                                   out_sample_pred.loc[1187:i]))  # 均方根误差

    final_mae[i - 1187] = mean_absolute_error(out_sample_df.loc[1187:i],
                                              out_sample_pred.loc[1187:i])  # 平均绝对误差

    final_mape[i - 1187] = mean_absolute_percentage_error(out_sample_df.loc[1187:i],
                                                          out_sample_pred.loc[1187:i])  # 平均绝对百分误差

    final_r2_score[i - 1187] = r2_score(out_sample_df.loc[1187:i],
                                        out_sample_pred.loc[1187:i])  # 决定系数

"保存结果"
np.savetxt('arima gpr predict performance.csv',
           np.column_stack((final_pred_index,
                            final_mse,
                            final_rmse,
                            final_mae,
                            final_mape,
                            final_r2_score)),
           fmt='%f',
           delimiter=',')  # 保存样本内预测性能指标
"绘图"
plt.figure(9)
plt.title("final_mse")
plt.plot(final_mse[0:30], label="final_mse")
plt.legend()
plt.show()

plt.figure(10)
plt.title("final_rmse")
plt.plot(final_rmse[0:30], label="final_rmse")
plt.legend()
plt.show()

plt.figure(11)
plt.title("final_mae")
plt.plot(final_mae[0:30], label="final_mae")
plt.legend()
plt.show()

plt.figure(12)
plt.title("final_mape")
plt.plot(final_mape[0:30], label="final_mape")
plt.legend()
plt.show()

plt.figure(13)
plt.title("final_r2_score")
plt.plot(final_r2_score, label="final_r2_score")
plt.legend()
plt.show()

# </editor-fold>
