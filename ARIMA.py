import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from math import sqrt
from time import time
from scipy.stats import shapiro, anderson
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.arima_model import ARMA, ARIMA  # 模型
from statsmodels.tsa.stattools import adfuller, kpss  # ADF检验
from statsmodels.stats.diagnostic import acorr_ljungbox  # 白噪声检验
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # 画图定阶
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# <editor-fold desc="step1：原始数据导入">
PATH = 'F:\\PycharmProjects\\DATA\\'
features = pd.read_csv(PATH + 'data0.csv', index_col=['Time'])
features.head()

print("The selected parameters are:Dis(m),P(dBm),TL(dB),Ph(deg),FSPL(dB),MTA(sec)")
df = features.loc[:, ["P(dBm)"]]
df_abs = abs(features.loc[:, ["P(dBm)"]])
df_log = np.log10(df_abs)
df_log = pd.DataFrame(df_log)  # 转换为DataFrame

plt.figure(1)
plt.plot(df, label="df")
plt.legend()
plt.show()

plt.figure(2)
plt.plot(df_log, label="df_log")
plt.legend()
plt.show()

# np.savetxt('df.csv', df, fmt='%f', delimiter=',')  # 保存一阶差分训练数据
# np.savetxt('df_log.csv', df_log, fmt='%f', delimiter=',')  # 保存一阶差分测试数据

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
df_adf_test = Stationarity_ADF_Test(df_log)  # 若p值＞0.05，则可以拒绝原假设，认为该序列为平稳

df_kpss_test = Stationarity_KPSS_Test(df_log)  # 若p值<0.05，则可以拒绝原假设，认为该序列为平稳

df_lb_test = Ljung_Box_Test(df_log)  # 若p值<0.05，则可以拒绝原假设，认为该序列不是白噪声序列


"""时间序列差分转换为平稳时间序列"""
'差分'
df_log_diff1 = df_log.diff(1)
df_log_diff1 = df_log_diff1.dropna()  # 一阶差分
# np.savetxt('df_log_diff1.csv', df_log_diff1, fmt='%f', delimiter=',')  # 保存一阶差分训练数据

'平稳检测'
df_log_diff1_adf_test = Stationarity_ADF_Test(df_log_diff1)
print(df_log_diff1_adf_test)

'绘制ACF、PACF初步确定ARIMA模型q、p参数'
f = plt.figure(facecolor='white')
ax1 = f.add_subplot(211)
plt.title("ACF")
plot_acf(df_log_diff1, lags=31, ax=ax1)
ax2 = f.add_subplot(212)
plt.title("PACF")
plot_pacf(df_log_diff1, lags=31, ax=ax2)
plt.show()

'对差分得到的平稳时间序列进行划分，训练集70%、测试集30%'
train_df = df_log_diff1[0:int(len(df_log_diff1) * 0.7)]  # 训练集70%
val_df = df_log_diff1[int(len(df_log_diff1) * 0.7):]  # 验证集30%
# </editor-fold>
# np.savetxt('train_df.csv', train_df, fmt='%f', delimiter=',')  # 保存一阶差分训练数据
# np.savetxt('val_df.csv', val_df, fmt='%f', delimiter=',')  # 保存一阶差分测试数据

# <editor-fold desc="step3：ARIMA建模">
"""模型定阶"""
# order = st.arma_order_select_ic(train_df,max_ar=19,max_ma=19,ic=['aic','bic'])
# order.bic_min_order

"""构建预测模型p=10,d=1,q=11"""
model = ARMA(train_df, order=(10, 11))  # 直接对差分后的数据进行拟合，只需用ARMA模型
train_t0 = time()
arma_model = model.fit(disp=-1, method='css')
train_t1 = time()
train_time = train_t1-train_t0
qqplot(arma_model.resid, line='q', fit=True)
plt.show()
# anderson (arma_model.resid, dist ='norm' )

stat, p = shapiro(arma_model.resid)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably Gaussian')
else:
    print('Probably not Gaussian')
arma_model.summary2()


"样本内预测"
pre_t0 = time()
in_sample_diff_pred = arma_model.predict()  # 样本内预测
pre_t1 = time()
pre_time = pre_t1-pre_t0

"数据还原：一阶差分、对数"
in_sample_diff_pred_df = pd.DataFrame(in_sample_diff_pred)
in_sample_diff_train_df_shift = df_log.loc[1:int(len(df_log_diff1) * 0.7)].shift(1)

in_sample_diff_pred_df_recover = in_sample_diff_pred_df.add(in_sample_diff_train_df_shift['P(dBm)'],  axis=0)  # 差分还原
in_sample_pred_recover = -1 * pow(10, in_sample_diff_pred_df_recover)  # 对数还原


in_sample_result_index = range(1, int(len(df) * 0.7)+1, 1)
in_sample_df = df.loc[1:int(len(df_log_diff1) * 0.7)+1]
np.savetxt('arima in sample predict.csv',
           np.column_stack((in_sample_result_index, in_sample_df, in_sample_pred_recover)),
           fmt='%f',
           delimiter=',')  # 保存样本内预测结果

"绘图"
in_sample_df = df.loc[1:int(len(df_log_diff1) * 0.7)]

plt.figure(3)
plt.title("in sample pred")
plt.plot(in_sample_df, label="real_value")
plt.plot(in_sample_pred_recover, label="predict_value")
plt.legend()
plt.show()

"分析残差ACF和PACF"
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(arma_model.resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(arma_model.resid, lags=40, ax=ax2)
plt.show()
# </editor-fold>


# <editor-fold desc="step4：预测结果与指标分析">
"样本外预测"
predict_len = len(df[int(len(df) * 0.7)-1:])  # 预测长度
arima_mse = np.zeros([predict_len+1, 1], dtype=float)  # 均方误差
arima_rmse = np.zeros([predict_len+1, 1], dtype=float)  # 均方根误差
arima_mae = np.zeros([predict_len+1, 1], dtype=float)  # 平均绝对误差
arima_mape = np.zeros([predict_len+1, 1], dtype=float)  # 平均绝对百分误差
arima_r2_score = np.zeros([predict_len+1, 1], dtype=float)  # 决定系数,越大于好，最大为1


for i in range(predict_len+1):
    print(i)
    start_point = int(len(df) * 0.7) - 1  # 预测开始时间点1185
    end_point = start_point + i  # 预测结束时间点
    out_sample_df = df.loc[start_point:end_point]
    "预测"
    out_sample_diff_pred = arma_model.predict(start=start_point,
                                              end=end_point,
                                              dynamic=True)  # 预测未来时刻
    "还原"
    out_sample_diff_pred_df = pd.DataFrame(out_sample_diff_pred)
    out_sample_diff_train_df_shift = df_log.loc[start_point - 1:end_point].shift(1).dropna()
    out_sample_diff_pred_df_recover = out_sample_diff_pred_df.add(out_sample_diff_train_df_shift['P(dBm)'],
                                                                  axis=0)  # 差分还原
    out_sample_pred_recover = -1 * pow(10, out_sample_diff_pred_df_recover)  # 对数还原

    "计算指标"
    arima_mse[i] = mean_squared_error(out_sample_df,
                                      out_sample_pred_recover)  # 均方误差

    arima_rmse[i] = sqrt(mean_squared_error(out_sample_df,
                                            out_sample_pred_recover))  # 均方根误差

    arima_mae[i] = mean_absolute_error(out_sample_df,
                                       out_sample_pred_recover)  # 平均绝对误差

    arima_mape[i] = mean_absolute_percentage_error(out_sample_df,
                                                   out_sample_pred_recover)  # 平均绝对百分误差

    arima_r2_score[i] = r2_score(out_sample_df,
                                 out_sample_pred_recover)  # 决定系数


"保存结果"
result_index = range(int(len(df) * 0.7)+1, int(len(df))+1, 1)

np.savetxt('arima predict result.csv',
           np.column_stack((result_index,
                            out_sample_df.loc[int(len(df) * 0.7)+1:],
                            out_sample_pred_recover.loc[int(len(df) * 0.7)+1:])),
           fmt='%f',
           delimiter=',')  # 保存样本内预测结果

lag = int(len(df) * 0.7)+1 - (int(len(df) * 0.7) - 1)
np.savetxt('arima predict performance.csv',
           np.column_stack((result_index,
                            arima_mse[lag:],
                            arima_rmse[lag:],
                            arima_mae[lag:],
                            arima_mape[lag:],
                            arima_r2_score[lag:])),
           fmt='%f',
           delimiter=',')  # 保存样本内预测结果
"绘图"
plt.figure(5)
plt.title("arima_mse")
plt.plot(arima_mse[2:32], label="arima_mse")
plt.legend()
plt.show()

plt.figure(6)
plt.title("arima_rmse")
plt.plot(arima_rmse, label="arima_rmse")
plt.legend()
plt.show()

plt.figure(7)
plt.title("arima_mae")
plt.plot(arima_mse, label="arima_mae")
plt.legend()
plt.show()

plt.figure(8)
plt.title("arima_mape")
plt.plot(arima_mape, label="arima_mape")
plt.legend()
plt.show()

plt.figure(9)
plt.title("arima_r2_score")
plt.plot(arima_r2_score, label="arima_r2_score")
plt.legend()
plt.show()
# </editor-fold>

# <editor-fold desc="step4：qita ">
# out_sample_diff_fore = arma_model.forecast(steps=100)
# out_sample_diff_fore_df = pd.DataFrame(out_sample_diff_fore[0])
# out_sample_diff_fore_df.index = range(1186, 1286, 1)
# out_sample_diff_train_df_shift = df_log[int(len(df_log_diff1) * 0.7)-1:int(len(df_log_diff1) * 0.7) + 100].shift(1)
# out_sample_diff_fore_df_recover = out_sample_diff_fore_df.add(out_sample_diff_train_df_shift['P(dBm)'],  axis=0)  # 差分还原
# out_sample_fore_recover = -1 * pow(10, out_sample_diff_fore_df_recover)  # 对数还原
# plt.plot(out_sample_fore_recover, label="forecast_value")
# ARIMA_fore_error = mean_squared_error(out_sample_fore_recover.dropna(),
#                                       out_sample_df)
# print('Test fore MSE: %.5f' % ARIMA_fore_error)


# '差分还原'
# valid_test = df_log[int(len(df_log_diff1) * 0.7):int(len(df_log_diff1) * 0.7) + 12]
# valid_test_shift_ts = valid_test.shift(1)
# valid_test_shift_ts = pd.Series(valid_test_shift_ts['P(dBm)'].values)
# forecast_recover = forecast_ts.add(valid_test_shift_ts)
# forecast_recover = forecast_recover.shift(-1)
# # forecast_recover = pow(10,forecast_recover)
#
# # np.savetxt('valid_test_shift_ts.csv', valid_test, fmt='%f', delimiter=',')  # 保存真实值
# # np.savetxt('forecast_recover.csv', forecast_recover, fmt='%f', delimiter=',')  # 保存预测值
#
#
# '测试集预测结果绘图'
# plt.figure()
# plt.title("df_diff1&predict_ts")
# plt.plot(valid_test['P(dBm)'].values, label="real_value")
# plt.plot(forecast_recover, label="predict_value")
# plt.legend()
# plt.show()


