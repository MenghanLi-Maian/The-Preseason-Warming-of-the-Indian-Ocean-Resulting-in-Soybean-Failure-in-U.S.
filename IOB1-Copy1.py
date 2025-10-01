#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import font_manager
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt 
import cartopy.feature as cfeat 
from cartopy.io.shapereader import Reader
import geopandas as gpd
import matplotlib.colors as mcolors
import xarray as xr
import warnings
from scipy import signal
from scipy.signal import detrend
from cartopy.util import add_cyclic_point
from eofs.standard import Eof
import numpy.ma as ma
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')


# In[2]:


plt.rcParams['font.size']=17
mpl.rcParams['font.family'] = 'Times New Roman'


# In[3]:


#产量数据
#读取单产和总产
yie_soy=pd.read_excel('/Users/limenghan/Desktop/大豆产量/delnan-aggerate-wrd-soy.xlsx',index_col=0, header=0)
pro_soy=pd.read_excel('/Users/limenghan/Desktop/大豆产量/delnan-pro-aggerate-wrd-soy.xlsx',index_col=0, header=0)
pro_soy=pro_soy.set_index(yie_soy.index)
##求收获面积
area_soy=pro_soy/yie_soy
##北美
north_america = ['Alabama', 'Arkansas',  'Delaware', 'Florida','Georgia', 'Illinois',
                 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maryland', 'Michigan',
                 'Minnesota', 'Mississippi','Missouri', 'Nebraska', 'New Jersey', 
                 'North Carolina','North Dakota', 'Ohio', 'Oklahoma','Pennsylvania',
                 'South Carolina', 'South Dakota', 'Tennessee','Texas', 'Virginia', 'Wisconsin',
                'West Virginia','New York']
north_america_pro=['Alabama', 'Arkansas',  'Delaware', 'Georgia', 'Illinois',
                 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maryland', 'Michigan',
                 'Minnesota', 'Mississippi','Missouri', 'Nebraska', 'New Jersey', 
                 'North Carolina','North Dakota', 'Ohio', 'Oklahoma','Pennsylvania',
                 'South Carolina', 'South Dakota', 'Tennessee','Texas', 'Virginia', 'Wisconsin',
                ]
##五年平均滑动法算趋势产量
def year_5_mean(y_series):
    ## 对一个序列进行五年平均滑动法
    year=y_series.index.values
    mean_series=[]
    for i in range(2,len(year)-2):
        mean_series.append((y_series.iloc[i-2]+y_series.iloc[i-1]+y_series.iloc[i]+y_series.iloc[i+1]+y_series.iloc[i+2])/5)
    return mean_series
def yield_5year_all(yield_X):
    ## 对一整个dataframe（索引是年份，然后是地区）进行五年平均滑动法
    yield_5year=[]
    year=yield_X.columns.values
    Area=yield_X.index
    for i in range(0,len(Area)):
        y_series=pd.Series(yield_X.iloc[i,:])
        yield_5year.append(year_5_mean(y_series))
    yield_5year=pd.DataFrame(data=yield_5year,index=Area,columns=np.arange(year[2],year[len(year)-2]))
    return yield_5year
def anom(Y,Y_trend):
    ##产量距平百分比公式
    return Y-Y_trend
def pro_anom(pro,area):
    pro=pro.loc[area]
    pro=pro.sum(axis=0)
    pro_=pro[2:-2]
    pro_exp=pd.Series(data=year_5_mean(pro),index=pro_.index)
    return pro_-pro_exp
##平均单产距平
def yie_anom(yie,yie_area,area):
    yie=yie.loc[area]
    yie_area=yie_area.loc[area]
    yie=yie.mul(yie_area).sum(axis=0)/(yie_area.sum(axis=0))
    yie_=yie[2:-2]
    yie_exp=pd.Series(data=year_5_mean(yie),index=yie_.index)
    return (yie_-yie_exp)/yie_exp
pro_anom_north_america=pro_anom(pro_soy.dropna(axis=0),north_america_pro)
yie_anom_per_north_america=yie_anom(yie_soy,area_soy,north_america)
def pro_anom_(pro,area):
    pro=pro.loc[area]
    pro=pro.sum(axis=0)
    pro_=pro[2:-2]
    pro_exp=pd.Series(data=year_5_mean(pro),index=pro_.index)
    return (pro_-pro_exp)/pro_exp
pro_anom_north_america_=pro_anom_(pro_soy.dropna(axis=0),north_america_pro)
pro_anom_north_america_
##平均单产距平
def yie_anomalies(yie,yie_area,area):
    yie=yie.loc[area]
    yie_area=yie_area.loc[area]
    yie=yie.mul(yie_area).sum(axis=0)/(yie_area.sum(axis=0))
    yie_=yie[2:-2]
    yie_exp=pd.Series(data=year_5_mean(yie),index=yie_.index)
    return (yie_-yie_exp)
yie_anom_=yie_anomalies(yie_soy,area_soy,north_america)
pro_anom_=yie_anom_*area_soy.loc[north_america].dropna(axis=0).sum(axis=0)
def pro_exp(pro,area):
    pro=pro.loc[area]
    pro=pro.sum(axis=0)
    pro_=pro[2:-2]
    pro_exp=pd.Series(data=year_5_mean(pro),index=pro_.index)
    return pro_exp
exp_soy_pro=pro_exp(pro_soy.dropna(axis=0),north_america_pro)


# In[4]:


##读取产量距平
yield_data=pd.read_excel('/Users/limenghan/Desktop/大豆产量/anomalies_soy_percent.xlsx',index_col=0, header=0)
##求气象单产
##五年平均滑动法算趋势产量
def year_5_mean(y_series):
    ## 对一个序列进行五年平均滑动法
    year=y_series.index.values
    mean_series=[]
    for i in range(2,len(year)-2):
        mean_series.append((y_series.iloc[i-2]+y_series.iloc[i-1]+y_series.iloc[i]+y_series.iloc[i+1]+y_series.iloc[i+2])/5)
    return mean_series
def yield_5year_all(yield_X):
    ## 对一整个dataframe（索引是年份，然后是地区）进行五年平均滑动法
    yield_5year=[]
    year=yield_X.columns.values
    Area=yield_X.index
    for i in range(0,len(Area)):
        y_series=pd.Series(yield_X.iloc[i,:])
        yield_5year.append(year_5_mean(y_series))
    yield_5year=pd.DataFrame(data=yield_5year,index=Area,columns=np.arange(year[2],year[len(year)-2]))
    return yield_5year
Area=yie_soy.index
##趋势单产
yield_5year_all=yield_5year_all(yie_soy)
def anom(Y,Y_trend):
    ##产量距平百分比公式
    return Y-Y_trend
new_yield_soy=pd.DataFrame(data=yie_soy,index=Area,columns=np.arange(1980,2018))##更改数据格式
##气象单产
anom_soy=anom(new_yield_soy,yield_5year_all)


# In[5]:


##读取气候指数
climate_data=pd.read_excel('/Users/limenghan/Desktop/气候数据/climate_series.xlsx',index_col=0, header=0)
# 将 'time' 列转换为日期时间格式
climate_data.index = pd.to_datetime(climate_data.index, format='%Y_%m')
# 使用切片选择时间范围
climate_data = climate_data['1976-01-01':'2017-12-31']
print(len(climate_data))
##去趋势
climate_data_detrend= signal.detrend(climate_data, axis=0, type='linear', bp=0)
climate_data_detrend= pd.DataFrame(climate_data_detrend, columns=climate_data.columns, index=climate_data.index)
for i in range(0,len(climate_data_detrend.columns)):
    for month in range(1,13):
        mon_series=[]
        for year in range(0,42):
            mon_series.append(climate_data_detrend.iloc[:,i][year*12+(month-1)])
        mon_detrend=signal.detrend(mon_series)
        for year in range(0,42):
            climate_data_detrend.iloc[:,i][year*12+(month-1)]=mon_detrend[year]
def month_3_mean(sea_surface_temperature):
    # 对整个时间序列进行三个月滑动平均
    rolling_mean = sea_surface_temperature.rolling(window=3).mean()
    return rolling_mean
climate_data_detrend=month_3_mean(climate_data_detrend)
for i in range(0,len(climate_data_detrend)-1):
    climate_data_detrend.iloc[i,:]=climate_data_detrend.iloc[i+1,:]
climate_data_detrend=climate_data_detrend[1:]
##去enso信号
##施密特
def orth(A,b):
    temp = np.dot(np.transpose(A),b)/np.dot(np.transpose(A),A)
    b_=  np.dot(temp,A)
    b_new = b-b_
    return b_new
##应用
index_nino=climate_data_detrend.columns.values
index_nino=np.delete(index_nino,[2,7,9,12,10])
index_nino
for i in index_nino:
    climate_data_detrend.loc[:,i]=orth(climate_data_detrend.loc[:,"nino34"],climate_data_detrend.loc[:,i])
    locals()[f'{i}']=climate_data_detrend.loc[:,f"{i}"]


# In[6]:


def month_name2(month):
    if month==1:
        return "DJF"
    elif month==2:
        return "JFM"
    elif month==3:
        return "FMA"
    elif month==4:
        return "MAM"
    elif month==5:
        return "AMJ"
    elif month==6:
        return "MJJ"
    elif month==7:
        return "JJA"
    elif month==8:
        return "JAS"
    elif month==9:
        return "ASO"
    elif month==10:
        return "SON"
    elif month==11:
        return "OND"
    elif month==12:
        return "NDJ"
months=[
 'ND(-1)J',
 'D(-1)JF',
 'JFM(0)',
 'FMA(0)',
 'MAM(0)',
 'AMJ(0)',
 'MJJ(0)',
 'JJA(0)',
 'JAS(0)']


# In[7]:


#读取海温指数计算IOB
#全球海温数据
sst=xr.open_dataset('/Users/limenghan/Desktop/气候数据/HadISST_sst.nc')
lon_name = 'longitude'  # whatever name is in the data
# Adjust lon values to make sure they are within (-180, 180)
sst['_longitude_adjusted'] = xr.where(
    sst[lon_name] > 180,
    sst[lon_name] - 360,
    sst[lon_name])
# reassign the new coords to as the main lon coords
# and sort DataArray using new coordinate values
sst = (
    sst
    .swap_dims({lon_name: '_longitude_adjusted'})
    .sel(**{'_longitude_adjusted': sorted(sst._longitude_adjusted)})
    .drop(lon_name))
sst = sst.rename({'_longitude_adjusted': lon_name})
##时间切片
sst= sst.sel(time=slice('1979-01-01', '2017-12-31'))
##lon和lat
lon = np.array(sst['longitude'])
lat = np.array(sst['latitude'])
sst= sst.rename({'longitude': 'lon'})
sst= sst.rename({'latitude': 'lat'})
sst=sst['sst']
# 选择热带印度洋区域的数据
tropical_io = sst.sel(lat=slice(20,-20), lon=slice(40, 110))
# 计算 IOB 指数，即在经度和纬度维度上进行平均
tropical_io_weight= tropical_io*np.sqrt(np.cos(tropical_io['lat'] * np.pi / 180))
IOB =tropical_io_weight.mean(dim=['lat', 'lon'])
# 选择 Nino3.4 区域的数据
nino34_region = sst.sel(lat=slice(5, -5), lon=slice(-170, -120))
nino34_region_weight= nino34_region*np.sqrt(np.cos(nino34_region['lat'] * np.pi / 180))
nino34=nino34_region_weight.mean(dim=['lat', 'lon'])
IOB=IOB.to_dataframe(name='IOB')
nino34=nino34.to_dataframe(name='nino34')
time=IOB.index
# 按月份重采样并计算每个月的总和
IOB_monthly= IOB.resample('M').sum()
# 提取年份和月份作为新的列
IOB_monthly['Year'] = IOB_monthly.index.year
IOB_monthly['Month'] = IOB_monthly.index.month
# 使用 pivot_table 将月份作为列，年份作为索引
IOB_pivot = IOB_monthly.pivot_table(index='Year', columns='Month', values=IOB_monthly.columns[:-2])
# 按月份重采样并计算每个月的总和
nino34_monthly= nino34.resample('M').sum()
# 提取年份和月份作为新的列
nino34_monthly['Year'] = nino34_monthly.index.year
nino34_monthly['Month'] = nino34_monthly.index.month
# 使用 pivot_table 将月份作为列，年份作为索引
nino34_pivot = nino34_monthly.pivot_table(index='Year', columns='Month', values=nino34_monthly.columns[:-2])
from sklearn.linear_model import LinearRegression
# 定义一个函数来去趋势
def detrend(column):
    # 创建一个模型
    model = LinearRegression()
    # 获取时间索引的整数值
    X = column.index.astype(int).values.reshape(-1, 1)
    # 获取列的值
    y = column.values.reshape(-1, 1)
    # 拟合模型
    model.fit(X, y)
    # 预测趋势
    trend = model.predict(X)
    # 去除趋势
    detrended = y - trend
    return detrended.flatten()

# 对每一列进行去趋势
IOB= IOB_pivot.apply(detrend, axis=0)
nino34= nino34_pivot.apply(detrend, axis=0)
# 将去趋势后的 DataFrame 还原回初始的索引是时间的形式
IOB_= IOB.stack().reset_index()

# 将列索引名设置为 'Time' 和原始的列名
IOB_.columns = ['Time', 'month', 'IOB']

# 将时间戳重新设为索引
IOB_.set_index('Time', inplace=True)
IOB=IOB_.drop(columns=['month'])
IOB.index=time
# 将去趋势后的 DataFrame 还原回初始的索引是时间的形式
nino34_= nino34.stack().reset_index()

# 将列索引名设置为 'Time' 和原始的列名
nino34_.columns = ['Time','month', 'nino34']

# 将时间戳重新设为索引
nino34_.set_index('Time', inplace=True)
nino34=nino34_.drop(columns=['month'])
nino34.index=time
IOB= signal.detrend(IOB, axis=0, type='linear', bp=0)
nino34= signal.detrend(nino34, axis=0, type='linear', bp=0)
##去enso信号
##施密特
def orth(A,b):
    temp = np.dot(np.transpose(A),b)/np.dot(np.transpose(A),A)
    b_=  np.dot(temp,A)
    b_new = b-b_
    return b_new
IOB=orth(nino34[:,0],IOB[:,0])
IOB= pd.DataFrame(IOB,index=time)
IOB_NDJ_1=IOB[(IOB.index.month == 11) | (IOB.index.month == 12) | (IOB.index.month == 1)]
IOB_NDJ_1=IOB_NDJ_1[1:-2]
IOB_DJF_1=IOB[(IOB.index.month == 12) | (IOB.index.month == 1) | (IOB.index.month == 2)]
IOB_DJF_1=IOB_DJF_1[2:-1]
IOB_JFM_0=IOB[(IOB.index.month == 1) | (IOB.index.month == 2) | (IOB.index.month == 3)]
IOB_JFM_0=IOB_JFM_0[3:]
IOB_FMA_0=IOB[(IOB.index.month == 2) | (IOB.index.month ==3) | (IOB.index.month == 4)]
IOB_FMA_0=IOB_FMA_0[3:]
IOB_MAM_0=IOB[(IOB.index.month == 3) | (IOB.index.month ==4) | (IOB.index.month == 5)]
IOB_MAM_0=IOB_MAM_0[3:]
IOB_AMJ_0=IOB[(IOB.index.month == 4) | (IOB.index.month ==5) | (IOB.index.month == 6)]
IOB_AMJ_0=IOB_AMJ_0[3:]
IOB_MJJ_0=IOB[(IOB.index.month == 5) | (IOB.index.month ==6) | (IOB.index.month == 7)]
IOB_MJJ_0=IOB_MJJ_0[3:]
IOB_JJA_0=IOB[(IOB.index.month == 6) | (IOB.index.month ==7) | (IOB.index.month == 8)]
IOB_JJA_0=IOB_JJA_0[3:]
IOB_JAS_0=IOB[(IOB.index.month == 7) | (IOB.index.month ==8) | (IOB.index.month == 9)]
IOB_JAS_0=IOB_JAS_0[3:]
index_name_list=['IOB_NDJ_1','IOB_DJF_1','IOB_JFM_0','IOB_FMA_0','IOB_MAM_0','IOB_AMJ_0','IOB_MJJ_0',
                 'IOB_JJA_0','IOB_JAS_0']
name_list=['NDJ(-1)','DJF(-1)','JFM(0)','FMA(0)','MAM(0)','AMJ(0)','MJJ(0)','JJA(0)','JAS(0)']
for i in range(0,len(index_name_list)):
    ##三月平均（NDJ）
    locals()[f'{index_name_list[i]}_reset'] =locals()[f'{index_name_list[i]}'].reset_index(drop=True)
    # 每三个一组，计算平均值
    locals()[f'{index_name_list[i]}_averages'] = locals()[f'{index_name_list[i]}_reset'] .groupby(locals()[f'{index_name_list[i]}_reset'] .index // 3).mean()
    ##标准化
    locals()[f'{index_name_list[i]}']= (locals()[f'{index_name_list[i]}_averages']-locals()[f'{index_name_list[i]}_averages'].mean()) /locals()[f'{index_name_list[i]}_averages'].std()


# In[8]:


##相关
def sel_index(index_,month_mid):
    time_series=[]
    for i in range(0,len(index_.index)):
        if index_.index[i].month==month_mid:
            time_series.append(index_[i])
    time_series=pd.Series(time_series,index=np.arange(1980,2018))
    return time_series
def corr_pro_index(pro_soy,index,mon_mid,year):
    if year==1:
        cli_index=index.loc['1979-01-01':'2016-12-31']
    elif year==0:
        cli_index=index.loc['1980-01-01':'2017-12-31']
    #标准化sst data
    cli_index=(cli_index- cli_index.mean()) / cli_index.std()
    # 合并两个 Series 为一个 DataFrame
    merged_df= pd.concat([pro_soy, sel_index(cli_index,mon_mid)], axis=1, keys=['yie', 'cli'])
    # 删除包含缺失值的行
    merged_df= merged_df.dropna()
    # 计算相关系数
    spr,pv= stats.pearsonr(merged_df['cli'],merged_df['yie'])
    return spr,pv
def corr_pro_index_year3(pro_soy,index):
    corr=[]
    pv=[]
    for year in [1,0]:
        if year ==1:
            for month in range(12,13):
                spr,p=corr_pro_index(pro_soy,index,month,year)
                corr.append(spr)
                pv.append(p)
        elif year ==0:
            for month in range(1,9):
                spr,p=corr_pro_index(pro_soy,index,month,year)
                corr.append(spr)
                pv.append(p)
    # 创建DataFrame
    result= pd.DataFrame({'Corr': corr, 'Pvalue': pv},index=months)
    return result.T
nino4=climate_data_detrend.loc[:,"nino4"]
nino34=climate_data_detrend.loc[:,"nino34"]
nino3=climate_data_detrend.loc[:,"nino3"]
nino12=climate_data_detrend.loc[:,"nino1+2"]
index_list=[AMM, TSA, nino4, AO, IOD, TNA, PDO, nino12,NPGO, nino34, NAO, nino3, NTA,EA, 
       EP_NP, PNA, EA_WR, SCA, POL, SAMI_AAOI, AMO, CAR,
       QBO, SOI, Trenberth_and_Hurrell_North_Pacific_Index,WHWP,
       NAST]

index_name2=['AMM', 'TSA', 'Niño4', 'AO', 'IOD', 'TNA', 'PDO', 'Niño1+2',
       'NPGO', 'Niño3.4',  'NAO', 'Niño3', 'NTA', 'EA', 'EP/NP', 'PNA', 
        'EA/WR', 'SCA', 'POL', 'SAMI', 'AMO', 'CAR',
       'QBO', 'SOI', 'NPI', 'WHWP','NAST']
index_name3=['AMM', 'TSA', 'Niño4', 'AO', 'IOD', 'TNA', 'PDO', 'Niño1+2',
       'NPGO', 'Niño3.4',  'NAO', 'Niño3', 'NTA', 'EA', 'EP/NP', 'PNA', 
        'EA/WR', 'SCA', 'POL', 'SAMI', 'AMO', 'CAR',
       'QBO', 'SOI', 'NPI', 'WHWP','NAST','IOB']


# In[9]:


for i in range(0,len(index_list)):
    locals()[f'north_america_yie_{index_name2[i]}']=corr_pro_index_year3(yie_anom_per_north_america,index_list[i])


# In[10]:


def corr(index,index_name,yie):
    yie.index=range(0,38)
    # 合并两个 Series 为一个 DataFrame
    merged_df = pd.concat([yie, index], axis=1, keys=['yie', index_name])
    # 删除包含缺失值的行
    merged_df= merged_df.dropna()
    # 计算回归系数
    corr,p_value = stats.pearsonr(merged_df[index_name].iloc[:,0].values,merged_df['yie'].iloc[:,0].values)
    #对每个地点加权
    return corr,p_value
for index in index_name_list:
    locals()[f'corr_{index}'],locals()[f'pvalue_{index}']=corr(locals()[f'{index}'],index,yie_anom_per_north_america)
# 创建一个字典，键为你想要的列名，值为 DataFrame
dfs = {'IOB_NDJ_1':corr_IOB_NDJ_1,
       'IOB_DJF_1':corr_IOB_DJF_1,
       'IOB_JFM_0':corr_IOB_JFM_0,
       'IOB_FMA_0':corr_IOB_FMA_0,
       'IOB_MAM_0':corr_IOB_MAM_0,
       'IOB_AMJ_0':corr_IOB_AMJ_0,
       'IOB_MJJ_0':corr_IOB_MJJ_0,
       'IOB_JJA_0':corr_IOB_JJA_0,
       'IOB_JAS_0':corr_IOB_JAS_0}
df_p = {'IOB_NDJ_1':pvalue_IOB_NDJ_1,
       'IOB_DJF_1':pvalue_IOB_DJF_1,
       'IOB_JFM_0':pvalue_IOB_JFM_0,
       'IOB_FMA_0':pvalue_IOB_FMA_0,
       'IOB_MAM_0':pvalue_IOB_MAM_0,
       'IOB_AMJ_0':pvalue_IOB_AMJ_0,
       'IOB_MJJ_0':pvalue_IOB_MJJ_0,
       'IOB_JJA_0':pvalue_IOB_JJA_0,
       'IOB_JAS_0':pvalue_IOB_JAS_0}
# 创建 DataFrame，分别插入 corr 和 pvalue
data = {
    'Corr': list(dfs.values()),  # corr 数据
    'pv': list(df_p.values())    # pvalue 数据
}

# 转换为 DataFrame，并使用 months 作为列
north_america_yie_IOB = pd.DataFrame(data,index=months)
north_america_yie_IOB=north_america_yie_IOB.T


# In[11]:


# 绘制柱状图
fig, ax = plt.subplots(figsize=(12,5))
colors = color_list = [
    '#1b9e77',  # [0] Teal green
    '#d95f02',  # [1] Orange
    '#7570b3',  # [2] Purple
    '#66c2a5',  # [3]
    '#e7298a',  # [4] Pink
    '#e6ab02',  # [5] Mustard yellow
    '#a6761d',  # [6] Brown
    '#666666',  # [7]
    '#8da0cb',  # [8]
    '#fc8d62',  # [9] Coral orange
    '#a6d854',  # [10] Lime green
    '#ffd92f',  # [11] Bright yellow
    '#e5c494',  # [12]
    '#b3b3b3',  # [13]
    '#66a61e',  # [14]
    '#a6cee3',  # [15] Sky blue
    '#fb9a99',  # [16]
    '#cab2d6',  # [17]
    '#ffffb3',  # [18]
    '#fdbf6f',  # [19]
    '#b2df8a',  # [20] Pale green
    '#fdae61',  # [21]
    '#bc80bd',  # [22]
    '#80b1d3',  # [23] Blue
    '#bebada',  # [24]
    '#fb8072',  # [25]
    '#ccebc5',   # [26]
    '#17becf'  # 亮青蓝，色盲友好、与粉红强对比
]

for i in [11,2,9,15,6,23,20,0,10,1,5,4,27]:
    plt.scatter(locals()[f'north_america_yie_{index_name3[i]}'].columns,locals()[f'north_america_yie_{index_name3[i]}'].loc['Corr'],s=90,c=colors[i],label=index_name3[i])
# 添加水平线
plt.axhline(y=0.312, color='gray', linestyle='--')
plt.text(8.5, 0.3,'95%', color='black') 
plt.axhline(y=-0.312, color='gray', linestyle='--')
plt.text(8.5, -0.32,'95%', color='black') 
plt.axhline(y=-0.403, color='gray', linestyle='--')
plt.text(8.5, -0.41,'99%', color='black') 
plt.axhline(y=0.403, color='gray', linestyle='--')
plt.text(8.5, 0.39,'99%', color='black') 
plt.ylabel('Corr')

# 设置图例，使其位于图表右侧，稍微远离图的右边界
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.05))

# 保存图片并显示
plt.savefig('/Users/limenghan/Desktop/大豆代码结果/论文图/Figure1.png', bbox_inches='tight',dpi=300)
plt.show()


# In[12]:


yield_data.columns=range(0,38)
yield_data.loc[north_america]


# In[13]:


yield_data=yield_data/100


# In[14]:


north_america_yie_IOB


# In[ ]:





# In[15]:


yield_data


# In[ ]:





# # 土壤湿度和最高温度与大豆产量岭回归

# In[16]:


smroot=xr.open_dataset('/Volumes/limenghan/气象要素/era5_volumetric_soil_water_layer_2.nc')
tmx=xr.open_dataset('/Volumes/limenghan/气象要素/cru_ts4.07.1901.2022.tmx.dat.nc')
tmx=tmx['tmx']


# In[17]:


tmp=xr.open_dataset('/Volumes/limenghan/气象要素/cru_ts4.07.1901.2022.tmp.dat.nc')
tmp=tmp['tmp']


# In[18]:


smroot


# In[19]:


smroot= smroot.rename({'longitude': 'lon'})
smroot= smroot.rename({'latitude': 'lat'})
smroot= smroot.rename({'valid_time': 'time'})

# 使用 coarsen 进行降采样
if len(smroot.lon)>180:
    # 定义经度和纬度的降采样因子
    lon_factor, lat_factor = int(len(smroot.lat)/360), int(len(smroot.lon)/720)  # 对应的降采样因子
    # 使用 coarsen 方法进行降采样
    smroot= smroot.coarsen(lon=lon_factor, lat=lat_factor, boundary="trim").mean()
smroot


# In[20]:


smroot=smroot['swvl2']


# In[21]:


##定义预处理函数
##时间切片
##时间切片
def time_sel(data,yearbegin,yearend):
    data= data.sel(time=slice(f'{yearbegin}-01-01', f'{yearend}-12-31'))
    return data
def detrend_weather(data):
    # 去除季节趋势
    lon=data['lon']
    lat=data['lat']
    time=data['time']
    data=data.values.reshape((12, int(len(data)/12), len(lat), len(lon)), order='F').transpose((1,0,2,3))
    data_season = np.mean(data, axis=0)
    data_diff = data - data_season
    data_diff = data_diff.transpose((1,0,2,3)).reshape((len(time),len(lat), len(lon)), order='F')
    data_diff = np.ma.masked_array(data_diff, mask=np.isnan(data))
    data_diff=xr.DataArray(data_diff, coords=[time,lat,lon], dims=['time', 'lat','lon'])
    ##去长期趋势
    data_detrend = signal.detrend(data_diff.fillna(0), axis=0, type='linear', bp=0)
    data_detrend=xr.DataArray(data_detrend, coords=[time,lat,lon], dims=['time', 'lat','lon'])
    return data_detrend


# In[22]:


tmx=time_sel(tmx,1978,2017)
smroot=time_sel(smroot,1978,2017)
tmx=detrend_weather(tmx)
smroot=detrend_weather(smroot)
tmx=time_sel(tmx,1979,2017)
smroot=time_sel(smroot,1979,2017)


# In[23]:


tmp=time_sel(tmp,1978,2017)
tmp=detrend_weather(tmp)
tmp=time_sel(tmp,1979,2017)


# In[24]:


pre=xr.open_dataset('/Volumes/limenghan/气象要素/cru_ts4.04.1901.2019.pre.dat.nc')
pre=pre['pre']
pre=time_sel(pre,1978,2017)
pre=detrend_weather(pre)
pre=time_sel(pre,1979,2017)


# In[25]:


#定义一个时间函数
def month_name(month):
    if month==1:
        return "DJF"
    elif month==2:
        return "JFM"
    elif month==3:
        return "FMA"
    elif month==4:
        return "MAM"
    elif month==5:
        return "AMJ"
    elif month==6:
        return "MJJ"
    elif month==7:
        return "JJA"
    elif month==8:
        return "JAS"
    elif month==9:
        return "ASO"
    elif month==10:
        return "SON"
    elif month==11:
        return "OND"
    elif month==12:
        return "NDJ"


# In[26]:


##转换lon 0-360变-180-180
def trans_lon(ds):
    lon_name = 'lon'  #你的nc文件中经度的命名
    ds['longitude_adjusted'] = xr.where(
        ds[lon_name] > 180,
        ds[lon_name] - 360,
        ds[lon_name])
    ds = (
        ds
        .swap_dims({lon_name: 'longitude_adjusted'})
        .sel(**{'longitude_adjusted': sorted(ds.longitude_adjusted)})
        .drop(lon_name))
    ds = ds.rename({'longitude_adjusted': lon_name})
    return ds


# In[27]:


if tmx.lon.max()>180:
    tmx=trans_lon(tmx)
if smroot.lon.max()>180:
    smroot=trans_lon(smroot)
smroot_ame=smroot.sel(lon=slice(-130, -65), lat=slice(55,20))
tmx_ame=tmx.sel(lon=slice(-130, -65),lat=slice(20, 55))


# In[28]:


if tmp.lon.max()>180:
    tmp=trans_lon(tmp)
tmp_ame=tmp.sel(lon=slice(-130, -65),lat=slice(20, 55))


# In[29]:


if pre.lon.max()>180:
    pre=trans_lon(pre)
pre_ame=pre.sel(lon=slice(-130, -65),lat=slice(20, 55))


# In[30]:


def capitalize_words(s):
    if isinstance(s, str):
        return ' '.join(word.capitalize() for word in s.split())
    else:
        return s

## 北美
USA = gpd.read_file("/Users/limenghan/Desktop/地图/次国家/gadm41_USA_shp/gadm41_USA_1.shp")
USA.rename(columns={'NAME_1': 'loc'}, inplace=True)
gdf_list_north = [USA]
gdf_list_north = [gdf.to_crs('EPSG:4326') for gdf in gdf_list_north]
filtered_gdf_list_north = [gdf[['loc', 'geometry']][gdf['loc'].isin(north_america)] for gdf in gdf_list_north]
result_gdf_north = pd.concat(filtered_gdf_list_north, ignore_index=True)


# In[31]:


from shapely.geometry import Point


# In[32]:


def nc_shp(geo_area, ds):
    array_all=[]
    # 获取几何形状
    for area in geo_area['loc']:
        array=[]
        filtered_gdf = geo_area[geo_area['loc'] == area]
        # 检查筛选后的DataFrame是否为空
        if not filtered_gdf.empty:
            shape = filtered_gdf['geometry'].iloc[0]
        if not shape.is_valid:
            # 可以尝试修复无效的几何形状或将其删除
            shape = shape.buffer(0) 
        # 为每个点检查它是否在形状内
        for i, lat_val in enumerate(ds['lat']):
            for j, lon_val in enumerate(ds['lon']):
                if shape.contains(Point(lon_val, lat_val)):
                    array.append(ds[:,i,j])
        # 检查 array 是否为空
        if array:
            array = xr.concat(array, dim='array').mean(dim='array')
            array = array.to_dataframe(area)[[area]]
            array_all.append(array)
    array_all=pd.concat(array_all, axis=1)
    return array_all


# In[33]:


result_gdf_north_tmx=nc_shp(result_gdf_north,tmx_ame)


# In[34]:


result_gdf_north_smroot=nc_shp(result_gdf_north,smroot_ame)


# In[35]:


##dtr
dtr=xr.open_dataset('/Volumes/limenghan/气象要素/cru_ts4.07.1901.2022.dtr.dat.nc')
dtr=dtr['dtr']
dtr=time_sel(dtr,1978,2017)
dtr=detrend_weather(dtr)
dtr=time_sel(dtr,1979,2017)
if dtr.lon.max()>180:
    dtr=trans_lon(dtr)
dtr_ame=dtr.sel(lon=slice(-130, -65),lat=slice(20, 55))


# In[36]:


result_gdf_north_tmp=nc_shp(result_gdf_north,tmp_ame)


# In[37]:


def avg_sept_data(df):
    # 将index转换为datetime类型
    df.index = pd.to_datetime(df.index)
    
    # 筛选7、8、9月的数据
    df_sept = df[(df.index.month >= 7) & (df.index.month <= 9)]
    
    # 按年份分组并计算每年7、8、9月的平均值
    df_avg = df_sept.groupby(df_sept.index.year).mean()
    
    return df_avg


# In[38]:


result_tmx=avg_sept_data(result_gdf_north_tmx)
result_tmx=result_tmx.iloc[1:]
result_smroot=avg_sept_data(result_gdf_north_smroot)
result_smroot=result_smroot.iloc[1:]


# In[39]:


result_tmp=avg_sept_data(result_gdf_north_tmp)
result_tmp=result_tmp.iloc[1:]


# In[40]:


result_gdf_north_pre=nc_shp(result_gdf_north,pre_ame)
result_pre=avg_sept_data(result_gdf_north_pre)
result_pre=result_pre.iloc[1:]


# In[41]:


result_gdf_north_dtr=nc_shp(result_gdf_north,dtr_ame)
result_dtr=avg_sept_data(result_gdf_north_dtr)
result_dtr=result_dtr.iloc[1:]


# In[42]:


def standardize_data(df):
    # 对每个列（地区）进行标准化
    df_standardized = df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    return df_standardized
result_smroot_standardized=standardize_data(result_smroot)
result_tmx_standardized=standardize_data(result_tmx)


# In[43]:


result_tmp_standardized=standardize_data(result_tmp)


# In[44]:


result_pre_standardized=standardize_data(result_pre)


# In[45]:


result_dtr_standardized=standardize_data(result_dtr)


# In[46]:


result_dtr_standardized


# In[47]:


yield_data


# In[48]:


yield_data=yield_data.loc[north_america]
yield_data=yield_data.T


# In[49]:


import statsmodels.api as sm
import pandas as pd

def location_regression(yield_data, result_tmx_standardized, result_smroot_standardized, result_pre_standardized):
    """
    对每个地点分别进行一元线性回归（y 对 tmx、y 对 smroot 和 y 对 pre），
    在遇到空值时去掉对应年份的数据，并返回回归系数、截距和 p 值。

    参数:
    - yield_data: DataFrame, 产量数据，index 为年份，columns 为地点
    - result_tmx_standardized: DataFrame, 标准化后的 tmx 数据，index 为年份，columns 为地点
    - result_smroot_standardized: DataFrame, 标准化后的 smroot 数据，index 为年份，columns 为地点
    - result_pre_standardized: DataFrame, 标准化后的 pre 数据，index 为年份，columns 为地点

    返回:
    - DataFrame: 包含每个地点的 tmx、smroot 和 pre 回归结果，columns 为 tmx 系数、截距、p 值，
                 smroot 系数、截距、p 值，pre 系数、截距和 p 值
    """
    # 初始化存放结果的 DataFrame
    results_df = pd.DataFrame(columns=['tmx_coef', 'tmx_intercept', 'tmx_pvalue', 
                                       'smroot_coef', 'smroot_intercept', 'smroot_pvalue',
                                       'pre_coef', 'pre_intercept', 'pre_pvalue'])

    # 遍历每个地点
    for location in yield_data.columns:
        # 获取当前地点的 y、tmx、smroot 和 pre 数据
        y = yield_data[location]
        x_tmx = result_tmx_standardized[location]
        x_smroot = result_smroot_standardized[location]
        x_pre = result_pre_standardized[location]
        
        # 将 y, tmx, smroot, pre 合并为一个 DataFrame，方便同时去除空值
        data = pd.DataFrame({
            'y': y,
            'tmx': x_tmx,
            'smroot': x_smroot,
            'pre': x_pre
        }).dropna()  # 去除包含空值的年份
        
        # 如果去掉空值后，数据点不足以进行回归，跳过该地点
        if len(data) < 2:
            print(f"地点 {location} 数据不足，跳过回归。")
            continue
        
        # 一元线性回归：y 对 tmx
        x_tmx = sm.add_constant(data['tmx'])  # 添加截距项
        model_tmx = sm.OLS(data['y'], x_tmx).fit()
        
        # 一元线性回归：y 对 smroot
        x_smroot = sm.add_constant(data['smroot'])  # 添加截距项
        model_smroot = sm.OLS(data['y'], x_smroot).fit()
        
        # 一元线性回归：y 对 pre
        x_pre = sm.add_constant(data['pre'])  # 添加截距项
        model_pre = sm.OLS(data['y'], x_pre).fit()
        
        # 将结果存入 DataFrame
        results_df.loc[location] = [
            model_tmx.params[1],  # tmx 系数
            model_tmx.params[0],  # tmx 截距
            model_tmx.pvalues[1],  # tmx p 值
            model_smroot.params[1],  # smroot 系数
            model_smroot.params[0],  # smroot 截距
            model_smroot.pvalues[1],  # smroot p 值
            model_pre.params[1],  # pre 系数
            model_pre.params[0],  # pre 截距
            model_pre.pvalues[1]  # pre p 值
        ]
    
    return results_df


# In[50]:


import statsmodels.api as sm
import pandas as pd

def location_regression(yield_data, result_tmx_standardized, result_smroot_standardized, result_dtr_standardized):
    """
    对每个地点分别进行一元线性回归（y 对 tmx、y 对 smroot 和 y 对 dtr），
    在遇到空值时去掉对应年份的数据，并返回回归系数、截距和 p 值。

    参数:
    - yield_data: DataFrame, 产量数据，index 为年份，columns 为地点
    - result_tmx_standardized: DataFrame, 标准化后的 tmx 数据，index 为年份，columns 为地点
    - result_smroot_standardized: DataFrame, 标准化后的 smroot 数据，index 为年份，columns 为地点
    - result_dtr_standardized: DataFrame, 标准化后的 dtr 数据，index 为年份，columns 为地点

    返回:
    - DataFrame: 包含每个地点的 tmx、smroot 和 dtr 回归结果，columns 为 tmx 系数、截距、p 值，
                 smroot 系数、截距、p 值，dtr 系数、截距和 p 值
    """
    # 初始化存放结果的 DataFrame
    results_df = pd.DataFrame(columns=['tmx_coef', 'tmx_intercept', 'tmx_pvalue', 
                                       'smroot_coef', 'smroot_intercept', 'smroot_pvalue',
                                       'dtr_coef', 'dtr_intercept', 'dtr_pvalue'])

    # 遍历每个地点
    for location in yield_data.columns:
        # 获取当前地点的 y、tmx、smroot 和 dtr 数据
        y = yield_data[location]
        x_tmx = result_tmx_standardized[location]
        x_smroot = result_smroot_standardized[location]
        x_dtr = result_dtr_standardized[location]
        
        # 将 y, tmx, smroot, dtr 合并为一个 DataFrame，方便同时去除空值
        data = pd.DataFrame({
            'y': y,
            'tmx': x_tmx,
            'smroot': x_smroot,
            'dtr': x_dtr
        }).dropna()  # 去除包含空值的年份
        
        # 如果去掉空值后，数据点不足以进行回归，跳过该地点
        if len(data) < 2:
            print(f"地点 {location} 数据不足，跳过回归。")
            continue
        
        # 一元线性回归：y 对 tmx
        x_tmx = sm.add_constant(data['tmx'])  # 添加截距项
        model_tmx = sm.OLS(data['y'], x_tmx).fit()
        
        # 一元线性回归：y 对 smroot
        x_smroot = sm.add_constant(data['smroot'])  # 添加截距项
        model_smroot = sm.OLS(data['y'], x_smroot).fit()
        
        # 一元线性回归：y 对 dtr
        x_dtr = sm.add_constant(data['dtr'])  # 添加截距项
        model_dtr = sm.OLS(data['y'], x_dtr).fit()
        
        # 将结果存入 DataFrame
        results_df.loc[location] = [
            model_tmx.params[1],  # tmx 系数
            model_tmx.params[0],  # tmx 截距
            model_tmx.pvalues[1],  # tmx p 值
            model_smroot.params[1],  # smroot 系数
            model_smroot.params[0],  # smroot 截距
            model_smroot.pvalues[1],  # smroot p 值
            model_dtr.params[1],  # dtr 系数
            model_dtr.params[0],  # dtr 截距
            model_dtr.pvalues[1]  # dtr p 值
        ]
    
    return results_df


# In[51]:


yield_data.index = range(1980,2018)  


# In[52]:


yield_data


# In[53]:


result = location_regression(yield_data*100, result_tmx_standardized, result_smroot_standardized,result_dtr_standardized)
result


# In[54]:


result.index.name = 'Area'


# In[55]:


##相关画图(yield数据是xr)
def plot_yield_index(yie_xr,type_):
    # 定义一个函数，将每个单词的首字母大写
    def capitalize_words(s):
        # 检查元素是否是字符串，如果是，就进行首字母大写操作，如果不是，返回原始值
        if isinstance(s, str):
            return ' '.join(word.capitalize() for word in s.split())
        else:
            return s
    fig, ax = plt.subplots(figsize=(18, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    proj=ccrs.PlateCarree()
    # 添加大陆和国家边界线
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=2)
    ax.add_feature(cfeature.COASTLINE, linestyle='-', edgecolor='darkgray')
    ax.add_feature(cfeature.OCEAN)
    ## 绘制产量分布图
    norm1 = mcolors.TwoSlopeNorm(vmin=-8, vmax=8, vcenter=0)

    ##美国
    USA=gpd.GeoDataFrame.from_file(f"/Users/limenghan/Desktop/地图/次国家/gadm41_USA_shp/gadm41_USA_1.shp")
    USA.rename(columns={'NAME_1': 'Area'}, inplace=True)
    data_USA= pd.merge(USA,yie_xr,on='Area') ## 连接
    USA.plot(ax=ax,color='grey',edgecolor='darkgrey')
    data_map=[data_USA]
    ##拼接
    target_crs = 'EPSG:4326'
    # 转换或设置坐标参考系统
    for gdf in data_map:
        gdf['geometry'] = gdf['geometry'].to_crs(target_crs)
    data_merge = pd.concat(data_map, axis=0)
    ##画图
    if type_=='Tmx':
        data_merge.plot(ax=ax,column='tmx_coef',legend=True,linewidth=0.5,cmap='PuOr',edgecolor="grey",norm=norm1)
        data_merge[(data_merge['tmx_pvalue']<= 0.05)].to_crs(proj).plot(ax=ax,facecolor = 'none',
                                                           edgecolor='grey',hatch='...',
                                                           linewidth=0.1,
                                                           alpha=0.75)
    elif type_=='SMroot':
        data_merge.plot(ax=ax,column='smroot_coef',legend=True,linewidth=0.5,cmap='PuOr',edgecolor="grey",norm=norm1)
        data_merge[(data_merge['smroot_pvalue']<= 0.05)].to_crs(proj).plot(ax=ax,facecolor = 'none',
                                                           edgecolor='grey',hatch='...',
                                                           linewidth=0.1,
                                                           alpha=0.75)
    elif type_=='pre':
        data_merge.plot(ax=ax,column='pre_coef',legend=True,linewidth=0.5,cmap='PuOr',edgecolor="grey",norm=norm1)
        data_merge[(data_merge['pre_pvalue']<= 0.05)].to_crs(proj).plot(ax=ax,facecolor = 'none',
                                                           edgecolor='grey',hatch='...',
                                                           linewidth=0.1,
                                                           alpha=0.75)
    elif type_=='dtr':
        data_merge.plot(ax=ax,column='dtr_coef',legend=True,linewidth=0.5,cmap='PuOr',edgecolor="grey",norm=norm1)
        data_merge[(data_merge['dtr_pvalue']<= 0.05)].to_crs(proj).plot(ax=ax,facecolor = 'none',
                                                           edgecolor='grey',hatch='...',
                                                           linewidth=0.1,
                                                           alpha=0.75)
    if type_=='SMroot':
        index='c'
    elif type_=='Tmx':
        index='a'
    else:
        index='b'
    ax.set_title(f'({index})  {type_}',loc='left')
    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])
    plt.show()


# In[56]:


result.index.name = 'Area'


# In[57]:


result


# In[ ]:


plt.rcParams['font.size']=18


# In[61]:


##相关画图(yield数据是xr)
def plot_yield_index(yie_xr, type_):
    # 定义一个函数，将每个单词的首字母大写
    def capitalize_words(s):
        # 检查元素是否是字符串，如果是，就进行首字母大写操作，如果不是，返回原始值
        if isinstance(s, str):
            return ' '.join(word.capitalize() for word in s.split())
        else:
            return s
    
    fig, ax = plt.subplots(figsize=(18, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    proj = ccrs.PlateCarree()
    # 添加大陆和国家边界线
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=2)
    ax.add_feature(cfeature.COASTLINE, linestyle='-', edgecolor='darkgray')
    ax.add_feature(cfeature.OCEAN)
    
    ## 绘制产量分布图
    norm1 = mcolors.TwoSlopeNorm(vmin=-8, vmax=8, vcenter=0)

    ## 美国
    USA = gpd.GeoDataFrame.from_file(f"/Users/limenghan/Desktop/地图/次国家/gadm41_USA_shp/gadm41_USA_1.shp")
    USA.rename(columns={'NAME_1': 'Area'}, inplace=True)
    data_USA = pd.merge(USA, yie_xr, on='Area')  ## 连接
    USA.plot(ax=ax, color='grey', edgecolor='darkgrey')
    data_map = [data_USA]
    
    ## 拼接
    target_crs = 'EPSG:4326'
    # 转换或设置坐标参考系统
    for gdf in data_map:
        gdf['geometry'] = gdf['geometry'].to_crs(target_crs)
    data_merge = pd.concat(data_map, axis=0)
    
    ## 画图
    if type_ == 'Tmx':
        plot = data_merge.plot(ax=ax, column='tmx_coef', legend=True, linewidth=0.5, 
                              cmap='PuOr', edgecolor="grey", norm=norm1)
        data_merge[(data_merge['tmx_pvalue'] <= 0.05)].to_crs(proj).plot(
            ax=ax, facecolor='none', edgecolor='grey', hatch='...', linewidth=0.1, alpha=0.75)
        
    elif type_ == 'SMroot':
        plot = data_merge.plot(ax=ax, column='smroot_coef', legend=True, linewidth=0.5, 
                              cmap='PuOr', edgecolor="grey", norm=norm1)
        data_merge[(data_merge['smroot_pvalue'] <= 0.05)].to_crs(proj).plot(
            ax=ax, facecolor='none', edgecolor='grey', hatch='...', linewidth=0.1, alpha=0.75)
        
    elif type_ == 'pre':
        plot = data_merge.plot(ax=ax, column='pre_coef', legend=True, linewidth=0.5, 
                              cmap='PuOr', edgecolor="grey", norm=norm1)
        data_merge[(data_merge['pre_pvalue'] <= 0.05)].to_crs(proj).plot(
            ax=ax, facecolor='none', edgecolor='grey', hatch='...', linewidth=0.1, alpha=0.75)
        
    elif type_ == 'dtr':
        plot = data_merge.plot(ax=ax, column='dtr_coef', legend=True, linewidth=0.5, 
                              cmap='PuOr', edgecolor="grey", norm=norm1)
        data_merge[(data_merge['dtr_pvalue'] <= 0.05)].to_crs(proj).plot(
            ax=ax, facecolor='none', edgecolor='grey', hatch='...', linewidth=0.1, alpha=0.75)
    
    # 设置颜色条标签为百分比每个标准差
    cbar = plt.gcf().axes[-1]  # 获取颜色条
    cbar.set_ylabel('% per σ', fontsize=12)  # 设置颜色条标签
    
    if type_ == 'SMroot':
        index = 'c'
    elif type_ == 'Tmx':
        index = 'a'
    else:
        index = 'b'
    
    ax.set_title(f'({index})  {type_}', loc='left')
    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])
    plt.show()


# In[63]:


# 定义绘图函数
def plot_yield_index(yie_xr, type_, ax):
    def capitalize_words(s):
        if isinstance(s, str):
            return ' '.join(word.capitalize() for word in s.split())
        else:
            return s
    
    proj = ccrs.PlateCarree()

    # 添加大陆和国家边界线
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=2)
    ax.add_feature(cfeature.COASTLINE, linestyle='-', edgecolor='darkgray')
    ax.add_feature(cfeature.OCEAN)
    
    # 产量分布图的标准化颜色映射
    norm1 = mcolors.TwoSlopeNorm(vmin=-10, vmax=10, vcenter=0)

    # 加载美国地图数据
    USA = gpd.GeoDataFrame.from_file(f"/Users/limenghan/Desktop/地图/次国家/gadm41_USA_shp/gadm41_USA_1.shp")
    USA.rename(columns={'NAME_1': 'Area'}, inplace=True)
    
    # 将 USA 和 yie_xr 数据进行合并
    data_USA = pd.merge(USA, yie_xr, on='Area') 
    USA.plot(ax=ax, color='grey', edgecolor='darkgrey')
    
    # 进行空间数据处理
    data_map = [data_USA]
    target_crs = 'EPSG:4326'

    for gdf in data_map:
        gdf['geometry'] = gdf['geometry'].to_crs(target_crs)

    data_merge = pd.concat(data_map, axis=0)
    # 根据 type_ 绘制不同的回归图
    if type_ == 'Tmx':
        im = data_merge.plot(ax=ax, column='tmx_coef', legend=False, linewidth=0.5, cmap='PuOr', edgecolor="grey", norm=norm1)
        data_merge[data_merge['tmx_pvalue'] <= 0.1].to_crs(proj).plot(ax=ax, facecolor='none', edgecolor='grey', hatch='...', linewidth=0.1, alpha=0.75)
    elif type_=='SMroot':
        im = data_merge.plot(ax=ax, column='smroot_coef', legend=False, linewidth=0.5, cmap='PuOr', edgecolor="grey", norm=norm1)
        data_merge[data_merge['smroot_pvalue'] <= 0.1].to_crs(proj).plot(ax=ax, facecolor='none', edgecolor='grey', hatch='...', linewidth=0.1, alpha=0.75)
    elif type_=='DTR':
        im = data_merge.plot(ax=ax, column='dtr_coef', legend=False, linewidth=0.5, cmap='PuOr', edgecolor="grey", norm=norm1)
        data_merge[data_merge['dtr_pvalue'] <= 0.1].to_crs(proj).plot(ax=ax, facecolor='none', edgecolor='grey', hatch='...', linewidth=0.1, alpha=0.75)

    # 创建一个 ScalarMappable 对象用于 colorbar
    sm = plt.cm.ScalarMappable(cmap='PuOr', norm=norm1)
    sm.set_array([])  # 没有数据数组时设为空
    
    # 添加 colorbar 并设置 shrink
    fig = ax.get_figure()
    cbar = fig.colorbar(sm, ax=ax, shrink=1)
    
    # 设置颜色条标签为百分比每个标准差
    cbar.set_label('Yield Change (% per σ)', fontsize=15)
    
    # 设置图像标题
    if type_ == 'SMroot':
        index='b'
    elif type_=='Tmx':
        index='c'
    elif type_=='DTR':
        index='a'
    ax.set_title(f'({index})  {type_}', loc='left')
    
    # 设置地图范围
    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])

# 创建一个 1x3 的子图
fig, axs = plt.subplots(1, 3, figsize=(18, 3.5), subplot_kw={'projection': ccrs.PlateCarree()})

# 分别绘制 DTR、SMroot 和 Tmx 的回归图
variables = ['DTR', 'SMroot', 'Tmx']

for i, var in enumerate(variables):
    plot_yield_index(result, variables[i], axs[i])

# 调整布局并保存图片
plt.tight_layout()
plt.savefig('/Users/limenghan/Desktop/大豆代码结果/论文图/Figure3.png', dpi=300)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




