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
north_america_pro=['Alabama', 'Arkansas',  'Delaware', 'Florida','Georgia', 'Illinois',
                 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maryland', 'Michigan',
                 'Minnesota', 'Mississippi','Missouri', 'Nebraska', 'New Jersey', 
                 'North Carolina','North Dakota', 'Ohio', 'Oklahoma','Pennsylvania',
                 'South Carolina', 'South Dakota', 'Tennessee','Texas', 'Virginia', 'Wisconsin',
                'West Virginia','New York']
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
pro_anom_north_america=pro_anom(pro_soy,north_america_pro)
yie_anom_per_north_america=yie_anom(yie_soy,area_soy,north_america)
def pro_anom_(pro,area):
    pro=pro.loc[area]
    pro=pro.sum(axis=0)
    pro_=pro[2:-2]
    pro_exp=pd.Series(data=year_5_mean(pro),index=pro_.index)
    return (pro_-pro_exp)/pro_exp
pro_anom_north_america_=pro_anom_(pro_soy,north_america_pro)
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
pro_anom_=yie_anom_*area_soy.loc[north_america].sum(axis=0)
def pro_exp(pro,area):
    pro=pro.loc[area]
    pro=pro.sum(axis=0)
    pro_=pro[2:-2]
    pro_exp=pd.Series(data=year_5_mean(pro),index=pro_.index)
    return pro_exp
exp_soy_pro=pro_exp(pro_soy,north_america_pro)


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
 'NDJ(-1)',
 'DJF(-1)',
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
index_name_list=['IOB_NDJ_1']
name_list=['NDJ(-1)']
for i in range(0,len(index_name_list)):
    ##三月平均（NDJ）
    locals()[f'{index_name_list[i]}_reset'] =locals()[f'{index_name_list[i]}'].reset_index(drop=True)
    # 每三个一组，计算平均值
    locals()[f'{index_name_list[i]}_averages'] = locals()[f'{index_name_list[i]}_reset'] .groupby(locals()[f'{index_name_list[i]}_reset'] .index // 3).mean()
    ##标准化
    locals()[f'{index_name_list[i]}']= (locals()[f'{index_name_list[i]}_averages']-locals()[f'{index_name_list[i]}_averages'].mean()) /locals()[f'{index_name_list[i]}_averages'].std()


# In[8]:


IOB_NDJ_1


# In[9]:


yield_data=yield_data.loc[north_america]
yield_data


# In[10]:


yield_data.columns=range(0,38)
yield_data


# In[48]:



##回归pattern
reg_slope=np.empty((len(yield_data.index),1))
reg_pv=np.empty((len(yield_data.index),1))
for loc in range(0,len(yield_data.index)):
    # 合并两个 Series 为一个 DataFrame
    merged_df = pd.concat([yield_data.iloc[loc,:], IOB_NDJ_1], axis=1, keys=['yie', 'cli'])
    # 删除包含缺失值的行
    merged_df = merged_df.dropna()
    # 计算回归系数
    slope,interpret, r_value, p_value, std_err = stats.linregress(merged_df['cli'].iloc[:,0].values,merged_df['yie'].iloc[:,0].values)
    reg_slope[loc,0]=slope
    reg_pv[loc,0]= p_value

# 创建新的 DataFrame
reg=pd.DataFrame(np.transpose([reg_slope[:,0], reg_pv[:,0]]), index=yield_data.index,columns=['reg_slope','reg_pv'])


# In[14]:



nino34= pd.DataFrame(nino34,index=time)
nino34_MJJ=nino34[(nino34.index.month == 5) | (nino34.index.month == 6) | (nino34.index.month == 7)]
nino34_MJJ=nino34_MJJ[3:]
index_name_list=['nino34_MJJ']
for i in range(0,len(index_name_list)):
    ##三月平均（NDJ）
    locals()[f'{index_name_list[i]}_reset'] =locals()[f'{index_name_list[i]}'].reset_index(drop=True)
    # 每三个一组，计算平均值
    locals()[f'{index_name_list[i]}_averages'] = locals()[f'{index_name_list[i]}_reset'] .groupby(locals()[f'{index_name_list[i]}_reset'] .index // 3).mean()
    ##标准化
    locals()[f'{index_name_list[i]}']= (locals()[f'{index_name_list[i]}_averages']-locals()[f'{index_name_list[i]}_averages'].mean()) /locals()[f'{index_name_list[i]}_averages'].std()
nino34_MJJ


# In[15]:



##回归pattern
reg_slope_nino=np.empty((len(yield_data.index),1))
reg_pv_nino=np.empty((len(yield_data.index),1))
for loc in range(0,len(yield_data.index)):
    # 合并两个 Series 为一个 DataFrame
    merged_df = pd.concat([yield_data.iloc[loc,:], nino34_MJJ], axis=1, keys=['yie', 'cli'])
    # 删除包含缺失值的行
    merged_df = merged_df.dropna()
    # 计算回归系数
    slope,interpret, r_value, p_value, std_err = stats.linregress(merged_df['cli'].iloc[:,0].values,merged_df['yie'].iloc[:,0].values)
    reg_slope_nino[loc,0]=slope
    reg_pv_nino[loc,0]= p_value

# 创建新的 DataFrame
reg_nino=pd.DataFrame(np.transpose([reg_slope_nino[:,0], reg_pv_nino[:,0]]), index=yield_data.index,columns=['reg_slope','reg_pv'])
reg_nino


# In[16]:


def plot_yield_index(yie_xr, index, type_, index_name):
    # 定义一个函数，将每个单词的首字母大写
    def capitalize_words(s):
        if isinstance(s, str):
            return ' '.join(word.capitalize() for word in s.split())
        else:
            return s

    # 创建图形和坐标轴，使用 PlateCarree 投影
    fig, ax = plt.subplots(figsize=(20, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    proj = ccrs.PlateCarree()

    # 添加国家边界、海岸线和海洋
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=2)
    ax.add_feature(cfeature.COASTLINE, linestyle='-', edgecolor='darkgray')
    ax.add_feature(cfeature.OCEAN)

    # 设置颜色映射标准
    if type_ == 'corr':
        norm1 = mcolors.TwoSlopeNorm(vmin=-0.8, vmax=0.8, vcenter=0)
    else:
        norm1 = mcolors.TwoSlopeNorm(vmin=-10, vmax=10, vcenter=0)

    # 读取美国地图数据
    USA = gpd.read_file(f"/Users/limenghan/Desktop/地图/次国家/gadm41_USA_shp/gadm41_USA_1.shp")
    USA.rename(columns={'NAME_1': 'Area'}, inplace=True)

    # 将美国地图数据与产量数据进行合并
    data_USA = pd.merge(USA, yie_xr, on='Area')

    # 转换坐标参考系统 (CRS)
    target_crs = 'EPSG:4326'
    data_USA = data_USA.to_crs(target_crs)

    # 绘制美国地图
    USA.plot(ax=ax, color='grey', edgecolor='darkgrey')

    # 根据不同类型绘制不同的数据
    if type_ == 'corr':
        data_USA.plot(ax=ax, column='corr', legend=True, linewidth=0.5, cmap='PuOr', edgecolor="grey", norm=norm1)
        data_USA[(data_USA['corr_pv'] <= 0.05)].to_crs(proj).plot(ax=ax, facecolor='none', edgecolor='grey', hatch='...', linewidth=0.1, alpha=0.75)
    else:
        data_USA.plot(ax=ax, column='reg_slope', legend=True, linewidth=0.5, cmap='PuOr', edgecolor="grey", norm=norm1)
        data_USA[(data_USA['reg_pv'] <= 0.05)].to_crs(proj).plot(ax=ax, facecolor='none', edgecolor='grey', hatch='...', linewidth=0.1, alpha=0.75)
    cbar.set_label('Yield Change (% per σ)', fontsize=15)
    # 设置图形标题和范围
    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])

    # 显示图形
    plt.show()


# In[17]:


plt.rcParams['font.size']=15


# In[18]:


plot_yield_index(reg_nino,nino34_MJJ,'reg_nino','nino34_MJJ')


# In[19]:



# 指定包含所有文件的文件夹路径
folder_path = '/Volumes/limenghan/气象要素/vertically_integrated_moisture_divergence/'
# 使用通配符 '*' 来匹配文件夹中所有的 nc 文件
files = folder_path + '*.nc'
# 使用 open_mfdataset 函数来打开所有匹配的文件，并将它们合并成一个 Dataset
vimd= xr.open_mfdataset(files, combine='nested', concat_dim='time')


# In[20]:


vimd


# In[21]:


pro_df_mean = pd.DataFrame(pro_soy.loc[north_america].mean(axis=1), columns=['pro_mean'])


# In[22]:


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

def plot_pro(data):
    # 定义一个函数，将每个单词的首字母大写
    def capitalize_words(s):
        if isinstance(s, str):
            return ' '.join(word.capitalize() for word in s.split())
        else:
            return s

    # 创建图形和坐标轴，使用 PlateCarree 投影
    fig, ax = plt.subplots(figsize=(18, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    proj = ccrs.PlateCarree()

    # 添加国家边界、海岸线和海洋
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=2)
    ax.add_feature(cfeature.COASTLINE, linestyle='-', edgecolor='darkgray')
    ax.add_feature(cfeature.OCEAN)

    # 修改颜色归一化范围，将最大值设为200对应新的单位
    norm1 = mcolors.Normalize(vmin=0, vmax=2000)

    # 读取美国地图数据
    USA = gpd.read_file(f"/Users/limenghan/Desktop/地图/次国家/gadm41_USA_shp/gadm41_USA_1.shp")
    USA.rename(columns={'NAME_1': 'Area'}, inplace=True)

    # 将美国地图数据与产量数据进行合并
    data_USA = pd.merge(USA, data, on='Area')

    # 转换坐标参考系统 (CRS)
    target_crs = 'EPSG:4326'
    data_USA = data_USA.to_crs(target_crs)

    # 绘制美国地图
    USA.plot(ax=ax, color='grey', edgecolor='darkgrey')

    # 绘制产量数据
    data_USA.plot(ax=ax, column='pro_mean', linewidth=0.5, cmap='Greens', edgecolor="grey", norm=norm1)

    # 设置 colorbar 的单位和范围
    sm = mpl.cm.ScalarMappable(cmap='Greens', norm=norm1)  # 创建可映射对象
    sm.set_array([])  # 设置空数组
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical")
    cbar.set_ticks([0, 500, 1000, 1500, 2000])  # 设置 colorbar 的刻度
    cbar.set_ticklabels([0, 500, 1000, 1500, 2000])  # 自定义刻度标签
    cbar.set_label('Unit:10,000 tonne', fontsize=12)  # 设置单位标签

    # 设置图形标题和范围
    ax.set_title(f'(a)', loc='left')
    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])

    # 显示图形
    plt.show()


# In[23]:


pro_df_mean


# In[24]:


pro_df_mean=pro_df_mean.sort_values(by='pro_mean', ascending=False)
pro_df_mean


# In[25]:


pro_df_mean.sum()


# In[26]:


pro_df_mean.iloc[0:5].sum()


# In[27]:


79226770.68/1.509263e+08


# In[28]:


plot_pro(pro_df_mean/10000)


# In[29]:


pro_soy.T[north_america].sum(axis=1).mean()


# In[30]:


index_name_list=['nino34_MJJ','IOB_NDJ_1']


# In[31]:


def reg(index,index_name,yie):
    slope_df=pd.DataFrame(np.zeros((31,1)),index=anom_soy.loc[north_america].index)
    pv_df=pd.DataFrame(np.zeros((31,1)),index=anom_soy.loc[north_america].index)
    for i in yie.index:
        # 合并两个 Series 为一个 DataFrame
        merged_df = pd.concat([yie.loc[i], index], axis=1, keys=['yie', index_name])
        # 删除包含缺失值的行
        merged_df= merged_df.dropna()
        # 计算回归系数
        slope,intercept, r_value, p_value, std_err = stats.linregress(merged_df[index_name].iloc[:,0].values,merged_df['yie'].iloc[:,0].values)
        #对每个地点加权
        slope=slope*area_soy.loc[i].mean()*yield_5year_all.loc[i].mean()
        slope_df.loc[i]=slope
        pv_df.loc[i]=p_value
    return slope_df,pv_df
for index in index_name_list:
    locals()[f'slope_{index}'],locals()[f'pv_{index}']=reg(locals()[f'{index}'],index,yield_data/100)


# In[32]:


slope_IOB_NDJ_1.columns=['slope']
pv_IOB_NDJ_1.columns=['pv']


# In[33]:


result_IOB_NDJ_1 = pd.merge(slope_IOB_NDJ_1, pv_IOB_NDJ_1, on='Area')
result_IOB_NDJ_1


# In[34]:


result_IOB_NDJ_1.sort_values(by='slope', ascending=True)


# In[35]:


result_IOB_NDJ_1.sum()


# In[36]:


slope_sum = result_IOB_NDJ_1[result_IOB_NDJ_1['pv'] < 0.1]['slope'].sum()
slope_sum


# In[37]:


pro_soy.T[north_america].sum(axis=1).mean()


# In[38]:


6.026849e+06/150621650.71200004


# In[39]:



def pro_bar(data):
    # 将数据按第一列的值从大到小排序
    data_sorted = data.sort_values(by=data.columns[0])

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(data_sorted.index, data_sorted.iloc[:, 0].values / 10000, color='cornflowerblue')

    # 遍历每列绘制柱状图
    for col in range(0, len(data_sorted.index)):
        # 如果显著性小于0.05，则使用条纹填充
        if data_sorted.iloc[col, 1] < 0.1:
            ax.bar(data_sorted.index[col], data_sorted.iloc[col, 0] / 10000, 
                   hatch='/', color='cornflowerblue', label=f'{col} (p < 0.05)')
    
    # 添加标签和标题
    ax.set_ylabel('Values')
    ax.set_title('(c)', loc='left')
    ax.set_title('Unit: 10,000 tonne', loc='right')

    # 设置横轴标签竖排显示，并将刻度字体大小调整
    ax.set_xticklabels(data_sorted.index, rotation=90, ha='center', fontsize=10)

    # 显示图形
    plt.tight_layout()
    plt.show()


# In[40]:


pro_bar(result_IOB_NDJ_1)


# In[41]:


def plot_pro(data, ax):
    # 定义一个函数，将每个单词的首字母大写
    def capitalize_words(s):
        if isinstance(s, str):
            return ' '.join(word.capitalize() for word in s.split())
        else:
            return s

    # 创建图形和坐标轴，使用 PlateCarree 投影
    proj = ccrs.PlateCarree()

    # 添加国家边界、海岸线和海洋
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=2)
    ax.add_feature(cfeature.COASTLINE, linestyle='-', edgecolor='darkgray')
    ax.add_feature(cfeature.OCEAN)
    norm1 = mcolors.Normalize(vmin=0, vmax=2000)

    # 读取美国地图数据
    USA = gpd.read_file(f"/Users/limenghan/Desktop/地图/次国家/gadm41_USA_shp/gadm41_USA_1.shp")
    USA.rename(columns={'NAME_1': 'Area'}, inplace=True)

    # 将美国地图数据与产量数据进行合并
    data_USA = pd.merge(USA, data, on='Area')

    # 转换坐标参考系统 (CRS)
    target_crs = 'EPSG:4326'
    data_USA = data_USA.to_crs(target_crs)

    # 绘制美国地图
    USA.to_crs(target_crs).plot(ax=ax, color='grey', edgecolor='darkgrey')

    # 绘制产量数据
    data_USA.plot(ax=ax, column='pro_mean',legend=False, linewidth=0.5, cmap='Greens', edgecolor="grey",norm=norm1)

    # 设置 colorbar 并与 fig 关联
    sm = mpl.cm.ScalarMappable(cmap='Greens',norm=norm1)
    sm.set_array([])  # 不需要数据数组，只是为了 colorbar
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical",shrink=0.7)  
    ax.set_title('Unit: 10000 tonne', loc='right')
    # 设置图形标题和范围
    ax.set_title(f'(a)', loc='left')
    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])
    ax.set_aspect(4/3)   



# In[52]:


def plot_yield_index(yie_xr, index, type_, index_name, ax):
    # 定义一个函数，将每个单词的首字母大写
    def capitalize_words(s):
        if isinstance(s, str):
            return ' '.join(word.capitalize() for word in s.split())
        else:
            return s

    # 创建图形和坐标轴，使用 PlateCarree 投影
    proj = ccrs.PlateCarree()

    # 添加国家边界、海岸线和海洋
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=2)
    ax.add_feature(cfeature.COASTLINE, linestyle='-', edgecolor='darkgray')
    ax.add_feature(cfeature.OCEAN)

    # 设置颜色映射标准
    norm1 = mcolors.TwoSlopeNorm(vmin=-10, vmax=10, vcenter=0)
    cmap = 'PuOr'

    # 读取美国地图数据
    USA = gpd.read_file(f"/Users/limenghan/Desktop/地图/次国家/gadm41_USA_shp/gadm41_USA_1.shp")
    USA.rename(columns={'NAME_1': 'Area'}, inplace=True)

    # 将美国地图数据与产量数据进行合并
    data_USA = pd.merge(USA, yie_xr, on='Area')

    # 转换坐标参考系统 (CRS)
    target_crs = 'EPSG:4326'
    data_USA = data_USA.to_crs(target_crs)

    # 绘制美国地图
    USA.to_crs(target_crs).plot(ax=ax, color='grey', edgecolor='darkgrey')

    # 根据不同类型绘制不同的数据
    if type_ == 'corr':
        data_USA.plot(ax=ax, column='corr', legend=False, linewidth=0.5, cmap=cmap, edgecolor="grey", norm=norm1)
        data_USA[(data_USA['corr_pv'] <= 0.05)].to_crs(target_crs).plot(ax=ax, facecolor='none', edgecolor='grey', hatch='...', linewidth=0.1, alpha=0.75)
    else:
        data_USA.plot(ax=ax, column='reg_slope', legend=False, linewidth=0.5, cmap=cmap, edgecolor="grey", norm=norm1)
        data_USA[(data_USA['reg_pv'] <= 0.05)].to_crs(target_crs).plot(ax=ax, facecolor='none', edgecolor='grey', hatch='...', linewidth=0.1, alpha=0.75)

    # 创建一个 ScalarMappable 对象用于 colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm1)
    sm.set_array([])  # 不需要数据数组，只是为了 colorbar

    # 添加 colorbar 并调整大小
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", shrink=0.7)
    cbar.set_label('Yield Change (% per σ)', fontsize=15)
    # 设置图形标题和范围
    ax.set_title(f'(b)', loc='left')
    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])
    ax.set_aspect(4/3)   



# In[43]:



def pro_bar(data,ax):
    # 将数据按第一列的值从大到小排序
    data_sorted = data.sort_values(by=data.columns[0])

    # 绘制柱状图
    ax.bar(data_sorted.index, data_sorted.iloc[:, 0].values / 10000, color='cornflowerblue')

    # 遍历每列绘制柱状图
    for col in range(0, len(data_sorted.index)):
        # 如果显著性小于0.05，则使用条纹填充
        if data_sorted.iloc[col, 1] < 0.1:
            ax.bar(data_sorted.index[col], data_sorted.iloc[col, 0] / 10000, 
                   hatch='/', color='cornflowerblue', label=f'{col} (p < 0.05)')
    
    # 添加标签和标题
    ax.set_ylabel('Values')
    ax.set_title('(c)', loc='left')
    ax.set_title('Unit: 10000 tonne', loc='right')

    # 设置横轴标签竖排显示，并将刻度字体大小调整
    ax.set_xticklabels(data_sorted.index, rotation=90, ha='center')


# In[44]:


plt.rcParams['font.size']=17


# In[53]:


from matplotlib import gridspec

# 创建画布和GridSpec布局
fig = plt.figure(figsize=(16,14))
# 设置第一行两个图的列宽一样，第二行的图在下方占满
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

# 第一个图 (a)
ax = fig.add_subplot(gs[0, 0],projection=ccrs.PlateCarree())
plot_pro(pro_df_mean/10000,ax)

# 第三个图 (b)
ax = fig.add_subplot(gs[0, 1],projection=ccrs.PlateCarree())
plot_yield_index(reg,IOB_NDJ_1,'reg','IOB_NDJ_1',ax)

# 第三个图 (c)
ax = fig.add_subplot(gs[1, :])  # 横跨整行
pro_bar(result_IOB_NDJ_1,ax)

# 整体布局调整
plt.tight_layout()
plt.savefig('/Users/limenghan/Desktop/大豆代码结果/论文图/Figure2.png',dpi=300)
plt.show()

