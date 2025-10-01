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
import math
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


def apply_land_mask(slp_data):
    """
    使用 Cartopy 的陆地边界掩膜数据，只保留陆地区域
    """
    land = cfeature.NaturalEarthFeature('physical', 'land', '110m')
    geoms = list(land.geometries())

    lon, lat = np.meshgrid(slp_data['lon'].values, slp_data['lat'].values)
    mask = np.full(lon.shape, False)
    for geom in geoms:
        mask |= shapely.vectorized.contains(geom, lon, lat)
    
    masked_data = slp_data.where(mask)  # 海洋区域设为 NaN
    return masked_data


# In[4]:


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
IOB= pd.DataFrame(IOB,index=time)
nino34= pd.DataFrame(nino34, index=time)
IOB_NDJ_1=IOB[(IOB.index.month == 11) | (IOB.index.month == 12) | (IOB.index.month == 1)]
IOB_NDJ_1=IOB_NDJ_1[1:-2]
nino34_NDJ_1=nino34[(nino34.index.month == 11) | (nino34.index.month == 12) | (nino34.index.month == 1)]
nino34_NDJ_1=nino34_NDJ_1[1:-2]
##对气候变率去enso信号
##施密特
def orth(A,b):
    temp = np.dot(np.transpose(A),b)/np.dot(np.transpose(A),A)
    b_=  np.dot(temp,A)
    b_new = b-b_
    return b_new
IOB_NDJ_1=orth(nino34_NDJ_1[0],IOB_NDJ_1[0])
##三月平均（NDJ）
IOB_NDJ_1_reset =IOB_NDJ_1.reset_index(drop=True)
# 每三个一组，计算平均值
IOB_NDJ_1_averages = IOB_NDJ_1_reset.groupby(IOB_NDJ_1_reset.index // 3).mean()
##标准化
IOB_NDJ_1= (IOB_NDJ_1-IOB_NDJ_1.mean()) /IOB_NDJ_1.std()
##标准化
IOB_NDJ_1_averages= (IOB_NDJ_1_averages-IOB_NDJ_1_averages.mean()) /IOB_NDJ_1_averages.std()
IOB_NDJ_1_averages


# In[5]:


nino34_JAS_1=nino34[(nino34.index.month == 7) | (nino34.index.month == 8) | (nino34.index.month == 9)]
nino34_JAS_1=nino34_JAS_1[3:]


# In[6]:


##三月平均（NDJ）
nino34_JAS_1_reset =nino34_JAS_1.reset_index(drop=True)
# 每三个一组，计算平均值
nino34_JAS_1_averages = nino34_JAS_1_reset.groupby(nino34_JAS_1_reset.index // 3).mean()
##标准化
nino34JAS_1= (nino34_JAS_1-nino34_JAS_1.mean()) /nino34_JAS_1.std()
##标准化
nino34_JAS_1_averages= (nino34_JAS_1_averages-nino34_JAS_1_averages.mean()) /nino34_JAS_1_averages.std()
nino34_JAS_1_averages


# In[7]:


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


# In[8]:


##cloud
cld=xr.open_dataset('/Volumes/limenghan/气象要素/cru_ts4.04.1901.2019.cld.dat.nc')
cld=cld['cld']
cld=time_sel(cld,1978,2017)
cld=detrend_weather(cld)
cld=time_sel(cld,1979,2017)
if cld.lon.max()>180:
    cld=trans_lon(cld)
cld_ame=cld.sel(lon=slice(-130, -65),lat=slice(20, 55))


# In[9]:


##hgt
# 指定包含所有文件的文件夹路径
folder_path = '/Volumes/limenghan/气象要素/era5_hgt/'
# 使用通配符 '*' 来匹配文件夹中所有的 nc 文件
files = folder_path + '*.nc'
# 使用 open_mfdataset 函数来打开所有匹配的文件，并将它们合并成一个 Dataset
hgt= xr.open_mfdataset(files, combine='nested', concat_dim='time')
hgt= hgt.rename({'longitude': 'lon'})
hgt= hgt.rename({'latitude': 'lat'})

# 使用 coarsen 进行降采样
if len(hgt.lon)>180:
    # 定义经度和纬度的降采样因子
    lon_factor, lat_factor = int(len(hgt.lat)/360), int(len(hgt.lon)/720)  # 2° x 2° 对应的降采样因子
    # 使用 coarsen 方法进行降采样
    hgt= hgt.coarsen(lon=lon_factor, lat=lat_factor, boundary="trim").mean()
g = 9.80665  # 标准重力加速度
hgt200 = hgt.sel(level=200)['z'] / g
hgt500 = hgt.sel(level=500)['z'] / g
hgt850 = hgt.sel(level=850)['z'] / g
hgt200=time_sel(hgt200,1978,2017)
hgt500=time_sel(hgt500,1978,2017)
hgt850=time_sel(hgt850,1978,2017)
hgt200=detrend_weather(hgt200)
hgt500=detrend_weather(hgt500)
hgt850=detrend_weather(hgt850)
hgt200=time_sel(hgt200,1979,2017)
hgt500=time_sel(hgt500,1979,2017)
hgt850=time_sel(hgt850,1979,2017)


# In[10]:


# 指定包含所有文件的文件夹路径
folder_path = '/Volumes/limenghan/气象要素/surface_pressure/'
# 使用通配符 '*' 来匹配文件夹中所有的 nc 文件
files = folder_path + '*.nc'
# 使用 open_mfdataset 函数来打开所有匹配的文件，并将它们合并成一个 Dataset
slp= xr.open_mfdataset(files, combine='nested', concat_dim='time')
slp=slp.rename({'longitude': 'lon'})
slp=slp.rename({'latitude': 'lat'})
slp=slp['sp']
slp=time_sel(slp,1978,2018)
slp=detrend_weather(slp)
slp=detrend_weather(slp)
slp=time_sel(slp,1979,2017)
if slp.lon.max()>180:
    slp=trans_lon(slp)
slp_ame=slp.sel(lon=slice(-130, -65), lat=slice(55,20))
slp


# In[11]:


##pre
pre=xr.open_dataset('/Volumes/limenghan/气象要素/cru_ts4.04.1901.2019.pre.dat.nc')
pre=pre['pre']
pre=time_sel(pre,1978,2017)
pre=detrend_weather(pre)
pre=time_sel(pre,1979,2017)
if pre.lon.max()>180:
    pre=trans_lon(pre)
pre_ame=pre.sel(lon=slice(-130, -65),lat=slice(20,55))


# In[12]:


##smroot
smroot=xr.open_dataset('/Volumes/limenghan/气象要素/era5_volumetric_soil_water_layer_2.nc')
smroot= smroot.rename({'longitude': 'lon'})
smroot= smroot.rename({'latitude': 'lat'})
smroot= smroot.rename({'valid_time': 'time'})
# 使用 coarsen 进行降采样
if len(smroot.lon)>180:
    # 定义经度和纬度的降采样因子
    lon_factor, lat_factor = int(len(smroot.lat)/360), int(len(smroot.lon)/720)  # 对应的降采样因子
    # 使用 coarsen 方法进行降采样
    smroot= smroot.coarsen(lon=lon_factor, lat=lat_factor, boundary="trim").mean()
smroot=smroot['swvl2']
smroot=time_sel(smroot,1978,2017)
smroot=detrend_weather(smroot)
smroot=time_sel(smroot,1979,2017)
if smroot.lon.max()>180:
    smroot=trans_lon(smroot)
smroot_ame=smroot.sel(lon=slice(-130, -65), lat=slice(55,20))


# In[13]:


##vwnd
# 指定包含所有文件的文件夹路径
folder_path = '/Volumes/limenghan/气象要素/v_component_of_wind/'
# 使用通配符 '*' 来匹配文件夹中所有的 nc 文件
files = folder_path + '*.nc'
# 使用 open_mfdataset 函数来打开所有匹配的文件，并将它们合并成一个 Dataset
vwnd= xr.open_mfdataset(files, combine='nested', concat_dim='time')
vwnd= vwnd.rename({'longitude': 'lon'})
vwnd= vwnd.rename({'latitude': 'lat'})
# 使用 coarsen 进行降采样
if len(vwnd.lon)>180:
    # 定义经度和纬度的降采样因子
    lon_factor, lat_factor = int(len(vwnd.lat)/90), int(len(vwnd.lon)/180)  # 2° x 2° 对应的降采样因子
    # 使用 coarsen 方法进行降采样
    vwnd= vwnd.coarsen(lon=lon_factor, lat=lat_factor, boundary="trim").mean()
##uwnd
# 指定包含所有文件的文件夹路径
folder_path = '/Volumes/limenghan/气象要素/u_component_of_wind/'
# 使用通配符 '*' 来匹配文件夹中所有的 nc 文件
files = folder_path + '*.nc'
# 使用 open_mfdataset 函数来打开所有匹配的文件，并将它们合并成一个 Dataset
uwnd= xr.open_mfdataset(files, combine='nested', concat_dim='time')
uwnd= uwnd.rename({'longitude': 'lon'})
uwnd= uwnd.rename({'latitude': 'lat'})
# 使用 coarsen 进行降采样
if len(uwnd.lon)>180:
    # 定义经度和纬度的降采样因子
    lon_factor, lat_factor = int(len(uwnd.lat)/90), int(len(uwnd.lon)/180)  # 2° x 2° 对应的降采样因子
    # 使用 coarsen 方法进行降采样
    uwnd= uwnd.coarsen(lon=lon_factor, lat=lat_factor, boundary="trim").mean()
uwnd=uwnd.sel(level=925)['u']
vwnd=vwnd.sel(level=925)['v']
uwnd=time_sel(uwnd,1978,2018)
vwnd=time_sel(vwnd,1978,2018)
uwnd=detrend_weather(uwnd)
vwnd=detrend_weather(vwnd)
uwnd=time_sel(uwnd,1979,2017)
vwnd=time_sel(vwnd,1979,2017)


# In[14]:


##tmx
tmx=xr.open_dataset('/Volumes/limenghan/气象要素/cru_ts4.07.1901.2022.tmx.dat.nc')
tmx=tmx['tmx']
tmx=time_sel(tmx,1978,2017)
tmx=detrend_weather(tmx)
tmx=time_sel(tmx,1979,2017)
if tmx.lon.max()>180:
    tmx=trans_lon(tmx)
tmx_ame=tmx.sel(lon=slice(-130, -65),lat=slice(20,55))


# In[15]:


# 定义一个函数用于按时间每三个进行平均
def group_mean(da):
    # 确保时间维度是 datetime 对象
    da['time'] = pd.to_datetime(da['time'].values)
    
    # 进行滚动平均
    rolling_mean = da.rolling(time=3, center=False).mean()
    
    # 选择每三个的时间点
    grouped_mean = rolling_mean.isel(time=slice(2, None, 3))
    
    return grouped_mean


# In[16]:


##气象要素去nino
# 对每个地点的时间序列应用 orth 函数
def orth_hgt(data,month):
    if month==12:
        s=data.sel(time=data['time.month'].isin([11,12, 1]))
        s= s.sel(time=slice('1979-11-01', '2017-02-01'))
    elif month==1:
        s=data.sel(time=data['time.month'].isin([12, 1,2]))
        s= s.sel(time=slice('1979-11-01', '2017-03-01'))
    elif 1<month<7:
        s=data.sel(time=data['time.month'].isin([month, month+1,month+2]))
        s= s.sel(time=slice(f'1980-0{month}-01', f'2017-0{month+3}-01'))
    else:
        s=data.sel(time=data['time.month'].isin([month, month+1,month+2]))
        s= s.sel(time=slice(f'1980-0{month}-01', f'2017-{month+3}-01'))
    result = []
    for alat in data['lat']:
        result_lat=[]
        for alon in data['lon']:
            result_lat.append(orth(nino34_NDJ_1[0], s.sel(lat=alat, lon=alon)).values)
        result.append(result_lat)
    data_= xr.DataArray(result, coords=[data['lat'],data['lon'],s['time']], dims=[ 'lat', 'lon','time'])
    return data_
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
def reg_location(x_data,y_data,variable_name,month):
    ##去nino:
    y_data=orth_hgt(y_data,month)
    if variable_name not in ['hgt200','hgt500','hgt850','slp','uwnd','vwnd','uwnd925','vwnd925','uwnd500','vwnd500','uwnd200','vwnd200','nwvf','ewvf']:
        #标准化气象要素场
        y_data=(y_data - y_data.mean(dim='time')) / y_data.std(dim='time')
    y=y_data
    # 使用 coarsen 进行降采样
    reg_array=np.zeros((len(y.lat),len(y.lon)))
    p_value_array=np.zeros((len(y.lat),len(y.lon)))
    lon=y.lon
    lat=y.lat
    for alon in range(0,len(y.lon)):
        for alat in range(0,len(y.lat)):
            reg_array[alat,alon], intercept, r_value, p_value_array[alat,alon], std_err = stats.linregress(x_data,y.sel(lat=lat[alat],lon=lon[alon]))
    reg_array = xr.DataArray(reg_array, coords={'lat': y.lat, 'lon': y.lon}, dims=['lat', 'lon'])
    p_value_array = xr.DataArray(p_value_array, coords={'lat': y.lat, 'lon': y.lon}, dims=['lat', 'lon'])
    reg_dataset = xr.Dataset(
    {"slope": reg_array, "pv": p_value_array},
    attrs={"description": "Regression Results"}
    )
    return reg_dataset


# In[17]:


def nino_reg_location(x_data,y_data,variable_name,month):
    if month==12:
        y_data=y_data.sel(time=y_data['time.month'].isin([11,12, 1]))
        y_data= s.sel(time=slice('1979-11-01', '2017-02-01'))
    elif month==1:
        y_data=y_data.sel(time=y_data['time.month'].isin([12, 1,2]))
        y_data= y_data.sel(time=slice('1979-11-01', '2017-03-01'))
    elif 1<month<7:
        y_data=y_data.sel(time=y_data['time.month'].isin([month, month+1,month+2]))
        y_data=y_data.sel(time=slice(f'1980-0{month}-01', f'2017-0{month+3}-01'))
    else:
        y_data=y_data.sel(time=y_data['time.month'].isin([month, month+1,month+2]))
        y_data= y_data.sel(time=slice(f'1980-0{month}-01', f'2017-{month+3}-01'))
    if variable_name not in ['hgt200','hgt500','hgt850','slp','uwnd','vwnd','uwnd925','vwnd925','uwnd500','vwnd500','uwnd200','vwnd200','nwvf','ewvf']:
        #标准化气象要素场
        y_data=(y_data - y_data.mean(dim='time')) / y_data.std(dim='time')
    y=y_data
    # 使用 coarsen 进行降采样
    reg_array=np.zeros((len(y.lat),len(y.lon)))
    p_value_array=np.zeros((len(y.lat),len(y.lon)))
    lon=y.lon
    lat=y.lat
    for alon in range(0,len(y.lon)):
        for alat in range(0,len(y.lat)):
            reg_array[alat,alon], intercept, r_value, p_value_array[alat,alon], std_err = stats.linregress(x_data,y.sel(lat=lat[alat],lon=lon[alon]))
    reg_array = xr.DataArray(reg_array, coords={'lat': y.lat, 'lon': y.lon}, dims=['lat', 'lon'])
    p_value_array = xr.DataArray(p_value_array, coords={'lat': y.lat, 'lon': y.lon}, dims=['lat', 'lon'])
    reg_dataset = xr.Dataset(
    {"slope": reg_array, "pv": p_value_array},
    attrs={"description": "Regression Results"}
    )
    return reg_dataset


# In[18]:


def tran_lon_360(da):
    lon_name = 'lon'
    da['longitude_adjusted'] = xr.where(da[lon_name] < 0, da[lon_name]+360, da[lon_name])
    da = (da
          .swap_dims({lon_name: 'longitude_adjusted'})
          .sel(**{'longitude_adjusted': sorted(da.longitude_adjusted)})
          .drop(lon_name))
    da = da.rename({'longitude_adjusted': lon_name})
    return da


# In[19]:


##tmp
tmp=xr.open_dataset('/Volumes/limenghan/气象要素/cru_ts4.07.1901.2022.tmp.dat.nc')
tmp=tmp['tmp']
tmp=time_sel(tmp,1978,2017)
tmp=detrend_weather(tmp)
tmp=time_sel(tmp,1979,2017)
if tmp.lon.max()>180:
    tmp=trans_lon(tmp)
tmp_ame=tmp.sel(lon=slice(-130, -65),lat=slice(20,55))
##vap
vap=xr.open_dataset('/Volumes/limenghan/气象要素/cru_ts4.07.1901.2022.vap.dat.nc')
vap=vap['vap']
vap=time_sel(vap,1978,2017)
vap=detrend_weather(vap)
vap=time_sel(vap,1979,2017)
if vap.lon.max()>180:
    vap=trans_lon(vap)
vap_ame=vap.sel(lon=slice(-130, -65),lat=slice(20,55))
# 将值为 0 的点 mask 掉，设为 NaN
tmp_ame = tmp_ame.where(tmp_ame!= 0, np.nan)
vap_ame = vap_ame.where(vap_ame!= 0, np.nan)
##vpd
e0 = 6.108 *np.exp(17.27 * tmp_ame / (tmp_ame + 237.3))
vpd=e0-vap_ame
vpd


# In[20]:


##dtr
dtr=xr.open_dataset('/Volumes/limenghan/气象要素/cru_ts4.07.1901.2022.dtr.dat.nc')
dtr=dtr['dtr']
dtr=time_sel(dtr,1978,2017)
dtr=detrend_weather(dtr)
dtr=time_sel(dtr,1979,2017)
if dtr.lon.max()>180:
    dtr=trans_lon(dtr)
dtr_ame=dtr.sel(lon=slice(-130, -65),lat=slice(20, 55))


# In[64]:


##修改图
def plot_cli_hgt_1_ax(data, name,uwnd_data,vwnd_data,ax):
    # 创建一个图形和子图
    hgt_data = data['slope']
    hgt_data_pv = data['pv']
    lon = data['lon']
    lat = data['lat']
    
    # 绘制气象要素填色
    # 设置填色图的数据范围
    other_contourf = ax.contourf(lon, lat, hgt_data, transform=ccrs.PlateCarree(), levels=13,vmin=-18,vmax=18, cmap='RdBu_r', extend='both')
    
    # 添加颜色条
    cbar = fig.colorbar(other_contourf, ax=ax, orientation='vertical',  pad=0.05, aspect=50, shrink=1,  extend='both')
    
    # 添加海岸线和国界线
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linestyle='-',linewidth=0.5,  edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    
    # 绘制风速箭头
    # 简化数据以提高可读性
    vwnd_data=vwnd_data['slope']
    uwnd_data=uwnd_data['slope']
    skip =2
    quiver = ax.quiver(uwnd_data['lon'][::skip].values, uwnd_data['lat'][::skip].values,
              uwnd_data.sel(lon=slice(None, None, skip), lat=slice(None, None, skip)).values, 
              vwnd_data.sel(lon=slice(None, None, skip), lat=slice(None, None, skip)).values,
              transform=ccrs.PlateCarree(), 
              scale=25, color='grey', width=0.0015)
    ax.quiverkey(quiver, 0.91, 1.03, 1, "1 m·s$^{-1}$",labelpos='E', coordinates='axes', fontproperties={ 'family':'Times New Roman'})
    # 添加经纬度标签
    ax.set_xticks([-120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xticklabels(['120° W', '60° W', '0', '60° E', '120° E'])
    ax.set_yticklabels(['90° S', '60° S', '30° S','0', '30° N', '60° N', '90° N'])
    cbar.set_label('m / σ', fontsize=15)
    # 设置图形标题
    ax.set_title('(a) GPH200',loc='left')
    ax.set_ylim([0, 80])
    ax.set_xlim([-110, 120])   


# In[22]:


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import numpy as np
import shapely.vectorized
import xarray as xr


# In[65]:


def plot_cli_smroot_ax(data, name,ax):
    # 创建一个图形和子图
    smroot_data = data['slope']
    data_pv = data['pv']
    lon = data['lon']
    lat = data['lat']
    # 应用陆地掩膜：只保留陆地区域
    smroot_data = apply_land_mask(smroot_data)
    data_pv  = apply_land_mask(data_pv )
    # 绘制气象要素填色
    # 设置填色图的数据范围
    other_contourf = ax.contourf(lon, lat, smroot_data, transform=ccrs.PlateCarree(), vmin=-0.36,vmax=0.36,levels=13, cmap='BrBG',
                                 extend='both')

    # 添加颜色条
    cbar = fig.colorbar(other_contourf, ax=ax, orientation='vertical',  pad=0.05, aspect=50, shrink=1,  extend='both')

    # 添加海岸线和国界线
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=2)
    ax.add_feature(cfeature.COASTLINE, linestyle='-', edgecolor='darkgrey')
    ax.add_feature(cfeature.OCEAN)
    # 读取 shapefile
    shapefile_path = '/Users/limenghan/Desktop/地图/次国家/gadm41_USA_shp/gadm41_USA_1.shp'
    shp = gpd.read_file(shapefile_path)
    shp.plot(ax=ax, color='none', edgecolor='#808080') 

    # 添加经纬度标签
    ax.set_xticks([-120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xticklabels(['120° W', '60° W', '0', '60° E', '120° E'])
    ax.set_yticklabels(['90° S', '60° S', '30° S','0', '30° N', '60° N', '90° N'])
    # 根据 other_data_pv 值添加斜线填充区域
    mask = data_pv < 0.1
    lons, lats = np.meshgrid(lon, lat)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j]:
                ax.fill_between([lons[i, j], lons[i, j] + 1], lats[i, j], lats[i, j] + 1, 
                                color='none', edgecolor='grey', hatch='///', linewidth=0.1, alpha=0.75)
    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])
    cbar.set_label('σ / σ', fontsize=15)
    # 设置图形标题
    ax.set_title('(e) SMroot', loc='left')


# In[66]:


def plot_cli_vpd_ax(data, name,ax):
    # 创建一个图形和子图
    smroot_data = data['slope']
    data_pv = data['pv']
    lon = data['lon']
    lat = data['lat']

    # 绘制气象要素填色
    # 设置填色图的数据范围 
    other_contourf = ax.contourf(lon, lat, smroot_data, transform=ccrs.PlateCarree(), vmin=-0.3,vmax=0.3,levels=13, cmap='PiYG_r',
                                 extend='both')

    # 添加颜色条 
    cbar = fig.colorbar(other_contourf, ax=ax, orientation='vertical', pad=0.05, aspect=50, shrink=1,  extend='both')

    # 添加海岸线和国界线
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=2)
    ax.add_feature(cfeature.COASTLINE, linestyle='-', edgecolor='darkgrey')
    ax.add_feature(cfeature.OCEAN)
    # 读取 shapefile
    shapefile_path = '/Users/limenghan/Desktop/地图/次国家/gadm41_USA_shp/gadm41_USA_1.shp'
    shp = gpd.read_file(shapefile_path)
    shp.plot(ax=ax, color='none', edgecolor='grey')  # 绘制 shapefile

    # 添加经纬度标签
    ax.set_xticks([-120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xticklabels(['120° W', '60° W', '0', '60° E', '120° E'])
    ax.set_yticklabels(['90° S', '60° S', '30° S','0', '30° N', '60° N', '90° N'])
    # 根据 other_data_pv 值添加斜线填充区域
    mask = data_pv < 0.1
    lons, lats = np.meshgrid(lon, lat)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j]:
                ax.fill_between([lons[i, j], lons[i, j] + 1], lats[i, j], lats[i, j] + 1, 
                                color='none', edgecolor='grey', hatch='///', linewidth=0.1, alpha=0.75)
    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])
    cbar.set_label('σ / σ', fontsize=15)
    # 设置图形标题
    ax.set_title(f'(d) VPD', loc='left')


# In[67]:


##修改图
def plot_cli_other_2_ax(data, name, variable,ax):
    # 创建一个图形和子图
    other_data = data['slope']
    data_pv = data['pv']
    lon = data['lon']
    lat = data['lat']
    # 绘制气象要素填色
    # 设置填色图的数据范围
    if variable=='tmp':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, vmin=-0.25,vmax=0.25,cmap='coolwarm', extend='both')
    elif variable=='tmx' or variable=='tmxx':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, vmin=-0.4,vmax=0.4,cmap='coolwarm', extend='both')
    elif variable=='pre':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13,vmin=-0.2,vmax=0.2, cmap='PRGn', extend='both')
    elif variable=='evap':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, vmin=-0.3,vmax=0.3, cmap='PuOr',extend='both')
    elif variable=='snowfall' or variable=='snow_depth':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, cmap='PuOr', extend='both')
    elif variable=='msdwswrf':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, cmap='bwr', extend='both')
    elif variable=='cld':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, cmap='PuOr',vmin=-0.25,vmax=0.25, extend='both')
    elif variable=='RH':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, cmap='PuOr',vmin=-0.25,vmax=0.25, extend='both')

    
    # 添加颜色条
    cbar = fig.colorbar(other_contourf, ax=ax, orientation='vertical', pad=0.05, aspect=50, shrink=1, extend='both')
    
    # 添加海岸线和国界线
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=2)
    ax.add_feature(cfeature.COASTLINE, linestyle='-', edgecolor='darkgrey')
    ax.add_feature(cfeature.OCEAN)
    # 读取 shapefile
    shapefile_path = '/Users/limenghan/Desktop/地图/次国家/gadm41_USA_shp/gadm41_USA_1.shp'
    shp = gpd.read_file(shapefile_path)
    shp.plot(ax=ax, color='none', edgecolor='#808080') 
    
     # 添加经纬度标签
    ax.set_xticks([-120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xticklabels(['120° W', '60° W', '0', '60° E', '120° E'])
    ax.set_yticklabels(['90° S', '60° S', '30° S','0', '30° N', '60° N', '90° N'])
    # 根据 other_data_pv 值添加斜线填充区域
    mask = data_pv < 0.1
    lons, lats = np.meshgrid(lon, lat)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j]:
                ax.fill_between([lons[i, j], lons[i, j] + 1], lats[i, j], lats[i, j] + 1,color='none', edgecolor='grey', hatch='///', linewidth=0.1, alpha=0.75)

    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])
    cbar.set_label('σ / σ', fontsize=15)
    # 设置图形标题
    if variable=='tmx':
        ax.set_title(f'(c) Tmx',loc='left')
    elif variable=='pre':
        ax.set_title(f'(b) Pre',loc='left')
    elif variable=='tmxx':
        ax.set_title(f'(e) Tmx',loc='left')
    elif variable=='tmp':
        ax.set_title(f'(c) Tmp',loc='left')
    elif variable=='RH':
        ax.set_title(f'(c) RH',loc='left')


# In[26]:


for variable in ['pre','tmx','tmp']:
    locals()[f'reg_IOB_NDJ_1_{variable}_NDJ']=reg_location(IOB_NDJ_1,locals()[f'{variable}_ame'],f'{variable}',12)
reg_IOB_NDJ_1_smroot_NDJ=reg_location(IOB_NDJ_1,smroot_ame,'smroot',12)
reg_IOB_NDJ_1_hgt200_NDJ=reg_location(IOB_NDJ_1,hgt200,'hgt200',12)
reg_IOB_NDJ_1_uwnd_NDJ=reg_location(IOB_NDJ_1,uwnd,'uwnd',12)
reg_IOB_NDJ_1_vwnd_NDJ=reg_location(IOB_NDJ_1,vwnd,'vwnd',12)
reg_IOB_NDJ_1_vpd_NDJ=reg_location(IOB_NDJ_1,vpd,'vpd',12)


# In[27]:


reg_IOB_NDJ_1_hgt200_NDJ=reg_location(IOB_NDJ_1,hgt200,'hgt200',12)


# In[28]:


reg_IOB_NDJ_1_tmp_NDJ=reg_location(IOB_NDJ_1,tmp,'tmp',12)


# In[68]:


from matplotlib import gridspec

# 创建画布和GridSpec布局
fig = plt.figure(figsize=(15,15))
# 设置第一行两个图的列宽一样，第二行的图在下方占满
gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1], wspace=0.1)

# 第一个图 (a)
ax = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree(central_longitude=180))
plot_cli_hgt_1_ax(reg_IOB_NDJ_1_hgt200_NDJ,'reg_IOB_NDJ_1_hgt200_NDJ',reg_IOB_NDJ_1_uwnd_NDJ,reg_IOB_NDJ_1_vwnd_NDJ,ax)

# 第二个图 (b)
ax = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
plot_cli_other_2_ax(reg_IOB_NDJ_1_pre_NDJ,'reg_IOB_NDJ_1_pre_NDJ','pre',ax)

# 第三个图 (c)
ax = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
plot_cli_other_2_ax(reg_IOB_NDJ_1_tmp_NDJ,'reg_IOB_NDJ_1_tmp_NDJ','tmp',ax)

# 第四个图 (d)
ax = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())
plot_cli_vpd_ax(locals()[f'reg_IOB_NDJ_1_vpd_NDJ'],f'reg_IOB_NDJ_1_vpd_NDJ',ax)
# 第五个图 (e)
ax = fig.add_subplot(gs[2, 1], projection=ccrs.PlateCarree())
plot_cli_smroot_ax(locals()[f'reg_IOB_NDJ_1_smroot_NDJ'],f'reg_IOB_NDJ_1_smroot_NDJ',ax)


# 整体布局调整
plt.tight_layout()
plt.savefig('/Users/limenghan/Desktop/大豆代码结果/论文图/Figure4.png',dpi=300)
plt.show()


# In[ ]:





# In[30]:


#evap
# 指定包含所有文件的文件夹路径
folder_path = '/Volumes/limenghan/气象要素/evap/'
# 使用通配符 '*' 来匹配文件夹中所有的 nc 文件
files = folder_path + '*.nc'
# 使用 open_mfdataset 函数来打开所有匹配的文件，并将它们合并成一个 Dataset
evap= xr.open_mfdataset(files, combine='nested', concat_dim='time')
evap= evap.rename({'longitude': 'lon'})
evap= evap.rename({'latitude': 'lat'})
# 使用 coarsen 进行降采样
if len(evap.lon)>180:
    # 定义经度和纬度的降采样因子
    lon_factor, lat_factor = int(len(evap.lat)/360), int(len(evap.lon)/720)  # 对应的降采样因子
    # 使用 coarsen 方法进行降采样
    evap= evap.coarsen(lon=lon_factor, lat=lat_factor, boundary="trim").mean()
evap=evap['e']
##时间切片
evap=time_sel(evap,1979,2017)
evap=detrend_weather(evap)
if evap.lon.max()>180:
    evap=trans_lon(evap)
evap_ame=evap.sel(lon=slice(-130, -65), lat=slice(55,20))


# In[31]:


##tmp
tmp=xr.open_dataset('/Volumes/limenghan/气象要素/cru_ts4.07.1901.2022.tmp.dat.nc')
tmp=tmp['tmp']
tmp=time_sel(tmp,1978,2017)
tmp=detrend_weather(tmp)
tmp=time_sel(tmp,1979,2017)
if tmp.lon.max()>180:
    tmp=trans_lon(tmp)
tmp_ame=tmp.sel(lon=slice(-130, -65),lat=slice(20, 55))


# In[32]:


##vap
vap=xr.open_dataset('/Volumes/limenghan/气象要素/cru_ts4.07.1901.2022.vap.dat.nc')
vap=vap['vap']
vap=time_sel(vap,1978,2017)
vap=detrend_weather(vap)
vap=time_sel(vap,1979,2017)
if vap.lon.max()>180:
    vap=trans_lon(vap)
vap_ame=vap.sel(lon=slice(-130, -65),lat=slice(20,55))
# 将值为 0 的点 mask 掉，设为 NaN
tmp_ame = tmp_ame.where(tmp_ame!= 0, np.nan)
vap_ame = vap_ame.where(vap_ame!= 0, np.nan)
##vpd
e0 = 6.108 *np.exp(17.27 * tmp_ame / (tmp_ame + 237.3))
vpd=e0-vap_ame
vpd_ame=vpd.sel(lon=slice(-130, -65),lat=slice(20, 55))
RH=vap_ame/e0


# In[33]:


RH_ame=vap_ame/e0


# In[34]:


for variable in ['tmp']:
    locals()[f'reg_IOB_NDJ_1_{variable}_NDJ']=reg_location(IOB_NDJ_1,locals()[f'{variable}_ame'],f'{variable}',12)


# In[69]:


from matplotlib import gridspec

# 创建画布和GridSpec布局
fig = plt.figure(figsize=(15,15))
# 设置第一行两个图的列宽一样，第二行的图在下方占满
gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1], wspace=0.1)

# 第一个图 (a)
ax = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree(central_longitude=180))
plot_cli_hgt_1_ax(reg_IOB_NDJ_1_hgt200_NDJ,'reg_IOB_NDJ_1_hgt200_NDJ',reg_IOB_NDJ_1_uwnd_NDJ,reg_IOB_NDJ_1_vwnd_NDJ,ax)

# 第二个图 (b)
ax = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
plot_cli_other_2_ax(reg_IOB_NDJ_1_pre_NDJ,'reg_IOB_NDJ_1_pre_NDJ','pre',ax)

# 第三个图 (c)
ax = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
plot_cli_other_2_ax(reg_IOB_NDJ_1_tmp_NDJ,'reg_IOB_NDJ_1_tmp_NDJ','tmp',ax)

# 第四个图 (d)
ax = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())
plot_cli_vpd_ax(locals()[f'reg_IOB_NDJ_1_vpd_NDJ'],f'reg_IOB_NDJ_1_vpd_NDJ',ax)
# 第五个图 (e)
ax = fig.add_subplot(gs[2, 1], projection=ccrs.PlateCarree())
plot_cli_smroot_ax(locals()[f'reg_IOB_NDJ_1_smroot_NDJ'],f'reg_IOB_NDJ_1_smroot_NDJ',ax)


# 整体布局调整
plt.tight_layout()
plt.show()


# In[ ]:





# # JAS

# In[36]:


###指数与海温相关
def corr_location(x_data,y_data,month):
    y=orth_hgt(y_data,month)
    corr=np.zeros((len(y['lat']),len(y['lon'])))
    pv=np.zeros((len(y['lat']),len(y['lon'])))
    lon=y['lon']
    lat=y['lat']
    # 将气候指数标准化
    x_data= (x_data-x_data.mean()) /x_data.std()
    for alon in range(0,len(y['lon'])):
        for alat in range(0,len(y['lat'])):
            if not np.any(np.isnan(y.sel(lat=lat[alat],lon=lon[alon]))):
                corr[alat,alon], pv[alat,alon] =stats.pearsonr(x_data,y.sel(lat=lat[alat],lon=lon[alon]))
    corr_array = xr.DataArray(corr, coords={'lat': y.lat, 'lon': y.lon}, dims=['lat', 'lon'])
    p_value_array = xr.DataArray(pv, coords={'lat': y.lat, 'lon': y.lon}, dims=['lat', 'lon'])
    corr_dataset = xr.Dataset(
    {"corr": corr_array, "pv": p_value_array},
    attrs={"description": "Correlation Results"}
    )
        #对异常值掩码
    # 创建一个掩码，标识小于-100的值
    mask = corr_dataset==0
    # 将掩码应用到 DataArray，将小于-100的值替换为 NaN
    corr_dataset= xr.where(mask, np.nan, corr_dataset)
    return corr_dataset


# In[37]:


sst_IOB_NDJ_1_JAS=corr_location(IOB_NDJ_1,sst,8)


# In[38]:


IOB_NDJ_1


# In[39]:


nino34_JAS_1=nino34_JAS_1[0]


# In[40]:


reg_nino34_JAS_1_pre_JAS=nino_reg_location(nino34_JAS_1,pre_ame,'pre_ame',8)
reg_nino34_JAS_1_tmx_JAS=nino_reg_location(nino34_JAS_1,tmx_ame,'tmx_ame',8)
reg_nino34_JAS_1_cld_JAS=nino_reg_location(nino34_JAS_1,cld_ame,'cld_ame',8)
reg_nino34_JAS_1_smroot_JAS=nino_reg_location(nino34_JAS_1,smroot_ame,'smroot_ame',8)
reg_nino34_JAS_1_slp_JAS=nino_reg_location(nino34_JAS_1,slp,'slp',8)
reg_nino34_JAS_1_vpd_JAS=nino_reg_location(nino34_JAS_1,vpd,'vpd_ame',8)
reg_nino34_JAS_1_dtr_JAS=nino_reg_location(nino34_JAS_1,dtr,'dtr_ame',8)
reg_nino34_JAS_1_hgt200_JAS=nino_reg_location(nino34_JAS_1,hgt200,'hgt200',8)
reg_nino34_JAS_1_uwnd_JAS=nino_reg_location(nino34_JAS_1,uwnd,'uwnd',8)
reg_nino34_JAS_1_vwnd_JAS=nino_reg_location(nino34_JAS_1,vwnd,'vwnd',8)


# In[41]:


reg_nino34_JAS_1_hgt200_JAS=nino_reg_location(nino34_JAS_1,hgt200,'hgt200',8)


# In[42]:


reg_IOB_NDJ_1_slp_JAS=reg_location(IOB_NDJ_1,slp,'slp',8)


# In[43]:


reg_IOB_NDJ_1_pre_JAS=reg_location(IOB_NDJ_1,pre_ame,'pre_ame',8)
reg_IOB_NDJ_1_tmx_JAS=reg_location(IOB_NDJ_1,tmx_ame,'tmx_ame',8)
reg_IOB_NDJ_1_smroot_JAS=reg_location(IOB_NDJ_1,smroot_ame,'smroot_ame',8)
reg_IOB_NDJ_1_vpd_JAS=reg_location(IOB_NDJ_1,vpd,'vpd_ame',8)
reg_IOB_NDJ_1_dtr_JAS=reg_location(IOB_NDJ_1,dtr,'dtr_ame',8)


# In[44]:


reg_IOB_NDJ_1_hgt200_JAS=reg_location(IOB_NDJ_1,hgt200,'hgt200',8)
reg_IOB_NDJ_1_uwnd_JAS=reg_location(IOB_NDJ_1,uwnd,'uwnd',8)
reg_IOB_NDJ_1_vwnd_JAS=reg_location(IOB_NDJ_1,vwnd,'vwnd',8)


# In[45]:


reg_IOB_NDJ_1_hgt200_JAS=reg_location(IOB_NDJ_1,hgt200,'hgt200',8)


# In[70]:


##修改图
def plot_cli_hgt_1_ax(data, name,uwnd_data,vwnd_data,ax):
    # 创建一个图形和子图
    hgt_data = data['slope']
    hgt_data_pv = data['pv']
    lon = data['lon']
    lat = data['lat']
    
    # 绘制气象要素填色
    # 设置填色图的数据范围
    other_contourf = ax.contourf(lon, lat, hgt_data, transform=ccrs.PlateCarree(), levels=13,vmin=-18,vmax=18, cmap='RdBu', extend='both')
    
    # 添加颜色条
    cbar = fig.colorbar(other_contourf, ax=ax, orientation='vertical',  pad=0.05, aspect=50, shrink=1,  extend='both')
    
    # 添加海岸线和国界线
    ax.add_feature(cfeature.BORDERS, linestyle='-',linewidth=0.5,  edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linestyle='-', linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    
    # 绘制风速箭头
    # 简化数据以提高可读性
    vwnd_data=vwnd_data['slope']
    uwnd_data=uwnd_data['slope']
    skip =2
    quiver = ax.quiver(uwnd_data['lon'][::skip].values, uwnd_data['lat'][::skip].values,
              uwnd_data.sel(lon=slice(None, None, skip), lat=slice(None, None, skip)).values, 
              vwnd_data.sel(lon=slice(None, None, skip), lat=slice(None, None, skip)).values,
              transform=ccrs.PlateCarree(), 
              scale=25, color='grey', width=0.0015)
    ax.quiverkey(quiver, 0.91, 1.03, 1, "1 m/s",labelpos='E', coordinates='axes', fontproperties={ 'family':'Times New Roman'})
    # 添加经纬度标签
    ax.set_xticks([-120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xticklabels(['120W', '60W', '0', '60E', '120E'])
    
    cbar.set_label('m / σ', fontsize=15)
    
    # 设置图形标题
    ax.set_title('(b) GPH200*(-1)',loc='left')
    ax.set_ylim([0, 80])
    ax.set_xlim([-110, 120])   


# In[71]:


def plot_cli_sst_ax(data, slp, name, number,ax):
    # 创建一个图形和子图
    data=tran_lon_360(data)
    slp=tran_lon_360(slp)
    sst_data = data['corr']
    sst_data_pv = data['pv']
    slp_data = slp['slope']
    lon = data['lon']
    lat = data['lat']    
    # 确定等高线级别
    levels = np.linspace(-100, 100, 20)  # 生成13个等高线级别，包括负值和正值

    # 绘制等高线
    contour = ax.contour(slp['lon'], slp['lat'], slp_data,
                         levels=levels, color='grey', linewidths=1, transform=ccrs.PlateCarree(central_longitude=0))
    # 分别设置实线和虚线
    for i, lvl in enumerate(contour.levels):
        if lvl < 0:
            contour.collections[i].set_linestyle('dashed')  # 负值用虚线
        else:
            contour.collections[i].set_linestyle('solid')   # 正值用实线
    # 添加等高线标签
    ax.clabel(contour, inline=True, fontsize=8, fmt='%1.0f')

    # 设置填色图的数据范围
    other_contourf = ax.contourf(lon, lat, sst_data, transform=ccrs.PlateCarree(), levels=13, cmap='coolwarm', extend='both')
    
    # 添加颜色条
    cbar =fig.colorbar(other_contourf, ax=ax, orientation='vertical', pad=0.05, aspect=50, shrink=1, extend='both')

    # 添加海岸线和国界线
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')

     # 添加经纬度标签
    ax.set_xticks([-120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xticklabels(['120° W', '60° W', '0', '60° E', '120° E'])

    cbar.set_label('Corr', fontsize=15)
    # 设置图形标题
    ax.set_title(f'({number}) SST', loc='left')
    ax.set_ylim([0, 80])
    ax.set_xlim([-110, 120])


# In[72]:


def plot_cli_smroot_ax(data, name,ax):
    # 创建一个图形和子图
    smroot_data = data['slope']
    data_pv = data['pv']
    lon = data['lon']
    lat = data['lat']
    # 应用陆地掩膜：只保留陆地区域
    smroot_data = apply_land_mask(smroot_data)
    data_pv  = apply_land_mask(data_pv)
    # 绘制气象要素填色
    # 设置填色图的数据范围
    other_contourf = ax.contourf(lon, lat, smroot_data, transform=ccrs.PlateCarree(), vmin=-0.36,vmax=0.36,levels=13, cmap='BrBG',
                                 extend='both')

    # 添加颜色条
    cbar = fig.colorbar(other_contourf, ax=ax, orientation='vertical',  pad=0.05, aspect=50, shrink=1,  extend='both')

    # 添加海岸线和国界线
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=2)
    ax.add_feature(cfeature.COASTLINE, linestyle='-', edgecolor='darkgrey')
    ax.add_feature(cfeature.OCEAN)
    # 读取 shapefile
    shapefile_path = '/Users/limenghan/Desktop/地图/次国家/gadm41_USA_shp/gadm41_USA_1.shp'
    shp = gpd.read_file(shapefile_path)
    shp.plot(ax=ax, color='none', edgecolor='grey')  # 绘制 shapefile

    # 添加经纬度标签
    ax.set_xticks([-120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xticklabels(['120° W', '60° W', '0', '60° E', '120° E'])

    # 根据 other_data_pv 值添加斜线填充区域
    mask = data_pv < 0.1
    lons, lats = np.meshgrid(lon, lat)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j]:
                ax.fill_between([lons[i, j], lons[i, j] + 1], lats[i, j], lats[i, j] + 1,
                                color='none', edgecolor='grey', hatch='///', linewidth=0.1, alpha=0.75)
    cbar.set_label('σ / σ', fontsize=15)
    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])
    # 设置图形标题
    ax.set_title('(e) SMroot', loc='left')


# In[73]:


def plot_cli_vpd_ax(data, name,ax):
    # 创建一个图形和子图
    smroot_data = data['slope']
    data_pv = data['pv']
    lon = data['lon']
    lat = data['lat']

    # 绘制气象要素填色
    # 设置填色图的数据范围 
    other_contourf = ax.contourf(lon, lat, smroot_data, transform=ccrs.PlateCarree(), vmin=-0.3,vmax=0.3,levels=13, cmap='PiYG_r',
                                 extend='both')

    # 添加颜色条 
    cbar = fig.colorbar(other_contourf, ax=ax, orientation='vertical', pad=0.05, aspect=50, shrink=1,  extend='both')

    # 添加海岸线和国界线
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=2)
    ax.add_feature(cfeature.COASTLINE, linestyle='-', edgecolor='darkgrey')
    ax.add_feature(cfeature.OCEAN)
    # 读取 shapefile
    shapefile_path = '/Users/limenghan/Desktop/地图/次国家/gadm41_USA_shp/gadm41_USA_1.shp'
    shp = gpd.read_file(shapefile_path)
    shp.plot(ax=ax, color='none', edgecolor='grey')  # 绘制 shapefile

    # 添加经纬度标签
    ax.set_xticks([-120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xticklabels(['120° W', '60° W', '0', '60° E', '120° E'])

    # 根据 other_data_pv 值添加斜线填充区域
    mask = data_pv < 0.1
    lons, lats = np.meshgrid(lon, lat)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j]:
                ax.fill_between([lons[i, j], lons[i, j] + 1], lats[i, j], lats[i, j] + 1, 
                                color='none', edgecolor='grey', hatch='///', linewidth=0.1, alpha=0.75)
    cbar.set_label('σ / σ', fontsize=15)
    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])
    # 设置图形标题
    ax.set_title(f'(d) VPD', loc='left')


# In[74]:


##修改图
def plot_cli_other_2_ax(data, name, variable,ax):
    # 创建一个图形和子图
    other_data = data['slope']
    data_pv = data['pv']
    lon = data['lon']
    lat = data['lat']
    
    # 绘制气象要素填色
    # 设置填色图的数据范围
    if variable=='tmp':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, vmin=-0.25,vmax=0.25,cmap='coolwarm', extend='both')
    elif variable=='tmx' or variable=='tmxx':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, vmin=-0.4,vmax=0.4,cmap='coolwarm', extend='both')
    elif variable=='pre':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13,vmin=-0.2,vmax=0.2, cmap='PRGn', extend='both')
    elif variable=='evap':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, vmin=-0.3,vmax=0.3, cmap='PuOr',extend='both')
    elif variable=='snowfall' or variable=='snow_depth':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, cmap='PuOr', extend='both')
    elif variable=='msdwswrf':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, cmap='bwr', extend='both')
    elif variable=='cld':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, cmap='PuOr',vmin=-0.25,vmax=0.25, extend='both')
    elif variable=='dtr':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, cmap='PuOr',vmin=-0.4,vmax=0.4, extend='both')
    elif variable=='tmn':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, cmap='bwr',vmin=-0.4,vmax=0.4, extend='both')


    # 添加颜色条
    cbar = fig.colorbar(other_contourf, ax=ax, orientation='vertical', pad=0.05, aspect=50, shrink=1, extend='both')
    
    # 添加海岸线和国界线
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=2)
    ax.add_feature(cfeature.COASTLINE, linestyle='-', edgecolor='darkgrey')
    ax.add_feature(cfeature.OCEAN)
    # 读取 shapefile
    shapefile_path = '/Users/limenghan/Desktop/地图/次国家/gadm41_USA_shp/gadm41_USA_1.shp'
    shp = gpd.read_file(shapefile_path)
    shp.plot(ax=ax, color='none', edgecolor='grey')  # 绘制 shapefile
    
     # 添加经纬度标签
    ax.set_xticks([-120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xticklabels(['120° W', '60° W', '0', '60° E', '120° E'])

    # 根据 other_data_pv 值添加斜线填充区域
    mask = data_pv < 0.1
    lons, lats = np.meshgrid(lon, lat)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j]:
                ax.fill_between([lons[i, j], lons[i, j] + 1], lats[i, j], lats[i, j] + 1, color='none', edgecolor='grey', hatch='///', linewidth=0.1, alpha=0.75)

    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])
    cbar.set_label('σ / σ', fontsize=15)
    # 设置图形标题
    if variable=='tmx':
        ax.set_title(f'(c) Tmx',loc='left')
    elif variable=='pre':
        ax.set_title(f'(c) Pre',loc='left')
    elif variable=='tmxx':
        ax.set_title(f'(f) Tmx',loc='left')
    elif variable=='dtr':
        ax.set_title(f'(h) dtr',loc='left')
    elif variable=='tmn':
        ax.set_title(f'(g) Tmn',loc='left')


# In[97]:


##修改图
def plot_cli_other_2_ax(data, name, variable,ax):
    # 创建一个图形和子图
    other_data = data['slope']
    data_pv = data['pv']
    lon = data['lon']
    lat = data['lat']
    
    # 绘制气象要素填色
    # 设置填色图的数据范围
    if variable=='tmp':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, vmin=-0.25,vmax=0.25,cmap='coolwarm', extend='both')
    elif variable=='tmx' or variable=='tmxx':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, vmin=-0.4,vmax=0.4,cmap='coolwarm', extend='both')
    elif variable=='pre':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13,vmin=-0.2,vmax=0.2, cmap='PRGn', extend='both')
    elif variable=='evap':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, vmin=-0.3,vmax=0.3, cmap='PuOr',extend='both')
    elif variable=='snowfall' or variable=='snow_depth':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, cmap='PuOr', extend='both')
    elif variable=='msdwswrf':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, cmap='bwr', extend='both')
    elif variable=='cld':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, cmap='PuOr',vmin=-0.25,vmax=0.25, extend='both')
    elif variable=='dtr':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, cmap='RdYlBu_r',vmin=-0.4,vmax=0.4, extend='both')
    elif variable=='tmn':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, cmap='bwr',vmin=-0.4,vmax=0.4, extend='both')


    # 添加颜色条
    cbar = fig.colorbar(other_contourf, ax=ax, orientation='vertical', pad=0.05, aspect=50, shrink=1, extend='both')
    
    # 添加海岸线和国界线
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=2)
    ax.add_feature(cfeature.COASTLINE, linestyle='-', edgecolor='darkgrey')
    ax.add_feature(cfeature.OCEAN)
    # 读取 shapefile
    shapefile_path = '/Users/limenghan/Desktop/地图/次国家/gadm41_USA_shp/gadm41_USA_1.shp'
    shp = gpd.read_file(shapefile_path)
    shp.plot(ax=ax, color='none', edgecolor='grey')  # 绘制 shapefile
    
     # 添加经纬度标签
    ax.set_xticks([-120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xticklabels(['120° W', '60° W', '0', '60° E', '120° E'])
    ax.set_yticklabels(['90° S', '60° S', '30° S','0', '30° N', '60° N', '90° N'])
    # 根据 other_data_pv 值添加斜线填充区域
    mask = data_pv < 0.1
    lons, lats = np.meshgrid(lon, lat)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j]:
                ax.fill_between([lons[i, j], lons[i, j] + 1], lats[i, j], lats[i, j] + 1, color='none', edgecolor='grey', hatch='///', linewidth=0.1, alpha=0.75)
    cbar.set_label('σ / σ', fontsize=15)
    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])
    # 设置图形标题
    if variable=='tmx':
        ax.set_title(f'(c) Tmx',loc='left')
    elif variable=='pre':
        ax.set_title(f'(a) Pre',loc='left')
    elif variable=='tmxx':
        ax.set_title(f'(b) Tmx',loc='left')
    elif variable=='dtr':
        ax.set_title(f'(e) DTR',loc='left')
    elif variable=='tmn':
        ax.set_title(f'(g) Tmn',loc='left')


# In[98]:


def plot_cli_smroot_ax(data, name,ax):
    # 创建一个图形和子图
    smroot_data = data['slope']
    data_pv = data['pv']
    lon = data['lon']
    lat = data['lat']
    # 应用陆地掩膜：只保留陆地区域
    smroot_data = apply_land_mask(smroot_data)
    data_pv  = apply_land_mask(data_pv)
    # 绘制气象要素填色
    # 设置填色图的数据范围
    other_contourf = ax.contourf(lon, lat, smroot_data, transform=ccrs.PlateCarree(), vmin=-0.36,vmax=0.36,levels=13, cmap='BrBG',
                                 extend='both')

    # 添加颜色条
    cbar = fig.colorbar(other_contourf, ax=ax, orientation='vertical',  pad=0.05, aspect=50, shrink=1,  extend='both')

    # 添加海岸线和国界线
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=2)
    ax.add_feature(cfeature.COASTLINE, linestyle='-', edgecolor='darkgrey')
    ax.add_feature(cfeature.OCEAN)
    # 读取 shapefile
    shapefile_path = '/Users/limenghan/Desktop/地图/次国家/gadm41_USA_shp/gadm41_USA_1.shp'
    shp = gpd.read_file(shapefile_path)
    shp.plot(ax=ax, color='none', edgecolor='grey')  # 绘制 shapefile

    # 添加经纬度标签
    ax.set_xticks([-120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xticklabels(['120° W', '60° W', '0', '60° E', '120° E'])
    ax.set_yticklabels(['90° S', '60° S', '30° S','0', '30° N', '60° N', '90° N'])
    # 根据 other_data_pv 值添加斜线填充区域
    mask = data_pv < 0.1
    lons, lats = np.meshgrid(lon, lat)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j]:
                ax.fill_between([lons[i, j], lons[i, j] + 1], lats[i, j], lats[i, j] + 1,
                                color='none', edgecolor='grey', hatch='///', linewidth=0.1, alpha=0.75)
    cbar.set_label('σ / σ', fontsize=15)
    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])
    # 设置图形标题
    ax.set_title('(d) SMroot', loc='left')


# In[99]:


def plot_cli_vpd_ax(data, name,ax):
    # 创建一个图形和子图
    smroot_data = data['slope']
    data_pv = data['pv']
    lon = data['lon']
    lat = data['lat']

    # 绘制气象要素填色
    # 设置填色图的数据范围 
    other_contourf = ax.contourf(lon, lat, smroot_data, transform=ccrs.PlateCarree(), vmin=-0.3,vmax=0.3,levels=13, cmap='PiYG_r',
                                 extend='both')

    # 添加颜色条 
    cbar = fig.colorbar(other_contourf, ax=ax, orientation='vertical', pad=0.05, aspect=50, shrink=1,  extend='both')

    # 添加海岸线和国界线
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=2)
    ax.add_feature(cfeature.COASTLINE, linestyle='-', edgecolor='darkgrey')
    ax.add_feature(cfeature.OCEAN)
    # 读取 shapefile
    shapefile_path = '/Users/limenghan/Desktop/地图/次国家/gadm41_USA_shp/gadm41_USA_1.shp'
    shp = gpd.read_file(shapefile_path)
    shp.plot(ax=ax, color='none', edgecolor='grey')  # 绘制 shapefile

    # 添加经纬度标签
    ax.set_xticks([-120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xticklabels(['120° W', '60° W', '0', '60° E', '120° E'])
    ax.set_yticklabels(['90° S', '60° S', '30° S','0', '30° N', '60° N', '90° N'])
    # 根据 other_data_pv 值添加斜线填充区域
    mask = data_pv < 0.1
    lons, lats = np.meshgrid(lon, lat)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j]:
                ax.fill_between([lons[i, j], lons[i, j] + 1], lats[i, j], lats[i, j] + 1, 
                                color='none', edgecolor='grey', hatch='///', linewidth=0.1, alpha=0.75)
    cbar.set_label('σ / σ', fontsize=15)
    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])
    # 设置图形标题
    ax.set_title(f'(c) VPD', loc='left')


# In[100]:


from matplotlib import gridspec

# 创建画布和GridSpec布局
fig = plt.figure(figsize=(22,10))
# 设置第一行两个图的列宽一样，第二行的图在下方占满
gs = gridspec.GridSpec(2,3, width_ratios=[1, 1,1], wspace=0.1)


# 第二个图 (b)
ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
plot_cli_other_2_ax(reg_IOB_NDJ_1_pre_JAS,'reg_IOB_NDJ_1_pre_JAS','pre',ax)
# 第三个图 (c)
ax = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
plot_cli_other_2_ax(locals()[f'reg_IOB_NDJ_1_tmx_JAS'],f'reg_IOB_NDJ_1_tmx_JAS','tmxx',ax)
# 第四个图 (d)
ax = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
plot_cli_vpd_ax(locals()[f'reg_IOB_NDJ_1_vpd_JAS'],f'reg_IOB_NDJ_1_vpd_JAS',ax)
# 第五个图 (e)
ax = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
plot_cli_smroot_ax(locals()[f'reg_IOB_NDJ_1_smroot_JAS'],f'reg_IOB_NDJ_1_smroot_JAS',ax)
# 第四个图 (d)
ax = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
plot_cli_other_2_ax(locals()[f'reg_IOB_NDJ_1_dtr_JAS'],f'reg_IOB_NDJ_1_dtr_JAS','dtr',ax)
# 整体布局调整
plt.tight_layout()
plt.show()


# In[81]:



# 指定包含所有文件的文件夹路径
folder_path = '/Volumes/limenghan/气象要素/vertically_integrated_moisture_divergence/'
# 使用通配符 '*' 来匹配文件夹中所有的 nc 文件
files = folder_path + '*.nc'
# 使用 open_mfdataset 函数来打开所有匹配的文件，并将它们合并成一个 Dataset
vimd= xr.open_mfdataset(files, combine='nested', concat_dim='time')
vimd=vimd.rename({'longitude': 'lon'})
vimd=vimd.rename({'latitude': 'lat'})
# 使用 coarsen 进行降采样
if len(vimd.lon)>180:
    # 定义经度和纬度的降采样因子
    lon_factor, lat_factor = int(len(vimd.lat)/360), int(len(vimd.lon)/720)  # 对应的降采样因子
    # 使用 coarsen 方法进行降采样
    vimd=vimd.coarsen(lon=lon_factor, lat=lat_factor, boundary="trim").mean()
vimd=vimd['vimd']
vimd=time_sel(vimd,1978,2018)
vimd=detrend_weather(vimd)
vimd=detrend_weather(vimd)
vimd=time_sel(vimd,1979,2017)
if vimd.lon.max()>180:
    vimd=trans_lon(vimd)
vimd_ame=vimd.sel(lon=slice(-130, -65), lat=slice(55,20))


# In[82]:


reg_IOB_NDJ_1_vimd_NDJ=reg_location(IOB_NDJ_1,vimd_ame,'vimd',12)


# In[103]:


def plot_cli_smroot(data, name):
    # 创建一个图形和子图
    smroot_data = data['slope']
    data_pv = data['pv']
    lon = data['lon']
    lat = data['lat']
    # 应用陆地掩膜：只保留陆地区域
    smroot_data = apply_land_mask(smroot_data)
    data_pv  = apply_land_mask(data_pv)
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(20, 15))

    # 绘制气象要素填色
    # 设置填色图的数据范围
    other_contourf = ax.contourf(lon, lat, smroot_data, transform=ccrs.PlateCarree(), vmin=-0.6,vmax=0.6,levels=13, cmap='RdBu',
                                 extend='both')

    # 添加颜色条
    cbar = plt.colorbar(other_contourf, ax=ax, orientation='vertical', pad=0.05, aspect=50, shrink=0.6, extend='both')

    # 添加海岸线和国界线
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    # 读取 shapefile
    shapefile_path = '/Users/limenghan/Desktop/地图/次国家/gadm41_USA_shp/gadm41_USA_1.shp'
    shp = gpd.read_file(shapefile_path)
    shp.plot(ax=ax, color='none', edgecolor='grey')  # 绘制 shapefile

    # 添加经纬度标签
    ax.set_xticks([-120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xticklabels(['120° W', '60° W', '0', '60° E', '120° E'])
    ax.set_yticklabels(['90° S', '60° S', '30° S','0', '30° N', '60° N', '90° N'])
    # 根据 other_data_pv 值添加斜线填充区域
    mask = data_pv < 0.1
    lons, lats = np.meshgrid(lon, lat)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j]:
                ax.fill_between([lons[i, j], lons[i, j] + 1], lats[i, j], lats[i, j] + 1, 
                                color='none', edgecolor='grey', hatch='///', linewidth=0.1, alpha=0.75)
    cbar.set_label('σ / σ', fontsize=15)
    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])

    # 显示图形
    plt.show()


# In[101]:


def plot_cli_vimd(data, name):
    # 创建一个图形和子图
    smroot_data = data['slope']
    data_pv = data['pv']
    lon = data['lon']
    lat = data['lat']
    

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(20, 15))

    # 绘制气象要素填色
    # 设置填色图的数据范围
    other_contourf = ax.contourf(lon, lat, smroot_data, transform=ccrs.PlateCarree(), vmin=-0.36,vmax=0.36,levels=13, cmap='RdBu',
                                 extend='both')

    # 添加颜色条
    cbar = plt.colorbar(other_contourf, ax=ax, orientation='vertical', pad=0.05, aspect=50, shrink=0.6, extend='both')

    # 添加海岸线和国界线
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    # 读取 shapefile
    shapefile_path = '/Users/limenghan/Desktop/地图/次国家/gadm41_USA_shp/gadm41_USA_1.shp'
    shp = gpd.read_file(shapefile_path)
    shp.plot(ax=ax, color='none', edgecolor='grey')  # 绘制 shapefile

    # 添加经纬度标签
    ax.set_xticks([-120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xticklabels(['120° W', '60° W', '0', '60° E', '120° E'])
    ax.set_yticklabels(['90° S', '60° S', '30° S','0', '30° N', '60° N', '90° N'])
    # 根据 other_data_pv 值添加斜线填充区域
    mask = data_pv < 0.1
    lons, lats = np.meshgrid(lon, lat)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j]:
                ax.fill_between([lons[i, j], lons[i, j] + 1], lats[i, j], lats[i, j] + 1, 
                                color='none', edgecolor='grey', hatch='///', linewidth=0.1, alpha=0.75)
    cbar.set_label('σ / σ', fontsize=15)
    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])

    # 显示图形
    plt.show()


# In[102]:



plot_cli_vimd(reg_IOB_NDJ_1_vimd_NDJ,'reg_IOB_NDJ_1_vimd_NDJ')


# In[104]:


##修改图
def plot_cli_hgt_1_ax(data, name,uwnd_data,vwnd_data,ax):
    # 创建一个图形和子图
    hgt_data = data['slope']
    hgt_data_pv = data['pv']
    lon = data['lon']
    lat = data['lat']
    
    # 绘制气象要素填色
    # 设置填色图的数据范围
    other_contourf = ax.contourf(lon, lat, hgt_data, transform=ccrs.PlateCarree(), levels=13,vmin=-27,vmax=27, cmap='RdBu_r', extend='both')
    
    # 添加颜色条
    cbar = fig.colorbar(other_contourf, ax=ax, orientation='vertical',  pad=0.05, aspect=50, shrink=1,  extend='both')
    
    # 添加海岸线和国界线
    ax.add_feature(cfeature.BORDERS, linestyle='-',linewidth=0.5,  edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linestyle='-', linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    
    # 绘制风速箭头
    # 简化数据以提高可读性
    vwnd_data=vwnd_data['slope']
    uwnd_data=uwnd_data['slope']
    skip =2
    quiver = ax.quiver(uwnd_data['lon'][::skip].values, uwnd_data['lat'][::skip].values,
              uwnd_data.sel(lon=slice(None, None, skip), lat=slice(None, None, skip)).values, 
              vwnd_data.sel(lon=slice(None, None, skip), lat=slice(None, None, skip)).values,
              transform=ccrs.PlateCarree(), 
              scale=50, color='grey', width=0.0015)
    ax.quiverkey(quiver, 0.91, 1.03, 1, "1 m·s$^{-1}$",labelpos='E', coordinates='axes', fontproperties={ 'family':'Times New Roman'})
    # 添加经纬度标签
    ax.set_xticks([-120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xticklabels(['120° W', '60° W', '0', '60° E', '120° E'])
    ax.set_yticklabels(['90° S', '60° S', '30° S','0', '30° N', '60° N', '90° N'])
    cbar.set_label('m / σ', fontsize=15)
    # 设置图形标题
    ax.set_title('(b) GPH200*(-1)',loc='left')
    ax.set_ylim([-20, 80])
    ax.set_xlim([-110, 120])   


# In[105]:


def plot_cli_sst_ax(data, slp, name, number,ax):
    # 创建一个图形和子图
    data=tran_lon_360(data)
    slp=tran_lon_360(slp)
    sst_data = data['corr']
    sst_data_pv = data['pv']
    slp_data = slp['slope']
    lon = data['lon']
    lat = data['lat']    
    # 确定等高线级别
    levels = np.linspace(-100, 100, 20)  # 生成13个等高线级别，包括负值和正值

    # 绘制等高线
    contour = ax.contour(slp['lon'], slp['lat'], slp_data,
                         levels=levels, color='grey', linewidths=1, transform=ccrs.PlateCarree(central_longitude=0))
    # 分别设置实线和虚线
    for i, lvl in enumerate(contour.levels):
        if lvl < 0:
            contour.collections[i].set_linestyle('dashed')  # 负值用虚线
        else:
            contour.collections[i].set_linestyle('solid')   # 正值用实线
    # 添加等高线标签
    ax.clabel(contour, inline=True, fontsize=8, fmt='%1.0f')

    # 设置填色图的数据范围
    other_contourf = ax.contourf(lon, lat, sst_data, transform=ccrs.PlateCarree(), levels=13, cmap='coolwarm', extend='both')
    
    # 添加颜色条
    cbar =fig.colorbar(other_contourf, ax=ax, orientation='vertical', pad=0.05, aspect=50, shrink=1, extend='both')

    # 添加海岸线和国界线
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')

     # 添加经纬度标签
    ax.set_xticks([-120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xticklabels(['120° W', '60° W', '0', '60° E', '120° E'])
    ax.set_yticklabels(['90° S', '60° S', '30° S','0', '30° N', '60° N', '90° N'])
    cbar.set_label('Corr', fontsize=15)
    # 设置图形标题
    ax.set_title(f'({number}) SST', loc='left')
    ax.set_ylim([-20, 80])
    ax.set_xlim([-110, 120])


# In[88]:


reg_IOB_NDJ_1_slp_JAS


# In[106]:


from matplotlib import gridspec

# 创建画布和GridSpec布局
fig = plt.figure(figsize=(15,12))
# 设置第一行两个图的列宽一样，第二行的图在下方占满
gs = gridspec.GridSpec(2,1 , wspace=0.1)


# 第一个图 (a)
ax = fig.add_subplot(gs[1, :], projection=ccrs.PlateCarree(central_longitude=180))
plot_cli_hgt_1_ax(reg_nino34_JAS_1_hgt200_JAS*-1,'reg_nino34_JAS_1_hgt200_JAS',-reg_nino34_JAS_1_uwnd_JAS,-reg_nino34_JAS_1_vwnd_JAS,ax)


# 第一个图 (a)
ax = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree(central_longitude=180))
plot_cli_sst_ax(sst_IOB_NDJ_1_JAS,reg_IOB_NDJ_1_slp_JAS,'sst_IOB_NDJ_1_JAS','a',ax)

plt.tight_layout()
plt.savefig('/Users/limenghan/Desktop/大豆代码结果/论文图/Figure5.png',dpi=300)
plt.show()


# In[107]:


##修改图
def plot_cli_other_2_ax(data, name, variable,ax):
    # 创建一个图形和子图
    other_data = data['slope']
    data_pv = -data['pv']
    lon = data['lon']
    lat = data['lat']
    
    # 绘制气象要素填色
    # 设置填色图的数据范围
    if variable=='tmp':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, vmin=-0.25,vmax=0.25,cmap='coolwarm', extend='both')
    elif variable=='tmx' or variable=='tmxx':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, vmin=-0.4,vmax=0.4,cmap='coolwarm', extend='both')
    elif variable=='pre':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13,vmin=-0.32,vmax=0.32, cmap='PRGn', extend='both')
    elif variable=='evap':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, vmin=-0.3,vmax=0.3, cmap='PuOr',extend='both')
    elif variable=='snowfall' or variable=='snow_depth':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, cmap='PuOr', extend='both')
    elif variable=='msdwswrf':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, cmap='bwr', extend='both')
    elif variable=='cld':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, cmap='PuOr',vmin=-0.3,vmax=0.3, extend='both')
    elif variable=='dtr':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, cmap='RdYlBu_r',vmin=-0.4,vmax=0.4, extend='both')
    elif variable=='tmn':
        other_contourf = ax.contourf(lon, lat, other_data, transform=ccrs.PlateCarree(), levels=13, cmap='bwr',vmin=-0.4,vmax=0.4, extend='both')


    # 添加颜色条
    cbar = fig.colorbar(other_contourf, ax=ax, orientation='vertical', pad=0.05, aspect=50, shrink=1, extend='both')
    
    # 添加海岸线和国界线
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=2)
    ax.add_feature(cfeature.COASTLINE, linestyle='-', edgecolor='darkgrey')
    ax.add_feature(cfeature.OCEAN)
    # 读取 shapefile
    shapefile_path = '/Users/limenghan/Desktop/地图/次国家/gadm41_USA_shp/gadm41_USA_1.shp'
    shp = gpd.read_file(shapefile_path)
    shp.plot(ax=ax, color='none', edgecolor='grey')  # 绘制 shapefile
    
     # 添加经纬度标签
    ax.set_xticks([-120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xticklabels(['120° W', '60° W', '0', '60° E', '120° E'])
    ax.set_yticklabels(['90° S', '60° S', '30° S','0', '30° N', '60° N', '90° N'])
    # 根据 other_data_pv 值添加斜线填充区域
    mask = data_pv < 0.1
    lons, lats = np.meshgrid(lon, lat)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j]:
                ax.fill_between([lons[i, j], lons[i, j] + 1], lats[i, j], lats[i, j] + 1, color='none', edgecolor='grey', hatch='///', linewidth=0.1, alpha=0.75)
    cbar.set_label('σ / σ', fontsize=15)
    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])
    # 设置图形标题
    if variable=='tmx':
        ax.set_title(f'(c) Tmx*(-1)',loc='left')
    elif variable=='pre':
        ax.set_title(f'(a) Pre*(-1)',loc='left')
    elif variable=='tmxx':
        ax.set_title(f'(b) Tmx*(-1)',loc='left')
    elif variable=='dtr':
        ax.set_title(f'(f) DTR*(-1)',loc='left')
    elif variable=='tmn':
        ax.set_title(f'(g) Tmn*(-1)',loc='left')
    elif variable=='cld':
        ax.set_title(f'(e) Cld*(-1)',loc='left')


# In[108]:


def plot_cli_smroot_ax(data, name,ax):
    # 创建一个图形和子图
    smroot_data = data['slope']
    data_pv = -data['pv']
    lon = data['lon']
    lat = data['lat']
    # 应用陆地掩膜：只保留陆地区域
    smroot_data = apply_land_mask(smroot_data)
    data_pv  = apply_land_mask(data_pv)
    # 绘制气象要素填色
    # 设置填色图的数据范围
    other_contourf = ax.contourf(lon, lat, smroot_data, transform=ccrs.PlateCarree(), vmin=-0.36,vmax=0.36,levels=13, cmap='BrBG',
                                 extend='both')

    # 添加颜色条
    cbar = fig.colorbar(other_contourf, ax=ax, orientation='vertical',  pad=0.05, aspect=50, shrink=1,  extend='both')

    # 添加海岸线和国界线
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=2)
    ax.add_feature(cfeature.COASTLINE, linestyle='-', edgecolor='darkgrey')
    ax.add_feature(cfeature.OCEAN)
    # 读取 shapefile
    shapefile_path = '/Users/limenghan/Desktop/地图/次国家/gadm41_USA_shp/gadm41_USA_1.shp'
    shp = gpd.read_file(shapefile_path)
    shp.plot(ax=ax, color='none', edgecolor='grey')  # 绘制 shapefile

    # 添加经纬度标签
    ax.set_xticks([-120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xticklabels(['120° W', '60° W', '0', '60° E', '120° E'])
    ax.set_yticklabels(['90° S', '60° S', '30° S','0', '30° N', '60° N', '90° N'])
    # 根据 other_data_pv 值添加斜线填充区域
    mask = data_pv < 0.1
    lons, lats = np.meshgrid(lon, lat)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j]:
                ax.fill_between([lons[i, j], lons[i, j] + 1], lats[i, j], lats[i, j] + 1,
                                color='none', edgecolor='grey', hatch='///', linewidth=0.1, alpha=0.75)
    cbar.set_label('σ / σ', fontsize=15)
    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])
    # 设置图形标题
    ax.set_title('(d) SMroot*(-1)', loc='left')


# In[109]:


def plot_cli_vpd_ax(data, name,ax):
    # 创建一个图形和子图
    smroot_data = data['slope']
    data_pv = -data['pv']
    lon = data['lon']
    lat = data['lat']

    # 绘制气象要素填色
    # 设置填色图的数据范围 
    other_contourf = ax.contourf(lon, lat, smroot_data, transform=ccrs.PlateCarree(), vmin=-0.3,vmax=0.3,levels=13, cmap='PiYG_r',
                                 extend='both')

    # 添加颜色条 
    cbar = fig.colorbar(other_contourf, ax=ax, orientation='vertical', pad=0.05, aspect=50, shrink=1,  extend='both')

    # 添加海岸线和国界线
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=2)
    ax.add_feature(cfeature.COASTLINE, linestyle='-', edgecolor='darkgrey')
    ax.add_feature(cfeature.OCEAN)
    # 读取 shapefile
    shapefile_path = '/Users/limenghan/Desktop/地图/次国家/gadm41_USA_shp/gadm41_USA_1.shp'
    shp = gpd.read_file(shapefile_path)
    shp.plot(ax=ax, color='none', edgecolor='grey')  # 绘制 shapefile

    # 添加经纬度标签
    ax.set_xticks([-120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xticklabels(['120° W', '60° W', '0', '60° E', '120° E'])
    ax.set_yticklabels(['90° S', '60° S', '30° S','0', '30° N', '60° N', '90° N'])
    # 根据 other_data_pv 值添加斜线填充区域
    mask = data_pv < 0.1
    lons, lats = np.meshgrid(lon, lat)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j]:
                ax.fill_between([lons[i, j], lons[i, j] + 1], lats[i, j], lats[i, j] + 1, 
                                color='none', edgecolor='grey', hatch='///', linewidth=0.1, alpha=0.75)
    cbar.set_label('σ / σ', fontsize=15)
    ax.set_xlim([-130, -65])
    ax.set_ylim([20, 55])
    # 设置图形标题
    ax.set_title(f'(c) VPD*(-1)', loc='left')


# In[110]:


from matplotlib import gridspec

# 创建画布和GridSpec布局
fig = plt.figure(figsize=(22,10))
# 设置第一行两个图的列宽一样，第二行的图在下方占满
gs = gridspec.GridSpec(2,3, width_ratios=[1, 1,1], wspace=0.1)


# 第二个图 (b)
ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
plot_cli_other_2_ax(-reg_nino34_JAS_1_pre_JAS,'reg_nino34_JAS_1_pre_JAS','pre',ax)
# 第三个图 (c)
ax = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
plot_cli_other_2_ax(-locals()[f'reg_nino34_JAS_1_tmx_JAS'],f'reg_nino34_JAS_1_tmx_JAS','tmxx',ax)
# 第四个图 (d)
ax = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
plot_cli_vpd_ax(-locals()[f'reg_nino34_JAS_1_vpd_JAS'],f'reg_nino34_JAS_1_vpd_JAS',ax)
# 第五个图 (e)
ax = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
plot_cli_smroot_ax(-locals()[f'reg_nino34_JAS_1_smroot_JAS'],f'reg_nino34_JAS_1_smroot_JAS',ax)
# 第四个图 (d)
ax = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
plot_cli_other_2_ax(-locals()[f'reg_nino34_JAS_1_cld_JAS'],f'reg_nino34_JAS_1_cld_JAS','cld',ax)
# 第四个图 (d)
ax = fig.add_subplot(gs[1, 2], projection=ccrs.PlateCarree())
plot_cli_other_2_ax(-locals()[f'reg_nino34_JAS_1_dtr_JAS'],f'reg_nino34_JAS_1_dtr_JAS','dtr',ax)
# 整体布局调整
plt.tight_layout()
plt.savefig('/Users/limenghan/Desktop/大豆代码结果/论文图/Figure6.png',dpi=300)
plt.show()


# In[ ]:





# In[ ]:




