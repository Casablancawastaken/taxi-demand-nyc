import hashlib
import os
import numpy as np
import pandas as pd
from .settings import FEATURES
from .db import quality, data_mode


def load_panel(conn, max_zones=None):
    rows = pd.read_sql_query('SELECT date,zone_id,trips FROM trips ORDER BY date,zone_id',conn,parse_dates=['date'])
    if rows.empty:
        raise RuntimeError('Нет поездок. Сначала выполни demo или sync.')
    zones = pd.read_sql_query('SELECT zone_id,borough,zone FROM zones',conn)
    if zones.empty:
        raise RuntimeError('Отсутствует справочник районов')
    dates_for_selection = sorted(rows.date.unique())
    cutoff = dates_for_selection[max(0, int(len(dates_for_selection)*0.60)-1)]
    available = rows[rows.date<=cutoff].groupby('zone_id')['trips'].sum().sort_values(ascending=False)
    if max_zones is None:
        max_zones = int(os.environ.get('TAXI_MAX_ZONES','30'))
    selected = available.head(max_zones).index if max_zones > 0 else available.index
    rows = rows.loc[rows.zone_id.isin(selected)].copy()
    dates = pd.date_range(rows.date.min(),rows.date.max(),freq='D')
    grid = pd.MultiIndex.from_product([selected,dates],names=['zone_id','date']).to_frame(index=False)
    panel = grid.merge(rows,on=['zone_id','date'],how='left').merge(zones,on='zone_id',how='left')
    panel['trips'] = panel['trips'].fillna(0).astype(float)
    panel['borough'] = panel['borough'].fillna('Unknown')
    weather = pd.read_sql_query('SELECT date,temperature,precipitation,wind FROM weather ORDER BY date',conn,parse_dates=['date'])
    panel = panel.merge(weather,on='date',how='left')
    for key in ['temperature','precipitation','wind']:
        panel[key] = pd.to_numeric(panel[key],errors='coerce')
        if key in ['precipitation','wind']:
            panel.loc[panel[key]<0,key] = np.nan
        panel[key] = panel.groupby('zone_id',sort=False)[key].ffill()
    return panel.sort_values(['zone_id','date']).reset_index(drop=True)


def build_features(panel):
    frame = panel.sort_values(['zone_id','date']).copy().reset_index(drop=True)
    group = frame.groupby('zone_id',sort=False)['trips']
    frame['dow'] = frame['date'].dt.dayofweek
    frame['month'] = frame['date'].dt.month
    frame['day_of_month'] = frame['date'].dt.day
    frame['is_weekend'] = (frame['dow']>=5).astype(int)
    frame['dow_sin'] = np.sin(frame['dow']*2*np.pi/7)
    frame['dow_cos'] = np.cos(frame['dow']*2*np.pi/7)
    frame['month_sin'] = np.sin((frame['month']-1)*2*np.pi/12)
    frame['month_cos'] = np.cos((frame['month']-1)*2*np.pi/12)
    for lag in [1,7,14,28]:
        frame[f'lag_{lag}'] = group.shift(lag)
    for window in [7,28]:
        frame[f'rolling_mean_{window}'] = group.transform(lambda s: s.shift(1).rolling(window,min_periods=window).mean())
    frame['rolling_std_7'] = group.transform(lambda s:s.shift(1).rolling(7,min_periods=7).std())
    frame['rolling_min_7'] = group.transform(lambda s:s.shift(1).rolling(7,min_periods=7).min())
    for src,dest in [('temperature','temperature_lag1'),('precipitation','precipitation_lag1'),('wind','wind_lag1')]:
        frame[dest] = frame.groupby('zone_id',sort=False)[src].shift(1)
    return frame


def training_frame(panel):
    result=build_features(panel)
    result=result.dropna(subset=['lag_28','rolling_mean_28','lag_7']).copy()
    result['zone_id'] = result['zone_id'].astype(str)
    return result.sort_values(['date','zone_id']).reset_index(drop=True)


def signature(frame):
    hashed = pd.util.hash_pandas_object(frame[['date','zone_id','trips','temperature_lag1','precipitation_lag1','wind_lag1']],index=False)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()


def validate_data(conn, panel):
    days=panel['date'].nunique()
    zones=panel['zone_id'].nunique()
    missing_weather=panel['temperature'].isna().sum()
    expected = days*zones
    checks = [
        ('Полная сетка район-дата',len(panel)==expected,f'{len(panel)} наблюдений, ожидалось {expected}'),
        ('Неотрицательный спрос',(panel['trips']>=0).all(),'Отрицательные значения числа поездок запрещены'),
        ('Наличие погоды',missing_weather==0,f'Нет погоды у {missing_weather} строк'),
        ('Исторические данные',days>=110,f'Дней: {days} (минимум 110)'),
        ('Уникальность ключей',not panel.duplicated(['date','zone_id']).any(),'Ключ: date + zone_id'),
    ]
    flags=0
    for _,grp in panel.groupby('zone_id'):
        median=grp.trips.median()
        flags += int((grp.trips > max(1,median)*5).sum())
    checks.append(('Контроль выбросов',flags==0,f'Аномально высоких дневных значений >5 медиан: {flags}; не удаляются автоматически'))
    if data_mode(conn)=='real':
        months={str(x) for x in pd.period_range(panel.date.min(),panel.date.max(),freq='M')}
        loaded={row[0].removeprefix('yellow_tripdata_').removesuffix('.parquet')
                for row in conn.execute("SELECT source_key FROM ingestion_log WHERE source='tlc' AND status='SUCCESS'")}
        missing=sorted(months-loaded)
        checks.append(('Непрерывность месячных источников TLC',not missing,
                       'Отсутствующие месяцы: '+(', '.join(missing) if missing else 'нет')))
    for name,passed,detail in checks:
        quality(conn,name,bool(passed),detail)
    hard_failed = [name for name,passed,_ in checks if not passed and name!='Контроль выбросов']
    if hard_failed:
        raise ValueError('Проверки качества не пройдены: '+', '.join(hard_failed))
    return checks
