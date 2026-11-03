import calendar
import csv
import io
import json
import math
import random
import time
import zipfile
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
import requests
import pandas as pd
from .settings import DATA, SEED
from .db import ingestion, data_mode

TRIP_URL = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet'
ZONE_CSV_URL = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv'
ZONE_GEO_URL = 'https://data.cityofnewyork.us/resource/8meu-9t5y.geojson?$limit=500'
ZONE_ZIP_URL = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip'
WEATHER_URL = 'https://archive-api.open-meteo.com/v1/archive'
DEMO_ZONES = [
    (48, 'Manhattan', 'Clinton East', -73.992,40.758),
    (79, 'Manhattan', 'East Village', -73.984,40.727),
    (100, 'Manhattan', 'Garment District', -73.989,40.752),
    (107, 'Manhattan', 'Gramercy', -73.983,40.738),
    (132, 'Queens', 'JFK Airport', -73.780,40.645),
    (138, 'Queens', 'LaGuardia Airport', -73.874,40.774),
    (142, 'Manhattan', 'Lincoln Square East', -73.981,40.774),
    (161, 'Manhattan', 'Midtown Center', -73.977,40.754),
    (170, 'Manhattan', 'Murray Hill', -73.977,40.748),
    (186, 'Manhattan', 'Penn Station/Madison Sq West', -73.994,40.748),
    (230, 'Manhattan', 'Times Sq/Theatre District', -73.985,40.760),
    (236, 'Manhattan', 'Upper East Side North', -73.956,40.775),
]

def request_file(url, path, *, params=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 100:
        return path
    last_error = None
    for attempt in range(3):
        try:
            with requests.get(url, params=params, stream=True, timeout=(15, 120), headers={'User-Agent':'TaxiDemandUniversityProject/1.0'}) as response:
                response.raise_for_status()
                temporary = path.with_suffix(path.suffix + '.part')
                with temporary.open('wb') as target:
                    for chunk in response.iter_content(chunk_size=1024 * 512):
                        if chunk:
                            target.write(chunk)
                if temporary.stat().st_size == 0:
                    raise RuntimeError('Пустой ответ сервера')
                temporary.replace(path)
                return path
        except (OSError, requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1 + attempt)
    raise RuntimeError(f'Источник недоступен: {url}; {last_error}')

def month_range(start, count):
    year, month = map(int, start.split('-'))
    for i in range(count):
        current = year * 12 + month - 1 + i
        yield f'{current // 12:04d}-{current % 12 + 1:02d}'

def _geo_lookup(geo):
    output = {}
    for feature in geo.get('features', []):
        props = {str(k).lower().replace('_',''): v for k,v in feature.get('properties', {}).items()}
        raw_id = next((props[k] for k in ['locationid','objectid'] if k in props and props[k]), None)
        if raw_id is None:
            continue
        try:
            zone_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        geom = feature.get('geometry')
        if geom and geom.get('type') in ['Polygon','MultiPolygon']:
            output[zone_id] = json.dumps(geom, ensure_ascii=False)
    return output

def _shapefile_geometries(path):
    import shapefile
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        prefix = next(x[:-4] for x in names if x.lower().endswith('.shp'))
        with archive.open(prefix+'.shp') as shp, archive.open(prefix+'.shx') as shx, archive.open(prefix+'.dbf') as dbf:
            reader = shapefile.Reader(shp=shp, shx=shx, dbf=dbf)
            columns = [x[0].lower() for x in reader.fields[1:]]
            result = {}
            for rec in reader.iterShapeRecords():
                properties = dict(zip(columns, rec.record))
                zone_id = properties.get('locationid')
                if zone_id is not None:
                    result[int(zone_id)] = json.dumps(rec.shape.__geo_interface__,ensure_ascii=False)
            return result

def sync_zones_real(conn):
    if conn.execute('SELECT count(*) FROM zones WHERE data_mode=?', ('real',)).fetchone()[0] > 0:
        ingestion(conn,'taxi_zones','NYC GeoJSON','real','SKIPPED',0,'Области уже загружены')
        return
    csv_path = request_file(ZONE_CSV_URL,DATA/'raw'/'taxi_zone_lookup.csv')
    geom = {}
    try:
        geo_path = request_file(ZONE_GEO_URL,DATA/'raw'/'taxi_zones.geojson')
        geom = _geo_lookup(json.loads(geo_path.read_text(encoding='utf-8')))
        if len(geom) < 100:
            raise ValueError('GeoJSON не содержит ожидаемое число районов')
    except (RuntimeError, ValueError, KeyError, json.JSONDecodeError):
        zip_path = request_file(ZONE_ZIP_URL, DATA/'raw'/'taxi_zones.zip')
        geom = _shapefile_geometries(zip_path)
    if len(geom) < 100:
        raise RuntimeError('Не удалось разобрать достаточное число геозон')
    with csv_path.open(encoding='utf-8-sig', newline='') as source:
        rows = list(csv.DictReader(source))
    inserted = 0
    for row in rows:
        zone_id = int(row['LocationID'])
        if zone_id in geom:
            conn.execute('INSERT OR REPLACE INTO zones VALUES(?,?,?,?,?)',
                         (zone_id,row['Borough'],row['Zone'],geom[zone_id],'real'))
            inserted += 1
    conn.commit()
    ingestion(conn,'taxi_zones','NYC TLC GeoJSON/Shapefile + lookup','real','SUCCESS',inserted,'Неструктурированные геометрии Polygon/MultiPolygon')

def aggregate_tlc_parquet(path, month):
    import pyarrow.parquet as pq
    source = pq.ParquetFile(path)
    month_start = pd.Timestamp(month+'-01')
    month_end = month_start + pd.offsets.MonthBegin(1)
    counts = Counter()
    total = invalid = 0
    for batch in source.iter_batches(columns=['tpep_pickup_datetime','PULocationID'],batch_size=150000):
        frame = batch.to_pandas()
        total += len(frame)
        timestamps = pd.to_datetime(frame['tpep_pickup_datetime'],errors='coerce')
        ids = pd.to_numeric(frame['PULocationID'],errors='coerce')
        valid = (timestamps >= month_start) & (timestamps < month_end) & ids.between(1,263)
        invalid += int((~valid).sum())
        selected = pd.DataFrame({'date':timestamps[valid].dt.strftime('%Y-%m-%d'), 'zone_id':ids[valid].astype(int)})
        counts.update({(str(d),int(z)):int(n) for (d,z),n in selected.value_counts().items()})
    return counts,total,invalid

def sync_trips_real(conn, start_month, months):
    for month in month_range(start_month, months):
        key = f'yellow_tripdata_{month}.parquet'
        if conn.execute('SELECT 1 FROM ingestion_log WHERE source=? AND source_key=? AND status=? LIMIT 1',('tlc',key,'SUCCESS')).fetchone():
            ingestion(conn,'tlc',key,'real','SKIPPED',0,'Месяц уже обработан, дубли отсутствуют')
            continue
        try:
            path = request_file(TRIP_URL.format(month=month),DATA/'raw'/'trips'/key)
            counts,total,invalid = aggregate_tlc_parquet(path,month)
            for (day,z),number in counts.items():
                conn.execute('INSERT INTO trips(date,zone_id,trips,source) VALUES(?,?,?,?) ON CONFLICT(date,zone_id) DO UPDATE SET trips=excluded.trips,source=excluded.source',
                             (day,z,number,key))
            conn.commit()
            ingestion(conn,'tlc',key,'real','SUCCESS',total,f'Неверных/вне периода: {invalid}; агрегированных групп: {len(counts)}')
        except Exception as exc:
            conn.rollback()
            ingestion(conn,'tlc',key,'real','FAILED',0,str(exc))
            raise

def sync_weather_real(conn):
    limits = conn.execute('SELECT MIN(date), MAX(date) FROM trips').fetchone()
    if not limits[0]:
        raise RuntimeError('Сначала нужны поездки TLC')
    all_days = pd.date_range(limits[0], limits[1], freq='D')
    present = {row[0] for row in conn.execute('SELECT date FROM weather')}
    missing = [day.strftime('%Y-%m-%d') for day in all_days if day.strftime('%Y-%m-%d') not in present]
    if not missing:
        ingestion(conn,'open_meteo','archive','real','SKIPPED',0,'Погода за период уже есть')
        return
    start, end = min(missing),max(missing)
    cache = DATA/'raw'/'weather'/f'weather_{start}_{end}.json'
    params = {'latitude':40.7128, 'longitude':-74.0060, 'start_date':start, 'end_date':end,
              'daily':'temperature_2m_mean,precipitation_sum,wind_speed_10m_max',
              'timezone':'America/New_York'}
    try:
        path = request_file(WEATHER_URL,cache,params=params)
        raw = json.loads(path.read_text(encoding='utf-8'))
        daily = raw['daily']
        count = 0
        for i,day in enumerate(daily['time']):
            if day not in missing:
                continue
            conn.execute('INSERT OR IGNORE INTO weather(date,temperature,precipitation,wind,source) VALUES(?,?,?,?,?)',
                         (day,daily['temperature_2m_mean'][i],daily['precipitation_sum'][i],daily['wind_speed_10m_max'][i], 'open_meteo'))
            count += 1
        conn.commit()
        ingestion(conn,'open_meteo',f'{start}:{end}','real','SUCCESS',count,'JSON ежедневных наблюдений; используется лаг 1 день')
    except Exception as exc:
        conn.rollback()
        ingestion(conn,'open_meteo',f'{start}:{end}','real','FAILED',0,str(exc))
        raise

def sync_real(conn, start_month='2024-01', months=6):
    if data_mode(conn) == 'demo':
        raise RuntimeError('Демо и реальные данные нельзя смешивать. Запусти reset перед переключением режима.')
    try:
        sync_zones_real(conn)
    except Exception as exc:
        ingestion(conn,'taxi_zones','NYC GeoJSON','real','FAILED',0,str(exc))
        raise
    sync_trips_real(conn,start_month,months)
    sync_weather_real(conn)
    return {'mode':'real','months':months,'trip_days':conn.execute('SELECT count(distinct date) FROM trips').fetchone()[0]}

def square_geo(x,y,w=.009,h=.007):
    return {'type':'Polygon','coordinates':[[[x-w/2,y-h/2],[x+w/2,y-h/2],[x+w/2,y+h/2],[x-w/2,y+h/2],[x-w/2,y-h/2]]]}

def generate_demo(conn, end_date='2026-11-02', days=240, reset=False):
    if reset:
        from .db import reset_data
        reset_data(conn)
    if data_mode(conn) == 'real':
        raise RuntimeError('Реальные данные нельзя смешивать с синтетическими. Используй reset.')
    key = f'{end_date}/{days}/{SEED}'
    if conn.execute('SELECT 1 FROM ingestion_log WHERE source=? AND source_key=? AND status=?',('demo_trips',key,'SUCCESS')).fetchone():
        ingestion(conn,'demo_trips',key,'demo','SKIPPED',0,'Идемпотентное выполнение')
        return
    last_day = date.fromisoformat(end_date)
    first_day = last_day-timedelta(days=days-1)
    rng = random.Random(SEED)
    zone_features = []
    geo_features = []
    for zone_id,borough,name,x,y in DEMO_ZONES:
        geom = square_geo(x,y)
        zone_features.append((zone_id,borough,name,json.dumps(geom,ensure_ascii=False),'demo'))
        geo_features.append({'type':'Feature','properties':{'location_id':zone_id,'borough':borough,'zone':name},'geometry':geom})
    (DATA/'raw'/'demo_zones.geojson').write_text(json.dumps({'type':'FeatureCollection','features':geo_features},ensure_ascii=False,indent=2),encoding='utf-8')
    weather_json = {'source':'synthetic','daily':{'time':[],'temperature_2m_mean':[], 'precipitation_sum':[],'wind_speed_10m_max':[]}}
    weather_rows=[]
    trip_rows=[]
    for day_index in range(days):
        dt=first_day+timedelta(days=day_index)
        season = math.sin(day_index*2*math.pi/120)
        temp=round(13 + 11*season+rng.gauss(0,3.5),2)
        rain=round(max(0,rng.gauss(2.4,4.6)),2)
        wind=round(max(0,rng.gauss(12,4)),2)
        weather_json['daily']['time'].append(dt.isoformat())
        weather_json['daily']['temperature_2m_mean'].append(temp)
        weather_json['daily']['precipitation_sum'].append(rain)
        weather_json['daily']['wind_speed_10m_max'].append(wind)
        weather_rows.append((dt.isoformat(),temp,rain,wind,'synthetic'))
        for k,(zone_id,borough,name,x,y) in enumerate(DEMO_ZONES):
            base=1100 + 95*k + (450 if borough == 'Manhattan' else -150)
            weekend=0.77 if dt.weekday() >= 5 else 1.0
            if 'Airport' in name:
                weekend=1.15 if dt.weekday() >= 5 else 1.0
            trend=1.0 + day_index*0.00065
            weekly=1.0+0.075*math.sin(dt.weekday()*2*math.pi/7)
            previous_weather=weather_rows[max(0,day_index-1)]
            rain_effect=1 + 0.012*min(previous_weather[2],20)
            expected=base*weekend*trend*weekly*rain_effect*(1+0.055*season)
            count=max(0,round(expected+rng.gauss(0,base*0.065)))
            trip_rows.append((dt.isoformat(),zone_id,count,'synthetic'))
    (DATA/'raw'/'demo_weather.json').write_text(json.dumps(weather_json,ensure_ascii=False,indent=2),encoding='utf-8')
    with (DATA/'raw'/'demo_trips.csv').open('w',encoding='utf-8',newline='') as output:
        writer=csv.writer(output)
        writer.writerow(['date','zone_id','trips'])
        writer.writerows([t[:3] for t in trip_rows])
    conn.executemany('INSERT OR REPLACE INTO zones VALUES(?,?,?,?,?)',zone_features)
    conn.executemany('INSERT OR REPLACE INTO weather VALUES(?,?,?,?,?)',weather_rows)
    conn.executemany('INSERT OR REPLACE INTO trips VALUES(?,?,?,?)',trip_rows)
    conn.commit()
    ingestion(conn,'demo_geojson',key,'demo','SUCCESS',len(zone_features),'Синтетические полигональные геоданные')
    ingestion(conn,'demo_weather',key,'demo','SUCCESS',len(weather_rows),'Синтетический JSON погоды')
    ingestion(conn,'demo_trips',key,'demo','SUCCESS',len(trip_rows),'Синтетическая агрегированная статистика поездок')
