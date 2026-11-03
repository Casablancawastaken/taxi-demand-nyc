import json
import sqlite3
from datetime import datetime, timezone
from .settings import DATABASE, ensure_paths

SCHEMA = '''
CREATE TABLE IF NOT EXISTS zones(
 zone_id INTEGER PRIMARY KEY, borough TEXT NOT NULL, zone TEXT NOT NULL,
 geometry_json TEXT NOT NULL, data_mode TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS trips(
 date TEXT NOT NULL, zone_id INTEGER NOT NULL, trips INTEGER NOT NULL CHECK(trips >= 0),
 source TEXT NOT NULL, PRIMARY KEY(date, zone_id));
CREATE TABLE IF NOT EXISTS weather(
 date TEXT PRIMARY KEY, temperature REAL, precipitation REAL, wind REAL,
 source TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ingestion_log(
 id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, source TEXT NOT NULL,
 source_key TEXT NOT NULL, mode TEXT NOT NULL, status TEXT NOT NULL,
 rows_processed INTEGER NOT NULL, detail TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS quality_log(
 id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
 test_name TEXT NOT NULL, status TEXT NOT NULL, detail TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiments(
 id TEXT PRIMARY KEY, created_at TEXT NOT NULL, version TEXT NOT NULL,
 data_signature TEXT NOT NULL, model TEXT NOT NULL, parameters TEXT NOT NULL,
 seed INTEGER NOT NULL, fold TEXT NOT NULL, mae REAL NOT NULL, rmse REAL NOT NULL,
 wape REAL NOT NULL);
CREATE TABLE IF NOT EXISTS models(
 version TEXT PRIMARY KEY, created_at TEXT NOT NULL, selected_model TEXT NOT NULL,
 metrics_json TEXT NOT NULL, features_json TEXT NOT NULL, params_json TEXT NOT NULL,
 data_signature TEXT NOT NULL, artifact TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS backtest(
 version TEXT NOT NULL, date TEXT NOT NULL, zone_id INTEGER NOT NULL,
 actual REAL NOT NULL, predicted REAL NOT NULL,
 PRIMARY KEY(version, date, zone_id));
CREATE TABLE IF NOT EXISTS forecasts(
 version TEXT NOT NULL, origin TEXT NOT NULL, date TEXT NOT NULL,
 zone_id INTEGER NOT NULL, predicted REAL NOT NULL,
 PRIMARY KEY(version, origin, date, zone_id));
CREATE TABLE IF NOT EXISTS feature_importance(
 version TEXT NOT NULL, method TEXT NOT NULL, period TEXT NOT NULL,
 feature TEXT NOT NULL, importance REAL NOT NULL,
 PRIMARY KEY(version, method, period, feature));
CREATE TABLE IF NOT EXISTS monitoring(
 id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, version TEXT NOT NULL,
 simulated INTEGER NOT NULL, last_window_wape REAL, psi REAL NOT NULL,
 drift_detected INTEGER NOT NULL, retrain_recommended INTEGER NOT NULL,
 detail TEXT NOT NULL);
'''

def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')

def connect():
    ensure_paths()
    conn = sqlite3.connect(DATABASE, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys=ON')
    conn.execute('PRAGMA journal_mode=WAL')
    conn.executescript(SCHEMA)
    return conn

def ingestion(conn, source, key, mode, status, count, detail=''):
    conn.execute('INSERT INTO ingestion_log(created_at,source,source_key,mode,status,rows_processed,detail) VALUES(?,?,?,?,?,?,?)',
                 (now(), source, key, mode, status, int(count), str(detail)[:1000]))
    conn.commit()

def quality(conn, name, passed, detail):
    conn.execute('INSERT INTO quality_log(created_at,test_name,status,detail) VALUES(?,?,?,?)',
                 (now(), name, 'PASS' if passed else 'FAIL', str(detail)[:1000]))
    conn.commit()

def active_model(conn):
    return conn.execute('SELECT * FROM models WHERE active=1 ORDER BY created_at DESC LIMIT 1').fetchone()

def data_mode(conn):
    row = conn.execute('SELECT data_mode,COUNT(*) AS n FROM zones GROUP BY data_mode ORDER BY n DESC LIMIT 1').fetchone()
    return row['data_mode'] if row else 'empty'

def reset_data(conn):
    for name in ['zones','trips','weather','ingestion_log','quality_log','experiments','models','backtest','forecasts','feature_importance','monitoring']:
        conn.execute(f'DELETE FROM {name}')
    conn.commit()
