from datetime import timedelta
import pandas as pd
import pytest
from taxi.db import connect
from taxi.sources import generate_demo
from taxi.features import build_features,load_panel,training_frame,signature
from taxi.modeling import training_pipeline,split_dates,produce_forecast,reproduce_experiment,rollback,load_bundle
from taxi.monitoring import run_monitor


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    from taxi import db,sources,modeling,settings
    folder=tmp_path_factory.mktemp('taxi_test')
    db.DATABASE=folder/'taxi.sqlite3'
    sources.DATA=folder
    (folder/'raw').mkdir()
    modeling.ARTIFACTS=folder/'artifacts'
    settings.ARTIFACTS.mkdir(exist_ok=True)
    modeling.ARTIFACTS.mkdir(exist_ok=True)
    conn=connect()
    generate_demo(conn,end_date='2026-09-30',days=150,reset=True)
    return conn


def test_import_incremental_and_idempotent(prepared):
    before=prepared.execute('SELECT count(*) FROM trips').fetchone()[0]
    generate_demo(prepared,end_date='2026-09-30',days=150)
    after=prepared.execute('SELECT count(*) FROM trips').fetchone()[0]
    assert before==after==12*150
    assert prepared.execute("SELECT count(*) FROM ingestion_log WHERE status='SKIPPED'").fetchone()[0]>=1


def test_feature_target_leakage(prepared):
    panel=load_panel(prepared,max_zones=0)
    original=build_features(panel)
    zone=int(panel.zone_id.iloc[0])
    day=panel.date.unique()[70]
    changed=panel.copy()
    mask=(changed.zone_id==zone)&(changed.date==day)
    changed.loc[mask,'trips']=1000000
    changed.loc[mask,'temperature']=999
    modified=build_features(changed)
    target_cols=['lag_1','lag_7','lag_14','lag_28','rolling_mean_7','rolling_mean_28','temperature_lag1']
    a=original.loc[mask,target_cols].reset_index(drop=True)
    b=modified.loc[mask,target_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b)
    following_day=day+pd.Timedelta(days=1)
    next_mask=(modified.zone_id==zone)&(modified.date==following_day)
    assert original.loc[next_mask,'lag_1'].iloc[0] != modified.loc[next_mask,'lag_1'].iloc[0]


def test_chronology(prepared):
    frame=training_frame(load_panel(prepared,max_zones=0))
    train,val,test=split_dates(frame)
    assert train[-1] < val[0]
    assert val[-1] < test[0]


@pytest.fixture(scope='module')
def trained(prepared):
    result=training_pipeline(prepared,max_zones=0)
    return prepared,result


def test_model_comparison_and_forecast(trained):
    conn,result=trained
    assert len(result['models'])==3
    for row in result['models'].values():
        assert 'validation' in row and 'test' in row
        assert row['test']['mae'] >= 0
        assert row['test']['rmse'] >= row['test']['mae']
    first=produce_forecast(conn,7)
    second=produce_forecast(conn,7)
    assert first['rows']==84
    assert second['rows']==first['rows']
    assert conn.execute('SELECT count(*) FROM forecasts').fetchone()[0]==84
    assert conn.execute('SELECT min(predicted) FROM forecasts').fetchone()[0]>=0


def test_experiment_reproducible(trained):
    conn,result=trained
    experiment=conn.execute("SELECT id FROM experiments WHERE fold='validation' AND model='Ridge' LIMIT 1").fetchone()[0]
    reproduction=reproduce_experiment(conn,experiment)
    assert reproduction['matched'],reproduction
    assert reproduction['diff'] < 1e-8


def test_drift_alert_and_registry(trained):
    conn,result=trained
    normal=run_monitor(conn,simulate=False)
    simulated=run_monitor(conn,simulate=True)
    assert not normal['retrain']
    assert simulated['retrain']
    assert simulated['max_psi'] > normal['max_psi']
    rollback(conn,result['version'])
    assert load_bundle(conn)['version']==result['version']


def test_real_source_endpoints_and_schema():
    from taxi.sources import TRIP_URL,ZONE_GEO_URL,WEATHER_URL,ZONE_CSV_URL
    assert TRIP_URL.startswith('https://d37ci6vzurychx.cloudfront.net/')
    assert '8meu-9t5y' in ZONE_GEO_URL
    assert WEATHER_URL.startswith('https://archive-api.open-meteo.com/')
    assert ZONE_CSV_URL.endswith('.csv')
