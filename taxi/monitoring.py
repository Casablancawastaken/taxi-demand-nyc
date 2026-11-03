import json
import numpy as np
import pandas as pd
from .db import now
from .features import load_panel,training_frame
from .modeling import load_bundle,metrics,predict_rows

WATCH_FEATURES=['lag_1','lag_7','rolling_mean_7','temperature_lag1','precipitation_lag1']

def psi(reference,recent):
    first=np.asarray(reference,dtype=float)
    second=np.asarray(recent,dtype=float)
    first=first[np.isfinite(first)]
    second=second[np.isfinite(second)]
    if len(first)<20 or len(second)<20:
        return 0.0
    edges=np.unique(np.quantile(first,np.linspace(0,1,11)))
    if len(edges)<3:
        return 0.0
    edges[0]=-np.inf
    edges[-1]=np.inf
    a=np.histogram(first,bins=edges)[0].astype(float)
    b=np.histogram(second,bins=edges)[0].astype(float)
    a=np.maximum(a/a.sum(),1e-6)
    b=np.maximum(b/b.sum(),1e-6)
    return float(np.sum((b-a)*np.log(b/a)))

def run_monitor(conn, simulate=False):
    bundle=load_bundle(conn)
    frame=training_frame(load_panel(conn,max_zones=0))
    frame=frame[frame.zone_id.isin({str(x) for x in bundle['zones']})]
    recent_dates=sorted(frame.date.unique())[-14:]
    recent=frame[frame.date.isin(recent_dates)].copy()
    if simulate:
        recent['lag_1']*=2.5
        recent['lag_7']*=1.8
        recent['rolling_mean_7']*=2.3
        recent['temperature_lag1']+=25
    previous_dates=sorted(frame.date.unique())[-56:-14]
    reference=frame[frame.date.isin(previous_dates)]
    drift={feature:psi(reference[feature],recent[feature]) for feature in WATCH_FEATURES}
    worst=max(drift[f] for f in ['lag_1','lag_7'])
    actual=recent.trips.to_numpy()
    estimates=predict_rows(bundle['name'],bundle['estimator'],recent)
    current_wape=metrics(actual,estimates)['wape']
    expected=bundle['all_results'][bundle['name']]['test']['wape']
    threshold=max(0.2,expected*1.35)
    drift_detected=worst>0.4
    degraded=current_wape>threshold
    should_retrain=drift_detected or degraded
    detail={'psi':drift,'reference_holdout_wape':expected,'threshold_wape':threshold,
            'quality_degraded':bool(degraded),'rule':'PSI лагов спроса lag_1/lag_7 > 0.4 (сравнение с предыдущими 42 днями) ИЛИ WAPE за 14 дней > max(20%, 1.35 × WAPE holdout). Остальные PSI диагностические из-за тренда/сезонности',
            'simulation':'Синтетическое искажение признаков только внутри мониторинга' if simulate else None}
    conn.execute('INSERT INTO monitoring(created_at,version,simulated,last_window_wape,psi,drift_detected,retrain_recommended,detail) VALUES(?,?,?,?,?,?,?,?)',
                 (now(),bundle['version'],int(simulate),current_wape,worst,int(drift_detected),int(should_retrain),json.dumps(detail,ensure_ascii=False)))
    conn.commit()
    return {'version':bundle['version'],'wape':current_wape,'max_psi':worst,
            'drift':drift_detected,'retrain':should_retrain,'simulated':simulate,'details':detail}
