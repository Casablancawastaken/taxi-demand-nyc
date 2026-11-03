import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .settings import ARTIFACTS, CATEGORICAL, FEATURES, NUMERICAL, SEED
from .db import now, active_model
from .features import load_panel, training_frame, build_features, signature, validate_data

CANDIDATES = {
    'Ridge': [{'alpha':1.0},{'alpha':25.0}],
    'HistGradientBoosting': [
        {'max_iter':75,'max_leaf_nodes':15,'learning_rate':0.07,'l2_regularization':1.0},
        {'max_iter':125,'max_leaf_nodes':23,'learning_rate':0.05,'l2_regularization':3.0},
    ],
}

def metrics(y,pred):
    y=np.asarray(y,dtype=float)
    pred=np.clip(np.asarray(pred,dtype=float),0,None)
    diff=np.abs(y-pred)
    return {'mae':float(diff.mean()),'rmse':float(np.sqrt(np.square(y-pred).mean())),
            'wape':float(diff.sum()/max(float(np.abs(y).sum()),1.0)),
            'total_abs_error':float(diff.sum()), 'n':int(len(y))}

def make_model(name,params,features=FEATURES):
    categorical = [f for f in CATEGORICAL if f in features]
    numerical = [f for f in NUMERICAL if f in features]
    preprocessor=ColumnTransformer([
        ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical),
        ('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]) if name=='Ridge'
         else SimpleImputer(strategy='median'),numerical)
    ],sparse_threshold=0)
    if name=='Ridge':
        estimator=Ridge(alpha=params.get('alpha',1.0))
    else:
        estimator=HistGradientBoostingRegressor(random_state=SEED,early_stopping=False, **params)
    return Pipeline([('prep',preprocessor),('estimator',estimator)])

def predict_rows(name,model,rows):
    if name=='SeasonalNaive':
        return np.clip(rows['lag_7'].to_numpy(dtype=float),0,None)
    return np.clip(model.predict(rows),0,None)

def split_dates(frame):
    dates=np.array(sorted(frame.date.unique()))
    if len(dates)<80:
        raise ValueError('Недостаточно дат после формирования лагов')
    train_end=int(len(dates)*0.6)
    validation_end=int(len(dates)*0.8)
    return dates[:train_end],dates[train_end:validation_end],dates[validation_end:]

def record_experiment(conn,version,data_sig,name,params,fold,score):
    ident=uuid.uuid4().hex[:12]
    conn.execute('INSERT INTO experiments VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                 (ident,now(),version,data_sig,name,json.dumps(params,ensure_ascii=False,sort_keys=True),SEED,fold,
                  score['mae'],score['rmse'],score['wape']))
    return ident

def _fit_predict(name,params,features,train_frame,eval_frame):
    if name=='SeasonalNaive':
        return None,predict_rows(name,None,eval_frame)
    fitted=make_model(name,params,features)
    fitted.fit(train_frame[features],train_frame['trips'])
    return fitted,predict_rows(name,fitted,eval_frame[features])

def training_pipeline(conn, max_zones=None, force=False):
    from .interpret import calculate_importances
    panel=load_panel(conn,max_zones)
    validate_data(conn,panel)
    frame=training_frame(panel)
    data_sig=signature(frame)
    if not force:
        old = conn.execute('SELECT version FROM models WHERE data_signature=? ORDER BY created_at DESC LIMIT 1',(data_sig,)).fetchone()
        if old:
            manifest_path=ARTIFACTS/f"manifest_{old['version']}.json"
            if manifest_path.exists():
                result=json.loads(manifest_path.read_text(encoding='utf-8'))
                result['reused_existing_experiment']=True
                return result
    incumbent=active_model(conn)
    incumbent_bundle=load_bundle(conn) if incumbent is not None else None
    train_dates,val_dates,test_dates=split_dates(frame)
    train=frame[frame.date.isin(train_dates)].copy()
    val=frame[frame.date.isin(val_dates)].copy()
    test=frame[frame.date.isin(test_dates)].copy()
    train_val=frame[frame.date.isin(np.concatenate([train_dates,val_dates]))].copy()
    version=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:5]
    results={}
    best_configs={}
    naive_val=metrics(val.trips,predict_rows('SeasonalNaive',None,val))
    record_experiment(conn,version,data_sig,'SeasonalNaive',{'lag':7,'train_end':str(train_dates[-1])[:10],
        'eval_start':str(val_dates[0])[:10],'eval_end':str(val_dates[-1])[:10]},'validation',naive_val)
    results['SeasonalNaive']={'validation':naive_val,'params':{'lag':7}}
    best_configs['SeasonalNaive']={'lag':7}
    for name,params_list in CANDIDATES.items():
        best_score=float('inf')
        best_params=None
        train_unique=np.array(sorted(train.date.unique()))
        for config in params_list:
            wapes=[]
            for fold in range(3):
                start=int(len(train_unique)*(0.52+0.12*fold))
                end=min(len(train_unique),start+max(12,int(len(train_unique)*0.11)))
                train_part=train[train.date.isin(train_unique[:start])]
                valid_part=train[train.date.isin(train_unique[start:end])]
                _,prediction=_fit_predict(name,config,FEATURES,train_part,valid_part)
                score=metrics(valid_part.trips,prediction)
                run_params={**config,'train_end':str(train_unique[start-1])[:10],
                    'eval_start':str(train_unique[start])[:10],'eval_end':str(train_unique[end-1])[:10]}
                record_experiment(conn,version,data_sig,name,run_params,f'cv{fold+1}',score)
                wapes.append(score['wape'])
            mean_score=float(np.mean(wapes))
            if mean_score<best_score:
                best_params=config.copy()
                best_score=mean_score
        model,prediction=_fit_predict(name,best_params,FEATURES,train,val)
        score=metrics(val.trips,prediction)
        run_params={**best_params,'train_end':str(train_dates[-1])[:10],
                    'eval_start':str(val_dates[0])[:10],'eval_end':str(val_dates[-1])[:10]}
        record_experiment(conn,version,data_sig,name,run_params,'validation',score)
        results[name]={'validation':score,'params':best_params,'cv_wape':best_score}
        best_configs[name]=best_params
    selected=min(results,key=lambda name:results[name]['validation']['wape'])
    fitted_models={}
    for name in results:
        model,prediction=_fit_predict(name,best_configs[name],FEATURES,train_val,test)
        results[name]['test']=metrics(test.trips,prediction)
        fitted_models[name]=model
        record_experiment(conn,version,data_sig,name,
                          {**best_configs[name],'train_end':str(val_dates[-1])[:10],
                           'eval_start':str(test_dates[0])[:10],'eval_end':str(test_dates[-1])[:10]},
                          'holdout',results[name]['test'])
    winner=fitted_models[selected]
    reference=train_val[FEATURES].sample(min(1500,len(train_val)),random_state=SEED).copy()
    bundle={'name':selected,'estimator':winner,'features':FEATURES,'reference':reference,
            'zones':sorted(int(x) for x in frame.zone_id.unique()),
            'data_signature':data_sig,'date_end':str(panel.date.max().date()),'version':version,
            'train_end':str(train_dates[-1])[:10],'validation_end':str(val_dates[-1])[:10],
            'test_end':str(test_dates[-1])[:10], 'params':best_configs[selected],
            'all_results':results}
    path=ARTIFACTS/f'model_{version}.joblib'
    joblib.dump(bundle,path,compress=3)
    test_pred=predict_rows(selected,winner,test[FEATURES] if selected!='SeasonalNaive' else test)
    for row,prediction in zip(test[['date','zone_id','trips']].itertuples(index=False),test_pred):
        conn.execute('INSERT INTO backtest VALUES(?,?,?,?,?)',(version,row.date.strftime('%Y-%m-%d'),int(row.zone_id),float(row.trips),float(prediction)))
    promotion={'decision':'activated_first_version','rule':'Первую обученную версию можно включить сразу'}
    activate=incumbent is None
    if incumbent_bundle is not None:
        shared=test[test.zone_id.isin({str(z) for z in incumbent_bundle['zones']})]
        if not shared.empty:
            old_score=metrics(shared.trips,predict_rows(incumbent_bundle['name'],incumbent_bundle['estimator'],shared))
            new_score=metrics(shared.trips,predict_rows(selected,winner,shared))
            activate=new_score['wape'] < old_score['wape']*0.98
            promotion={'decision':'activated' if activate else 'staged',
                       'rule':'Новая версия включается, если WAPE ниже действующей на ≥2% на одном контрольном отрезке',
                       'current_wape':old_score['wape'],'candidate_wape':new_score['wape'],
                       'comparable_zones':int(shared.zone_id.nunique())}
        else:
            promotion={'decision':'staged','rule':'Нет общих районов для сравнительного испытания'}
    if activate:
        conn.execute('UPDATE models SET active=0')
    conn.execute('INSERT INTO models VALUES(?,?,?,?,?,?,?,?,?)',
                 (version,now(),selected,json.dumps(results,ensure_ascii=False),json.dumps(FEATURES),
                  json.dumps(best_configs,ensure_ascii=False),data_sig,str(path),int(activate)))
    conn.commit()
    importance_info=calculate_importances(conn,bundle,train,val)
    manifest={'version':version,'selected':selected,'models':results,'data_signature':data_sig,
              'seed':SEED,'train_dates':[str(train_dates[0])[:10],str(train_dates[-1])[:10]],
              'val_dates':[str(val_dates[0])[:10],str(val_dates[-1])[:10]],
              'test_dates':[str(test_dates[0])[:10],str(test_dates[-1])[:10]],
              'features':FEATURES,'importance':importance_info,'promotion':promotion}
    (ARTIFACTS/f'manifest_{version}.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    return manifest

def load_bundle(conn,version=None):
    if version is None:
        row=active_model(conn)
    else:
        row=conn.execute('SELECT * FROM models WHERE version=?',(version,)).fetchone()
    if row is None:
        raise RuntimeError('Модель ещё не обучена')
    path=Path(row['artifact'])
    if not path.exists():
        path=ARTIFACTS/path.name
    if not path.exists():
        raise FileNotFoundError(f'Не найден файл модели: {path}')
    return joblib.load(path)

def future_feature_rows(conn,bundle,days=7):
    panel=load_panel(conn,max_zones=0)
    selected=set(bundle['zones'])
    panel=panel[panel.zone_id.isin(selected)].copy()
    all_new=[]
    latest_date=panel.date.max()
    if latest_date.strftime('%Y-%m-%d')!=bundle['date_end']:
        raise RuntimeError('После добавления новых фактов необходимо переобучить модель: python -m taxi.cli train')
    weather=panel[['temperature','precipitation','wind']].dropna().tail(1)
    carry=weather.iloc[0].to_dict() if not weather.empty else {'temperature':0,'precipitation':0,'wind':0}
    zones=panel[['zone_id','borough','zone']].drop_duplicates()
    for step in range(days):
        current=latest_date+pd.Timedelta(days=step+1)
        new=zones.assign(date=current,trips=np.nan,**carry)
        combined=pd.concat([panel,new],ignore_index=True)
        features=build_features(combined)
        rows=features[features.date.eq(current)].copy()
        rows['zone_id']=rows['zone_id'].astype(str)
        predictions=predict_rows(bundle['name'],bundle['estimator'],rows[FEATURES] if bundle['name']!='SeasonalNaive' else rows)
        rows['predicted']=predictions
        all_new.append(rows.copy())
        new['trips']=predictions
        panel=pd.concat([panel,new],ignore_index=True)
    return pd.concat(all_new,ignore_index=True)

def produce_forecast(conn,days=7):
    bundle=load_bundle(conn)
    rows=future_feature_rows(conn,bundle,days)
    origin=bundle['date_end']
    conn.executemany('INSERT OR REPLACE INTO forecasts(version,origin,date,zone_id,predicted) VALUES(?,?,?,?,?)',
                     [(bundle['version'],origin,record.date.strftime('%Y-%m-%d'),int(record.zone_id),float(record.predicted))
                      for record in rows.itertuples(index=False)])
    conn.commit()
    return {'model':bundle['name'],'version':bundle['version'],'origin':origin,
            'days':days,'zones':len(bundle['zones']),'rows':len(rows)}

def rollback(conn,version):
    row=conn.execute('SELECT version FROM models WHERE version=?',(version,)).fetchone()
    if row is None:
        raise RuntimeError('Версия не найдена')
    conn.execute('UPDATE models SET active=CASE WHEN version=? THEN 1 ELSE 0 END',(version,))
    conn.commit()
    return version

def reproduce_experiment(conn, experiment_id):
    row=conn.execute('SELECT * FROM experiments WHERE id=?',(experiment_id,)).fetchone()
    if row is None:
        raise ValueError('Эксперимент с таким ID не существует')
    bundle=load_bundle(conn,row['version'])
    panel=load_panel(conn,max_zones=0)
    panel=panel[panel.zone_id.isin(bundle['zones'])]
    frame=training_frame(panel)
    if signature(frame)!=row['data_signature']:
        raise ValueError('Исходные данные изменились: воспроизведение без снимка данных невозможно')
    params=json.loads(row['parameters'])
    train=frame[frame.date<=pd.Timestamp(params['train_end'])]
    validation=frame[(frame.date>=pd.Timestamp(params['eval_start'])) & (frame.date<=pd.Timestamp(params['eval_end']))]
    if validation.empty:
        raise RuntimeError('Пустая проверочная часть')
    config={k:v for k,v in params.items() if k not in ['train_end','eval_start','eval_end','lag']}
    _,pred=_fit_predict(row['model'],config,FEATURES,train,validation)
    score=metrics(validation.trips,pred)
    return {'experiment_id':experiment_id,'model':row['model'],'original_wape':row['wape'],
            'reproduced_wape':score['wape'],'diff':abs(row['wape']-score['wape']),
            'matched':abs(row['wape']-score['wape'])<1e-8}
