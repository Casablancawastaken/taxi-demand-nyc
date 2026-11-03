import numpy as np
import pandas as pd
from .settings import FEATURES,SEED
from .db import now


def _score(model,name,frame):
    from .modeling import metrics, predict_rows
    values=predict_rows(name,model,frame if name=='SeasonalNaive' else frame[FEATURES])
    return metrics(frame.trips,values)['mae']

def permutation_scores(model,name,frame,features=None,repeats=2):
    features=features or FEATURES
    if name=='SeasonalNaive':
        return {feature:1.0 if feature=='lag_7' else 0.0 for feature in features}
    data=frame.sample(min(1000,len(frame)),random_state=SEED).copy().reset_index(drop=True)
    reference=_score(model,name,data)
    rng=np.random.default_rng(SEED)
    result={}
    for feature in features:
        scores=[]
        for _ in range(repeats):
            shuffled=data.copy()
            shuffled[feature]=rng.permutation(shuffled[feature].to_numpy())
            scores.append(_score(model,name,shuffled)-reference)
        result[feature]=float(np.mean(scores))
    return result

def calculate_importances(conn,bundle,train,val):
    from .modeling import make_model, metrics, predict_rows
    name=bundle['name']
    version=bundle['version']
    features=bundle['features']
    params=bundle['params']
    if name=='SeasonalNaive':
        permutation=permutation_scores(None,name,val,features)
        ablation={f:permutation[f] for f in features}
        first=last=permutation.copy()
    else:
        training_model=make_model(name,params)
        training_model.fit(train[features],train.trips)
        permutation=permutation_scores(training_model,name,val,features)
        base_mae=metrics(val.trips,predict_rows(name,training_model,val[features]))['mae']
        ablation={}
        for feature in ['lag_1','lag_7','lag_28','rolling_mean_7','dow','temperature_lag1','precipitation_lag1']:
            reduced=[f for f in features if f!=feature]
            model=make_model(name,params,reduced)
            model.fit(train[reduced],train.trips)
            score=metrics(val.trips,predict_rows(name,model,val[reduced]))['mae']
            ablation[feature]=float(score-base_mae)
        val_days=sorted(val.date.unique())
        middle=val_days[len(val_days)//2]
        first=permutation_scores(training_model,name,val[val.date<middle],features,repeats=1)
        last=permutation_scores(training_model,name,val[val.date>=middle],features,repeats=1)
    for method,period,scores in [('permutation','validation',permutation),('ablation','validation',ablation),
                                 ('permutation','early',first),('permutation','late',last)]:
        for feature,importance in scores.items():
            conn.execute('INSERT OR REPLACE INTO feature_importance VALUES(?,?,?,?,?)',
                         (version,method,period,feature,float(importance)))
    conn.commit()
    return {'methods':['permutation','ablation'],'features_tested':len(features),
            'ablation_features':len(ablation),'time_comparison':'early vs late validation'}

def explain_row(bundle,row):
    from .modeling import predict_rows
    features=bundle['features']
    model=bundle['estimator']
    name=bundle['name']
    actual=row[features].iloc[0:1].copy().reset_index(drop=True)
    if name=='SeasonalNaive':
        target=float(actual['lag_7'].iloc[0])
        return {'reference':0.0,'prediction':target,
                'impacts':pd.DataFrame([{'feature':'lag_7','value':target,'impact':target}])}
    reference=actual.copy()
    sample=bundle['reference']
    for feature in features:
        if feature in ['zone_id','borough']:
            continue
        median=pd.to_numeric(sample[feature],errors='coerce').median()
        if np.isfinite(median):
            reference[feature]=median
    current=reference.copy()
    base=float(predict_rows(name,model,current)[0])
    previous=base
    impacts=[]
    for feature in features:
        current[feature]=actual[feature]
        value=float(predict_rows(name,model,current)[0])
        impacts.append({'feature':feature,'value':str(actual[feature].iloc[0]),'impact':value-previous})
        previous=value
    return {'reference':base,'prediction':previous,
            'impacts':pd.DataFrame(impacts).sort_values('impact',key=abs,ascending=False)}

def partial_dependence(bundle,feature,steps=12):
    from .modeling import predict_rows
    if feature not in bundle['features'] or feature in ['zone_id','borough']:
        raise ValueError('Выбери числовой признак')
    sample=bundle['reference'].sample(min(500,len(bundle['reference'])),random_state=SEED).copy()
    vals=pd.to_numeric(sample[feature],errors='coerce').dropna()
    if vals.empty:
        raise ValueError('Нет данных для графика')
    levels=np.linspace(float(vals.quantile(.05)),float(vals.quantile(.95)),steps)
    predictions=[]
    for value in levels:
        temp=sample.copy()
        temp[feature]=value
        predictions.append(float(predict_rows(bundle['name'],bundle['estimator'],temp).mean()))
    return pd.DataFrame({'Значение признака':levels,'Средний прогноз':predictions})
