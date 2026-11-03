import json
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from taxi.db import connect,active_model,data_mode
from taxi.features import load_panel,training_frame
from taxi.modeling import load_bundle,future_feature_rows,produce_forecast,reproduce_experiment
from taxi.interpret import explain_row,partial_dependence
from taxi.monitoring import run_monitor

st.set_page_config(page_title='TaxiDemand NYC · Вариант 16',page_icon='🚕',layout='wide')
st.markdown('''<style>
.block-container{padding-top:1.8rem;max-width:1480px}
[data-testid="stMetric"]{background:#14243b0b;border:1px solid #8596a42e;border-radius:12px;padding:14px}
h1,h2,h3{letter-spacing:-.02em}
</style>''',unsafe_allow_html=True)

@st.cache_data(ttl=45,show_spinner=False)
def get_tables():
    con=connect()
    try:
        return {name:pd.read_sql_query(sql,con) for name,sql in {
            'zones':'SELECT zone_id, borough, zone, geometry_json,data_mode FROM zones',
            'trips':'SELECT date,zone_id,trips FROM trips',
            'weather':'SELECT date,temperature,precipitation,wind FROM weather',
            'forecasts':'SELECT * FROM forecasts',
            'backtest':'SELECT * FROM backtest',
            'models':'SELECT * FROM models ORDER BY created_at DESC',
            'experiments':'SELECT * FROM experiments ORDER BY created_at DESC',
            'monitoring':'SELECT * FROM monitoring ORDER BY id DESC LIMIT 50',
            'ingestion':'SELECT * FROM ingestion_log ORDER BY id DESC LIMIT 80',
            'quality':'SELECT * FROM quality_log ORDER BY id DESC LIMIT 35',
            'importance':'SELECT * FROM feature_importance'
        }.items()}
    finally:
        con.close()

def reload_data():
    st.cache_data.clear()
    st.rerun()

con=connect()
tables=get_tables()
if tables['zones'].empty or tables['models'].empty:
    st.title('🚕 TaxiDemand NYC')
    st.warning('База или обученная модель пока отсутствует. Выполни команды в терминале проекта.')
    st.code('python -m taxi.cli demo --reset\npython -m taxi.cli train\npython -m taxi.cli forecast\nstreamlit run app.py',language='bash')
    st.stop()
mode=data_mode(con)
active=active_model(con)
bundle=load_bundle(con)
version=active['version']
selected_name=active['selected_model']
zones=tables['zones'].copy()
zones['display']=zones['borough']+' · '+zones['zone']
allowed=set(bundle['zones'])
zones=zones[zones.zone_id.isin(allowed)].copy()
trips=tables['trips'].merge(zones[['zone_id','display','borough']],on='zone_id',how='inner')
forecast=tables['forecasts']
forecast=forecast[(forecast.version==version)&(forecast.origin==bundle['date_end'])].merge(zones[['zone_id','display','borough']],on='zone_id',how='inner')
backtest=tables['backtest'][tables['backtest'].version==version].merge(zones[['zone_id','display']],on='zone_id',how='inner')

st.sidebar.title('🚕 TaxiDemand NYC')
st.sidebar.caption('Проектный практикум · Вариант №16')
st.sidebar.caption('Данные: **'+('СИНТЕТИЧЕСКАЯ ДЕМОНСТРАЦИЯ' if mode=='demo' else 'ОФИЦИАЛЬНЫЕ NYC TLC + Open-Meteo')+'**')
st.sidebar.write('Модель: **'+selected_name+'**')
st.sidebar.caption('Версия: '+version)
st.sidebar.caption('Последняя дата фактов: '+bundle['date_end'])
if st.sidebar.button('Обновить прогноз на 7 дней',use_container_width=True):
    try:
        with st.spinner('Формируется прогноз'):
            produce_forecast(con,7)
        reload_data()
    except Exception as exc:
        st.sidebar.error(str(exc))
if st.sidebar.button('Перезагрузить дашборд',use_container_width=True):
    reload_data()
st.sidebar.divider()
st.sidebar.caption('Интерфейс аналитика таксопарка: оценка спроса по зонам отправления. Объёмы — количество поездок, не пассажиров.')

tab1,tab2,tab3,tab4,tab5,tab6=st.tabs([
    '📊 Оперативная панель','🗺️ Районы и прогноз','🏆 Модели и метрики',
    '🔎 Объяснение прогноза','🛡️ Мониторинг и качество','🧪 Эксперименты и загрузки'])

with tab1:
    st.title('Спрос на такси по районам Нью-Йорка')
    if mode=='demo':
        st.warning('ДЕМО: поездки, погода и геометрия районов синтетические. Показатели не описывают реальный спрос в Нью-Йорке.')
    st.caption('Решение: где распределять автомобили с учётом прогноза поездок на следующие 7 дней.')
    last_date=trips.date.max()
    yesterday=trips[trips.date==last_date]
    next_day=forecast.date.min() if not forecast.empty else None
    next_frame=forecast[forecast.date==next_day] if next_day else pd.DataFrame()
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Последний день факта',str(last_date))
    c2.metric('Поездок за день',f'{int(yesterday.trips.sum()):,}'.replace(',',' '))
    c3.metric('Прогноз следующего дня',f'{int(next_frame.predicted.sum()):,}'.replace(',',' ') if not next_frame.empty else 'Нет')
    c4.metric('Ошибка holdout · WAPE',f"{bundle['all_results'][selected_name]['test']['wape']:.1%}")
    historical=trips.groupby('date',as_index=False).trips.sum().tail(65)
    fig=go.Figure()
    fig.add_scatter(x=historical.date,y=historical.trips,name='Факт: поездки',mode='lines+markers',line=dict(color='#159895',width=3))
    if not forecast.empty:
        future=forecast.groupby('date',as_index=False).predicted.sum().sort_values('date')
        fig.add_scatter(x=future.date,y=future.predicted,name='Прогноз: поездки',mode='lines+markers',line=dict(color='#e49a32',width=3,dash='dash'))
    fig.update_layout(title='Число поездок: история и следующий прогнозный период',xaxis_title='Дата',yaxis_title='Поездок в сутки',hovermode='x unified',legend=dict(orientation='h'))
    st.plotly_chart(fig,use_container_width=True)
    left,right=st.columns([1.2,1])
    with left:
        st.subheader('Где ожидается наибольший спрос?')
        if not next_frame.empty:
            top=next_frame.nlargest(10,'predicted').sort_values('predicted')
            st.plotly_chart(px.bar(top,x='predicted',y='display',orientation='h',labels={'predicted':'Поездок','display':'Район'},color_discrete_sequence=['#159895']),use_container_width=True)
    with right:
        st.subheader('Насколько точен прогноз?')
        if not backtest.empty:
            totals=backtest.groupby('date',as_index=False)[['actual','predicted']].sum().tail(28)
            fig2=px.line(totals,x='date',y=['actual','predicted'],labels={'value':'Число поездок','date':'Дата','variable':'Ряд'},color_discrete_sequence=['#159895','#e49a32'])
            st.plotly_chart(fig2,use_container_width=True)
        st.caption('Backtest: прогноз на один день при известных фактах прошлых дней. Прогноз на 7 дней строится рекурсивно и может накапливать ошибку.')

with tab2:
    st.header('Прогноз по отдельным районам')
    if forecast.empty:
        st.info('Сначала выполни python -m taxi.cli forecast')
    else:
        col_a,col_b=st.columns(2)
        pick=col_a.selectbox('Район отправления',zones.display.sort_values().tolist(),key='zone_select')
        day=col_b.select_slider('Дата прогноза',options=sorted(forecast.date.unique()))
        chosen=forecast[(forecast.display==pick)]
        history=trips[trips.display==pick].sort_values('date').tail(35)
        chart=go.Figure()
        chart.add_scatter(x=history.date,y=history.trips,name='Факт',line=dict(color='#159895'))
        chart.add_scatter(x=chosen.date,y=chosen.predicted,name='Прогноз',line=dict(color='#e49a32',dash='dash'),mode='lines+markers')
        chart.update_layout(title=pick,xaxis_title='Дата',yaxis_title='Поездок в сутки')
        st.plotly_chart(chart,use_container_width=True)
        selection=forecast[forecast.date==day].sort_values('predicted',ascending=False)
        st.metric('Ожидаемый спрос в выбранном районе',f"{chosen[chosen.date==day].predicted.iloc[0]:,.0f}".replace(',',' ')+' поездок')
        st.subheader('Рейтинг районов на '+day)
        st.dataframe(selection[['display','predicted']].rename(columns={'display':'Район','predicted':'Прогноз поездок'}),hide_index=True,use_container_width=True)
        csv=selection[['date','zone_id','display','predicted']].to_csv(index=False).encode('utf-8-sig')
        st.download_button('Скачать прогноз CSV',csv,file_name=f'taxi_forecast_{day}.csv',mime='text/csv')
        with st.expander('Карта зон поездок'):
            geo={'type':'FeatureCollection','features':[]}
            for entry in zones.itertuples(index=False):
                try:
                    shape=json.loads(entry.geometry_json)
                    geo['features'].append({'type':'Feature','id':str(entry.zone_id),'properties':{'name':entry.zone},'geometry':shape})
                except (ValueError,TypeError):
                    continue
            if geo['features']:
                selection_map=selection.assign(id=selection.zone_id.astype(str))
                fig=px.choropleth_mapbox(selection_map,geojson=geo,locations='id',featureidkey='id',
                                         color='predicted',hover_name='display',zoom=10,center={'lat':40.73,'lon':-73.98},
                                         mapbox_style='carto-positron',opacity=.6,labels={'predicted':'Поездок'},
                                         color_continuous_scale='Teal')
                fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),height=480)
                st.plotly_chart(fig,use_container_width=True)
                if mode=='demo':st.caption('Контуры демо-районов схематические и не соответствуют официальным границам.')

with tab3:
    st.header('Сравнение прогнозных моделей')
    records=[]
    for model,details in bundle['all_results'].items():
        for part in ['validation','test']:
            score=details[part]
            records.append({'Модель':model,'Выборка':'Валидация' if part=='validation' else 'Holdout',
                            'MAE, поездок':round(score['mae'],1),'RMSE, поездок':round(score['rmse'],1),
                            'WAPE, %':round(score['wape']*100,2),'Сумма абсолютных ошибок':int(score['total_abs_error'])})
    stats=pd.DataFrame(records)
    st.dataframe(stats,hide_index=True,use_container_width=True)
    st.plotly_chart(px.bar(stats,x='Модель',y='WAPE, %',color='Выборка',barmode='group',title='Сопоставление по одинаковым временным отрезкам'),use_container_width=True)
    st.success('Действующая модель выбирается только по валидации, а holdout используется для окончательной проверки.')
    st.caption(f"Train до {bundle['train_end']} · Валидация до {bundle['validation_end']} · Holdout до {bundle['test_end']} · Seed = 42 · Версия {version}")
    st.subheader('Перевод ошибки в понятный показатель')
    st.write('MAE — средняя абсолютная ошибка количества поездок за сутки в одном районе. WAPE показывает совокупное отклонение от реального числа поездок.')

with tab4:
    st.header('Почему модель сформировала такой прогноз?')
    items=tables['importance']
    items=items[items.version==version]
    l,r=st.columns(2)
    for column,method,heading in [(l,'permutation','1. Перемешивание признака'),(r,'ablation','2. Исключение признака и переобучение')]:
        with column:
            score=items[(items.method==method)&(items.period=='validation')].nlargest(12,'importance').sort_values('importance')
            st.subheader(heading)
            st.plotly_chart(px.bar(score,x='importance',y='feature',orientation='h',labels={'importance':'Увеличение MAE','feature':'Признак'},color_discrete_sequence=['#159895']),use_container_width=True)
    st.caption('Положительная важность означает, что удаление информации увеличивает MAE. Для методов использована временная валидационная выборка, а не holdout.')
    compare=items[(items.method=='permutation')&(items.period.isin(['early','late']))]
    if not compare.empty:
        compare=compare.pivot(index='feature',columns='period',values='importance').fillna(0)
        important=compare.abs().max(axis=1).nlargest(8).index
        st.subheader('Стабильность важности: начало и конец валидационного периода')
        st.plotly_chart(px.bar(compare.loc[important].reset_index().melt(id_vars='feature'),x='feature',y='value',color='period',barmode='group'),use_container_width=True)
    st.divider()
    st.subheader('Объяснение одной поездкодаты и района')
    date_type=st.radio('Для какого периода?', ['Исторический день','Прогнозный день'],horizontal=True)
    chosen_zone=st.selectbox('Район',zones.display.sort_values().tolist(),key='explain_zone')
    zone_id=int(zones[zones.display==chosen_zone].zone_id.iloc[0])
    if date_type=='Прогнозный день':
        available=sorted(forecast.date.unique())
        select_day=st.selectbox('Прогнозная дата',available)
        rowset=future_feature_rows(con,bundle,7)
        rowset=rowset[(rowset.zone_id==str(zone_id))&(rowset.date==pd.Timestamp(select_day))]
    else:
        history=training_frame(load_panel(con,max_zones=0))
        history=history[history.zone_id==str(zone_id)]
        available=sorted(history.date.dt.strftime('%Y-%m-%d').tolist())
        select_day=st.selectbox('Историческая дата',available,index=len(available)-1)
        rowset=history[history.date==pd.Timestamp(select_day)]
    if not rowset.empty:
        explanation=explain_row(bundle,rowset)
        m1,m2=st.columns(2)
        m1.metric('Прогноз',f"{explanation['prediction']:,.0f}".replace(',',' ')+' поездок')
        m2.metric('Базовая точка объяснения',f"{explanation['reference']:,.0f}".replace(',',' ')+' поездок')
        impact=explanation['impacts'].head(12).sort_values('impact')
        st.plotly_chart(px.bar(impact,x='impact',y='feature',orientation='h',color='impact',color_continuous_scale='RdYlGn',labels={'impact':'Вклад в прогноз, поездок','feature':'Признак'}),use_container_width=True)
        st.caption('Локальные вклады точно складываются в прогноз (с учётом базовой точки), но для нелинейной модели зависят от порядка замены признаков. Это не причинно-следственные эффекты.')
    if bundle['name']!='SeasonalNaive':
        st.subheader('Зависимость среднего прогноза от одного признака (PDP)')
        number_feature=st.selectbox('Признак для проверки зависимости',['lag_1','lag_7','rolling_mean_7','dow','temperature_lag1','precipitation_lag1'])
        pdp=partial_dependence(bundle,number_feature)
        st.plotly_chart(px.line(pdp,x='Значение признака',y='Средний прогноз',markers=True),use_container_width=True)

with tab5:
    st.header('Мониторинг качества и дрейфа')
    c1,c2=st.columns(2)
    with c1:
        if st.button('Проверить текущие данные',use_container_width=True):
            run_monitor(con,False)
            reload_data()
    with c2:
        if st.button('Смоделировать ухудшение данных',use_container_width=True):
            run_monitor(con,True)
            reload_data()
    recent_log=tables['monitoring']
    if not recent_log.empty:
        item=recent_log.iloc[0]
        drift_info=json.loads(item.detail)
        if item.retrain_recommended:
            st.warning('Рекомендовано переобучение модели')
        else:
            st.success('Отклонений по пороговому правилу не выявлено')
        if item.simulated:
            st.info('Это искусственный сбой для защиты. Исходная база и обученная модель не менялись.')
        a,b,c=st.columns(3)
        a.metric('WAPE последних 14 дней',f'{item.last_window_wape:.1%}')
        b.metric('Максимальный PSI (отслеживаемые)',f'{item.psi:.2f}')
        c.metric('Порог WAPE',f"{drift_info['threshold_wape']:.1%}")
        drift_table=pd.DataFrame([{'Признак':key,'PSI':value} for key,value in drift_info['psi'].items()])
        st.plotly_chart(px.bar(drift_table,x='Признак',y='PSI',color='PSI',title='Изменение распределений признаков'),use_container_width=True)
        st.caption(drift_info['rule'])
        st.subheader('Последние проверки')
        st.dataframe(recent_log[['created_at','simulated','last_window_wape','psi','drift_detected','retrain_recommended']].rename(columns={'created_at':'Время','simulated':'Симуляция','last_window_wape':'WAPE','psi':'PSI','drift_detected':'Дрейф','retrain_recommended':'Переобучение'}),hide_index=True,use_container_width=True)
    st.subheader('Контроль качества витрины')
    if not tables['quality'].empty:
        st.dataframe(tables['quality'][['created_at','test_name','status','detail']],hide_index=True,use_container_width=True)
    st.subheader('Реестр версий моделей')
    st.dataframe(tables['models'][['version','created_at','selected_model','active']],hide_index=True,use_container_width=True)
    st.caption('Откат версии: python -m taxi.cli rollback ВЕРСИЯ; затем python -m taxi.cli forecast.')

with tab6:
    st.header('Воспроизводимые эксперименты и журнал загрузок')
    st.subheader('Подбор гиперпараметров и временная валидация')
    experiments=tables['experiments']
    st.dataframe(experiments[['id','model','fold','mae','rmse','wape','seed','version']].head(100),hide_index=True,use_container_width=True)
    if not experiments.empty:
        experiment_id=st.selectbox('Эксперимент для повторения',experiments.id.tolist())
        if st.button('Воспроизвести эксперимент'):
            try:
                result=reproduce_experiment(con,experiment_id)
                st.json(result)
            except Exception as exc:
                st.error(str(exc))
    st.subheader('Журнал загрузок из источников')
    st.dataframe(tables['ingestion'][['created_at','source','source_key','mode','status','rows_processed','detail']],hide_index=True,use_container_width=True)
    st.caption('SUCCESS — загружено; SKIPPED — повтор не создал дублей; FAILED — зафиксирована ошибка. Исходные файлы сохраняются в data/raw/.')

con.close()
