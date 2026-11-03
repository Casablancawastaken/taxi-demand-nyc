import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from pptx import Presentation
from pptx.util import Inches,Pt as SlidePt
from pptx.dml.color import RGBColor as SlideColor
from pptx.enum.shapes import MSO_SHAPE

ROOT=Path(__file__).resolve().parent.parent
DOCS=ROOT/'docs'
MANIFEST=sorted((ROOT/'artifacts').glob('manifest_*.json'))[-1]
DATA=json.loads(MANIFEST.read_text(encoding='utf-8'))
WINNER=DATA['selected']
WINNER_TEST=DATA['models'][WINNER]['test']
TRAIN_START,TRAIN_END=DATA['train_dates']
VAL_START,VAL_END=DATA['val_dates']
TEST_START,TEST_END=DATA['test_dates']
SOURCE_PERIOD='01.01–30.06.2024'
SOURCE_DAYS=182
ZONE_COUNT=30
FORECAST_DAYS=7

def ru_date(value):
    year,month,day=value.split('-')
    return f'{day}.{month}.{year}'

def metric(value):
    return f'{value:.2f}'.replace('.',',')

def percentage(value):
    return f'{value:.2%}'.replace('.',',')

COLORS={'navy':'#0E1D32','blue':'#1D3553','teal':'#0CB7A5','gold':'#F0AD4E','white':'#FFFFFF','muted':'#9FADBE','light':'#EAF2F5'}
plt.rcParams['font.family']='DejaVu Sans'


def build_architecture():
    fig,ax=plt.subplots(figsize=(16,8.2),dpi=170)
    fig.patch.set_facecolor('#0E1D32')
    ax.set_facecolor('#0E1D32')
    ax.set_xlim(0,16)
    ax.set_ylim(0,8.2)
    ax.axis('off')
    nodes=[
        (0.3,5.9,2.25,1.08,'NYC TLC\nParquet / CSV','#24577A'),
        (0.3,4.35,2.25,1.08,'Open-Meteo\nJSON','#24577A'),
        (0.3,2.8,2.25,1.08,'NYC Taxi Zones\nGeoJSON / SHP','#24577A'),
        (3.15,4.35,2.45,1.35,'Сбор + журнал\nи обработка ошибок','#287B80'),
        (6.1,4.35,2.45,1.35,'SQLite\nФакт / погода / зоны','#22476B'),
        (9.05,4.35,2.55,1.35,'Признаки + контроль\nкачества / утечки','#22476B'),
        (12.1,4.35,3.3,1.35,'3 модели +\nограниченный подбор','#0B7F76'),
        (12.1,2.4,3.3,1.15,'Прогноз на 7 дней\n(рекурсивно)','#0B7F76'),
        (9.05,0.55,6.35,1.05,'Streamlit + Plotly: факт, прогноз, метрики, карта','#2A566B'),
        (5.0,1.6,3.0,1.12,'PSI / WAPE / версии\nмониторинг и откат','#3E506C'),
        (5.0,0.23,3.0,1.12,'Интерпретация\nPDP / важность / вклад','#3E506C'),
    ]
    edges=[((2.55,6.44),(3.15,5.25)),((2.55,4.89),(3.15,5.02)),((2.55,3.34),(3.15,4.65)),
           ((5.6,5.02),(6.1,5.02)),((8.55,5.02),(9.05,5.02)),((11.6,5.02),(12.1,5.02)),
           ((13.75,4.35),(13.75,3.55)),((13.75,2.4),(13.75,1.58)),
           ((12.1,4.55),(8.0,2.3)),((8.0,2.15),(8.05,1.75)),((9.05,4.65),(7.95,1.12))]
    for a,b in edges:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,linewidth=1.9,color='#79A6B5',alpha=.75,connectionstyle='arc3,rad=.0'))
    for x,y,w,h,label,color in nodes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.11,rounding_size=.18',facecolor=color,edgecolor='#5AB9B7',linewidth=1.2))
        ax.text(x+w/2,y+h/2,label,color='white',ha='center',va='center',fontsize=12,fontweight='bold',linespacing=1.5)
    ax.text(.35,7.62,'TAXIDEMAND NYC  ·  АРХИТЕКТУРА',fontweight='bold',color='white',fontsize=24)
    ax.text(.35,7.22,'Источники → хранение → признаки → модели → прогноз → решение',color='#ACCBDB',fontsize=13)
    plt.savefig(DOCS/'architecture.png',bbox_inches='tight',pad_inches=.18,facecolor=fig.get_facecolor())
    plt.close(fig)


def build_metrics_chart():
    names=['Naive','Ridge','Boosting']
    val=[DATA['models'][key]['validation']['wape']*100 for key in ['SeasonalNaive','Ridge','HistGradientBoosting']]
    test=[DATA['models'][key]['test']['wape']*100 for key in ['SeasonalNaive','Ridge','HistGradientBoosting']]
    fig,ax=plt.subplots(figsize=(10.2,4),dpi=160)
    x=list(range(3))
    ax.bar([i-.18 for i in x],val,width=.35,color='#0CB7A5',label='Валидация')
    ax.bar([i+.18 for i in x],test,width=.35,color='#F0AD4E',label='Holdout')
    ax.set_xticks(x,names)
    ax.set_ylim(0,max(val+test)*1.2)
    ax.set_ylabel('WAPE, %')
    ax.set_title('Сравнение моделей: NYC TLC, январь–июнь 2024')
    ax.legend()
    ax.grid(axis='y',alpha=.18)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS/'metrics_comparison.png',bbox_inches='tight')
    plt.close(fig)


def add_doc_paragraph(document,text='',bold=False,style=None):
    p=document.add_paragraph(style=style)
    run=p.add_run(text)
    run.bold=bold
    return p


def add_doc_table(document,headers,rows,widths=None):
    table=document.add_table(rows=1, cols=len(headers))
    table.style='Light Shading Accent 1'
    table.alignment=WD_TABLE_ALIGNMENT.CENTER
    for cell,header in zip(table.rows[0].cells,headers):
        cell.text=str(header)
    for row in rows:
        cells=table.add_row().cells
        for cell,value in zip(cells,row):
            cell.text=str(value)
    document.add_paragraph()
    return table


def build_docx():
    doc=Document()
    section=doc.sections[0]
    section.top_margin=Cm(1.9)
    section.bottom_margin=Cm(1.9)
    section.left_margin=Cm(2.05)
    section.right_margin=Cm(1.55)
    styles=doc.styles
    normal=styles['Normal']
    normal.font.name='DejaVu Sans'
    normal.font.size=Pt(9.5)
    normal.paragraph_format.space_after=Pt(5)
    for key,sz in [('Title',18),('Heading 1',13),('Heading 2',11)]:
        s=styles[key]
        s.font.name='DejaVu Sans'
        s.font.size=Pt(sz)
        s.font.color.rgb=RGBColor(20,73,90)
    footer=section.footer.paragraphs[0]
    footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run('TaxiDemand NYC · Проектный практикум · Вариант №16')
    front=doc.add_paragraph()
    front.alignment=WD_ALIGN_PARAGRAPH.CENTER
    front.paragraph_format.space_before=Pt(34)
    front.add_run('ОТЧЁТ О ПРОЕКТНОЙ РАБОТЕ').bold=True
    top=doc.add_paragraph()
    top.alignment=WD_ALIGN_PARAGRAPH.CENTER
    run=top.add_run('Прогноз спроса на поездки такси\nпо районам города Нью-Йорка')
    run.font.size=Pt(18)
    run.bold=True
    add_doc_paragraph(doc,'Дисциплина: «Проектный практикум». Предметная область №16 — «Городской трафик и такси».')
    add_doc_paragraph(doc,'Программный прототип: TaxiDemand NYC (Python, SQLite, scikit-learn, Streamlit).')
    add_doc_paragraph(doc,f'Состояние: 03.11.2026. Прототип испытан на реальных открытых данных NYC TLC за {SOURCE_PERIOD}: {SOURCE_DAYS} календарных дня, {ZONE_COUNT} зон для моделирования. Источники поездок, погоды и геозон загружены.')
    doc.add_heading('1. Постановка задачи',level=1)
    add_doc_paragraph(doc,'Требуется прогнозировать количество поездок жёлтого такси, начинающихся в конкретной зоне NYC TLC за календарные сутки. Цель — предложить диспетчеру обоснованный ориентир для распределения автомобилей на ближайшие семь дней. Объект прогноза: (дата, ID зоны отправления), целевая переменная y — число записей о начавшихся поездках за сутки.')
    add_doc_paragraph(doc,'Функции системы: сбор разнородных данных, хранение исходных и обработанных результатов, проверка качества, формирование признаков, сравнение трёх моделей, объяснение прогноза, контроль изменений данных и модели, воспроизводимость экспериментов, интерактивная аналитика.')
    doc.add_heading('2. Предметная область и источники данных',level=1)
    add_doc_paragraph(doc,'Нью-Йорк выбран ввиду открытых помесячных записей Taxi and Limousine Commission. В качестве района используется taxi zone (зона NYC TLC), а не административный округ. Количество поездок характеризует зарегистрированный спрос на жёлтое такси в зоне, но не все услуги перевозок города.')
    add_doc_table(doc,['Источник','Формат, роль','Инкрементальность'],[
      ['NYC TLC Yellow Taxi','Parquet: время начала поездки, PULocationID; целевой показатель','Новые месячные файлы'],
      ['Open-Meteo Historical','JSON: суточная температура, осадки, ветер','Недостающие календарные дни'],
      ['NYC Taxi Zones','GeoJSON / SHP: Polygon и MultiPolygon; CSV-справочник имён','Загрузка справочника при отсутствии']
    ])
    add_doc_paragraph(doc,'Основной эксперимент выполнен на реальных исторических данных NYC TLC Yellow Taxi, Open-Meteo и геоданных NYC Taxi Zones. Дополнительно предусмотрен автономный режим с синтетическими CSV, JSON и GeoJSON для проверки функциональности без сети. Персональные данные в агрегатах не используются.')
    doc.add_heading('3. Архитектура системы',level=1)
    p=doc.add_paragraph()
    p.add_run().add_picture(str(DOCS/'architecture.png'),width=Cm(16))
    add_doc_paragraph(doc,'Исходные файлы сохраняются в data/raw/, агрегаты и журналы — в SQLite, обученные модели и параметры — в artifacts/. CLI обеспечивает автоматизированное выполнение, Streamlit — интерфейс диспетчера. Объединение происходит по дате и zone_id; геометрии зон разбираются из вложенных структур геоданных.')
    doc.add_heading('4. Получение, обработка и качество данных',level=1)
    add_doc_paragraph(doc,'Месячный Parquet скачивается один раз, запись начинается в .part и после успешной загрузки атомарно переименовывается; предусмотрены таймауты, три попытки HTTP и журнал SUCCESS / FAILED / SKIPPED. Повторный импорт уже успешного месяца пропускается. Агрегации в SQLite используют уникальный ключ (date, zone_id). Погода догружается только за ещё отсутствующие дни.')
    add_doc_paragraph(doc,'Ошибочные записи поездок, даты которых не относятся к месяцу исходного файла, а также недопустимые геозоны исключаются. Для полного ряда зона–дата формируется календарная сетка, отсутствующие суточные счётчики трактуются как ноль. Погода обрабатывается преобразованием типов, удалением отрицательных осадков и ветра, каузальным заполнением последним известным значением. Проверяются неотрицательность, полнота, уникальность, размер истории и подозрительные выбросы выше пяти медиан. Подозрительные реальные пики спроса только фиксируются, но не удаляются автоматически.')
    doc.add_heading('5. Формирование признаков и защита от утечки',level=1)
    add_doc_paragraph(doc,'Используются 21 исходный признак модели: зона и Borough (one-hot), день недели, месяц, число месяца, выходной, синус/косинус недели и месяца; лаги спроса 1/7/14/28; скользящие средние 7/28, стандартное отклонение и минимум за 7 суток; вчерашняя температура, осадки и ветер.')
    add_doc_paragraph(doc,'Для даты t все лаги и окна спроса берутся максимум по t−1: lag_k(t)=y(t−k), rolling_mean_7(t)=среднее y(t−7)…y(t−1). Погодные признаки также используют только t−1, исключая фактическую погоду прогнозируемого дня. Расчёт признаков един для обучения, мониторинга и прогноза: функция build_features. Тест искусственно меняет значение спроса y(t) и подтверждает неизменность признаков этой же строки.')
    doc.add_heading('6. Модели и воспроизводимая валидация',level=1)
    add_doc_paragraph(doc,'Базовый метод SeasonalNaive повторяет число поездок в тот же день предыдущей недели. Для сравнения применяются Ridge (линейная регуляризованная регрессия) и HistGradientBoostingRegressor (нелинейный бустинг по деревьям). Данные разделяются по дате: 60% train, 20% validation и 20% holdout, без случайного перемешивания. Подбор ограничен двумя конфигурациями на класс и тремя расширяющимися временными фолдами. Фиксируется random seed 42. Победитель выбирается по WAPE валидации; holdout для финального отчёта не участвует в подборе.')
    add_doc_paragraph(doc,'Метрики: MAE = среднее |y−ŷ|, RMSE = корень из среднего (y−ŷ)², WAPE = Σ|y−ŷ| / Σ|y|. MAE выражена в понятных пользователю единицах — ошибке количества поездок на зону за день.')
    add_doc_table(doc,['Модель','WAPE val','MAE holdout','RMSE holdout','WAPE holdout'],[
      [key, f"{DATA['models'][key]['validation']['wape']:.2%}",
       f"{DATA['models'][key]['test']['mae']:.2f}",
       f"{DATA['models'][key]['test']['rmse']:.2f}",
       f"{DATA['models'][key]['test']['wape']:.2%}"]
      for key in ['SeasonalNaive','Ridge','HistGradientBoosting']
    ])
    doc.add_paragraph().add_run().add_picture(str(DOCS/'metrics_comparison.png'),width=Cm(14.5))
    add_doc_paragraph(doc,f'Реальный эксперимент: train {ru_date(TRAIN_START)}–{ru_date(TRAIN_END)}, validation {ru_date(VAL_START)}–{ru_date(VAL_END)}, holdout {ru_date(TEST_START)}–{ru_date(TEST_END)}. Модель, выбранная по валидации, — {WINNER}; на holdout MAE = {metric(WINNER_TEST["mae"])}, RMSE = {metric(WINNER_TEST["rmse"])}, WAPE = {percentage(WINNER_TEST["wape"])}. На отложенном периоде оценено {WINNER_TEST["n"]} наблюдений «зона–день».')
    doc.add_heading('7. Компоненты по выбору: интерпретация, мониторинг, эксперименты',level=1)
    doc.add_heading('7.1. Интерпретируемость',level=2)
    add_doc_paragraph(doc,'Первый метод оценки важности — перестановка признака на validation с вычислением прироста MAE. Второй независимый метод — исключение признака, переобучение модели и оценка ухудшения MAE (абляция). Дополнительно реализован partial dependence plot, сравнение важностей в ранней и поздней частях validation, локальная последовательная аддитивная декомпозиция конкретного прогноза. Последняя точно суммируется в прогноз, но зависит от порядка замены признаков и не является SHAP или причинным объяснением.')
    doc.add_heading('7.2. Мониторинг качества и версии',level=2)
    add_doc_paragraph(doc,'Для лагов lag_1 и lag_7 рассчитывается PSI между последними 14 днями и предшествующими 42 днями; погода и скользящее среднее показываются диагностически. Дополнительно оценивается WAPE по последним известным фактам. Правило рекомендации переобучения: PSI > 0,4 или WAPE_14 > max(0,2; 1,35 × WAPE_holdout). Есть моделирование дрейфа без изменения базы. Версии модели регистрируются с параметрами, признаками и метриками. После первого обучения активируется первая версия, остальные — только при улучшении WAPE не менее чем на 2% на общем проверочном периоде. Предусмотрен ручной откат.')
    doc.add_heading('7.3. Автоматический подбор и журнал экспериментов',level=2)
    add_doc_paragraph(doc,'Журнал сохраняет ID запуска, время, версию данных SHA-256, random seed, класс модели, параметры, выборку/фолд, MAE, RMSE и WAPE. Команда reproduce ID повторяет обучение на прежнем календарном разбиении и сравнивает результаты. Для повторения необходим неизменный исходный набор данных. На реальных данных проверено воспроизведение эксперимента Ridge: исходный и повторный WAPE совпали (расхождение 0).')
    doc.add_heading('8. Дашборд и практическое использование',level=1)
    add_doc_paragraph(doc,'На панели представлены фактическое и прогнозное число поездок, рейтинг наиболее востребованных зон, динамика выбранного района, карта геозон, показатели качества, сравнение моделей, объяснение отдельного прогноза и журналы. Прогноз на 7 дней строится рекурсивно: с каждым будущим днём ранее полученные предсказания становятся историческими лагами. При невозможности узнать будущую погоду используется последнее известное наблюдение; эта аппроксимация отражена в ограничениях.')
    doc.add_heading('9. Технологии, воспроизводимость и испытания',level=1)
    add_doc_paragraph(doc,'Стек: Python 3.11–3.13, pandas, NumPy, scikit-learn, PyArrow, Requests, pyshp, SQLite, joblib, Streamlit, Plotly, pytest, Docker Compose. Развёртывание одной цепочкой: make all, интерфейс: streamlit run app.py. Конфигурируются директории и число районов через TAXI_DATA_DIR, TAXI_ARTIFACT_DIR, TAXI_MAX_ZONES.')
    add_doc_paragraph(doc,'Проверено 7 тестов pytest: инкрементальность, отсутствие утечки, хронология валидации, обучение и прогноз, точное воспроизведение, имитация дрейфа, структура URL источников. Результат: 7/7 успешны. Дополнительно в WSL выполнена реальная загрузка шести месячных файлов TLC, 182 дней погоды и геозон; обучены три модели, сформирован семидневный прогноз на 30 зон (210 записей), проверены воспроизведение эксперимента, имитация дрейфа, регистрация версий и ручной откат. Отсутствие дубликатов при повторной загрузке требует отдельной проверки SQL.')
    doc.add_heading('10. Выводы и ограничения',level=1)
    add_doc_paragraph(doc,f'Разработан и проверен на реальных исторических данных прототип полного цикла анализа и прогнозирования спроса по зонам такси, включая контроль качества, три модели, интерактивную панель, интерпретацию, мониторинг и воспроизводимые эксперименты. По WAPE валидации выбрана модель {WINNER}; её WAPE на независимом holdout составила {percentage(WINNER_TEST["wape"])}. Прогноз построен на 7 дней для 30 зон. Это ретроспективная оценка, а не гарантия точности будущих прогнозов.')
    add_doc_paragraph(doc,'Ограничения: исходные данные TLC публикуются с задержкой; одна городская точка погоды не отражает микроклимат зон; будущая погода заменяется последним известным наблюдением; качество рекурсивного прогноза на 7 дней может быть ниже оценки one-step holdout; учитывается только жёлтое такси; в реальном эксперименте анализ ограничен 30 зонами и шестью месяцами; прототип не является промышленной диспетчерской системой.')
    doc.add_heading('11. Источники',level=1)
    for line in [
        'NYC TLC. TLC Trip Record Data. https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page',
        'NYC Open Data. NYC Taxi Zones. https://data.cityofnewyork.us/d/8meu-9t5y',
        'Open-Meteo. Historical Weather API. https://open-meteo.com/en/docs/historical-weather-api',
        'Методические указания к проектной работе по дисциплине «Проектный практикум», 8 с., предоставлены студентом.'
    ]:
        add_doc_paragraph(doc,line)
    doc.add_heading('Приложение. Команды запуска',level=1)
    for line in ['python -m taxi.cli sync --start 2024-01 --months 6','python -m taxi.cli train','python -m taxi.cli forecast --days 7','python -m taxi.cli monitor --simulate-drift','python -m taxi.cli registry','python -m pytest -q','streamlit run app.py']:
        add_doc_paragraph(doc,line)
    doc.add_heading('Приложение. Состав хранилища',level=1)
    add_doc_table(doc,['Таблица SQLite','Назначение'],[
        ['zones','ID, Borough, название и геометрия зоны'],
        ['trips','Суточный агрегат поездок по уникальной паре дата/зона'],
        ['weather','Суточные признаки погоды'],
        ['ingestion_log / quality_log','Итоги загрузок и проверки качества'],
        ['experiments / models','Эксперименты, параметры, реестр и активная версия'],
        ['forecasts / backtest','Будущие прогнозы и результаты отложенной проверки'],
        ['feature_importance / monitoring','Важность признаков и журнал дрейфа']
    ])
    doc.add_heading('Приложение. Контроль выполнения требований',level=1)
    add_doc_paragraph(doc,'Источники: минимум 2 разных формата + геометрии GeoJSON. Данные: инкрементальная загрузка, идемпотентное повторение, журнал и явные ошибки. Признаки: единый код, временные лаги и окна, one-hot, контроль leakage. Обучение: 3 модели разных классов, минимум 2 метрики, time split и seed. Аналитика: сравнение факта/прогноза и качества. Дополнительно: два способа важности, локальная интерпретация, PDP, PSI, реестр версий, подбор параметров, журнал и повтор экспериментов. Документы: README, архитектура, отчёт, презентация.')
    add_doc_paragraph(doc,f'Реальный режим проверен на данных за {SOURCE_PERIOD}; выбрана модель {WINNER}, её WAPE holdout — {percentage(WINNER_TEST["wape"])}. Отдельно следует подтвердить SQL-проверкой отсутствие дублей после повторной загрузки, а также визуально проверить готовые PDF и работу локального объяснения прогноза.')
    doc.save(DOCS/'Отчет_Вариант16.docx')


SLIDE_W,SLIDE_H=13.333,7.5

def txt(slide,text,x,y,w,h,size=18,color='FFFFFF',bold=False):
    box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=box.text_frame
    tf.word_wrap=True
    for index,line in enumerate(text.split('\n')):
        p=tf.paragraphs[0] if index==0 else tf.add_paragraph()
        p.text=line
        p.font.name='DejaVu Sans'
        p.font.size=SlidePt(size)
        p.font.bold=bold
        p.font.color.rgb=SlideColor.from_string(color)
    return box

def panel(slide,x,y,w,h,color='17314A',radius=True):
    shape=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb=SlideColor.from_string(color)
    shape.line.fill.background()
    return shape

def slide_start(prs,title,subtitle=''):
    slide=prs.slides.add_slide(prs.slide_layouts[6])
    panel(slide,0,0,SLIDE_W,SLIDE_H,'0E1D32',False)
    txt(slide,title,.67,.48,12,.65,30,'FFFFFF',True)
    panel(slide,.68,1.25,1.3,.055,'0CB7A5',False)
    if subtitle: txt(slide,subtitle,.69,1.41,12,.48,13,'9FADBE')
    txt(slide,'TAXIDEMAND NYC  ·  ПРОЕКТНЫЙ ПРАКТИКУМ  ·  №16',.7,7.14,11,.25,8,'9FADBE')
    return slide

def bullet(slide,items,x,y,w,size=18,line_height=.66):
    for i,item in enumerate(items):
        txt(slide,'•',x,y+i*line_height,.32,.36,size,'0CB7A5',True)
        txt(slide,item,x+.39,y+i*line_height,w-.39,line_height-.04,size,'EAF2F5')


def build_pptx():
    prs=Presentation()
    prs.slide_width=Inches(SLIDE_W)
    prs.slide_height=Inches(SLIDE_H)
    s=prs.slides.add_slide(prs.slide_layouts[6])
    panel(s,0,0,SLIDE_W,SLIDE_H,'0E1D32',False)
    panel(s,.72,1,1.75,.09,'0CB7A5',False)
    txt(s,'TAXIDEMAND\nNYC',.75,1.4,11,2.1,48,'FFFFFF',True)
    txt(s,'Прогноз спроса на поездки такси по районам города',.78,4.03,11,1.0,23,'CDE8EF')
    txt(s,'Проектный практикум · Тема №16 · Реальные данные NYC TLC',.78,5.55,11,.5,15,'9FADBE')
    panel(s,.76,6.35,8.0,.46,'17314A')
    txt(s,'ЯНВАРЬ–ИЮНЬ 2024 · 30 ЗОН · 182 ДНЯ',.95,6.43,7.5,.25,10,'F0AD4E',True)

    s=slide_start(prs,'01  Задача и практическая ценность','Решение ориентировано на аналитика и диспетчера автопарка')
    panel(s,.7,2,5.8,3.95)
    txt(s,'Что прогнозируем?',1.0,2.35,5,.5,24,'0CB7A5',True)
    txt(s,'Число начинающихся поездок жёлтого такси\nпо зонам NYC TLC за сутки',1.0,3.0,5,1.6,21)
    panel(s,6.8,2,5.8,3.95)
    txt(s,'Какое решение?',7.1,2.35,5,.5,24,'F0AD4E',True)
    txt(s,'В каких районах ожидается повышенный спрос\nи как распределить автомобили',7.1,3.0,5,1.6,21)

    s=slide_start(prs,'02  Источники данных','Три разных формата, включая слабоструктурированные геоданные')
    for i,(top,bottom,accent) in enumerate([
        ('NYC TLC','Parquet · дата поездки + зона отправления','0CB7A5'),
        ('Open-Meteo','JSON · температура, осадки и ветер','F0AD4E'),
        ('NYC Taxi Zones','GeoJSON / SHP · полигоны районов','6CB1E1')]):
        x=.8+i*4.18
        panel(s,x,2.2,3.84,3.15)
        panel(s,x+.3,2.55,.7,.07,accent,False)
        txt(s,top,x+.3,2.95,3.25,.55,23,'FFFFFF',True)
        txt(s,bottom,x+.3,3.7,3.25,1.2,16,'CFD9E5')
    txt(s,'Реальная загрузка проверена: 6 месяцев TLC, 182 дня погоды и геоданные зон. Синтетический режим доступен отдельно.',.81,6.04,11.8,.58,13,'F0AD4E')

    s=slide_start(prs,'03  Архитектура сквозного решения','Из источников данных до прогноза и принятия решения')
    s.shapes.add_picture(str(DOCS/'architecture.png'),Inches(1.67),Inches(1.95),height=Inches(4.92))

    s=slide_start(prs,'04  Обработка и защита от утечки','Единый расчёт признаков для обучения и эксплуатации')
    bullet(s,[
        'Полная сетка «район × день», проверка пропусков и аномалий',
        'Лаги спроса: 1, 7, 14, 28 дней',
        'Оконные показатели: средние 7/28, разброс и минимум 7',
        'Календарные признаки и one-hot кодирование зон',
        'Погода только за предыдущий день: без знания будущего',
        'Отдельный тест: изменение y(t) не влияет на X(t)'],.8,2.0,11.75,18,.72)

    s=slide_start(prs,'05  Три прогнозные модели','Одинаковые временные периоды и две основные метрики плюс WAPE')
    panel(s,.73,2.03,3.83,2.45)
    panel(s,4.73,2.03,3.83,2.45)
    panel(s,8.73,2.03,3.83,2.45)
    for x,header,detail in [(.98,'SeasonalNaive','То же число поездок\n7 дней назад'),(4.97,'Ridge','Линейная модель\nс регуляризацией'),(8.97,'Boosting (HGB)','Нелинейные деревья\nс бустингом')]:
        txt(s,header,x,2.35,3.25,.7,20,'0CB7A5',True)
        txt(s,detail,x,3.1,3.3,1.0,16)
    txt(s,'Разбиение: train 60%  →  validation 20%  →  holdout 20%',.8,5.12,11,.55,21,'FFFFFF',True)
    txt(s,'Подбор: 2 конфигурации × 3 хронологических фолда; seed = 42',.8,5.84,11,.48,16,'CFD9E5')

    s=slide_start(prs,'06  Результаты сравнения',f'NYC TLC, {SOURCE_PERIOD}; {ZONE_COUNT} зон, {SOURCE_DAYS} дня')
    s.shapes.add_picture(str(DOCS/'metrics_comparison.png'),Inches(.88),Inches(1.94),width=Inches(8.7))
    panel(s,9.7,2.26,2.83,3.08)
    txt(s,percentage(WINNER_TEST['wape']),10.04,2.85,2.15,.95,33,'0CB7A5',True)
    txt(s,f'WAPE holdout\n{WINNER}',10.0,3.84,2.4,1.3,15)
    txt(s,f'MAE: {metric(WINNER_TEST["mae"])} поездки/зона/день · RMSE: {metric(WINNER_TEST["rmse"])}',.84,6.5,11,.37,15,'CFD9E5')

    s=slide_start(prs,'07  Интерпретируемость','Объяснение не только глобального качества, но и любого выбранного прогноза')
    bullet(s,[
        'Permutation importance: насколько ухудшится MAE после перемешивания',
        'Анализ абляции: исключить признак и переобучить модель',
        'Локальная декомпозиция прогноза: вклад каждой переменной',
        'Partial Dependence Plot для отдельного числового признака',
        'Сравнение важностей в двух частях валидационного периода'],.82,2.13,11.7,18,.81)

    s=slide_start(prs,'08  Мониторинг и воспроизводимость','Система фиксирует изменение поведения и хранит эксперименты')
    panel(s,.77,2.01,5.73,3.74)
    txt(s,'Контроль качества',1.05,2.36,5,.5,23,'0CB7A5',True)
    bullet(s,['PSI лагов спроса > 0,4','WAPE за 14 дней выше порога','Имитация дрейфа без порчи базы','Реестр моделей и откат'],1.05,3.05,5.03,16,.54)
    panel(s,6.75,2.01,5.73,3.74)
    txt(s,'Журнал экспериментов',7.04,2.36,5,.5,23,'F0AD4E',True)
    bullet(s,['2 конфигурации × 3 time folds','ID, параметры, seed, SHA-256','MAE / RMSE / WAPE','Воспроизведение по ID'],7.04,3.05,5.03,16,.54)

    s=slide_start(prs,'09  Демонстрация работающего прототипа','Streamlit и CLI: полный цикл без ручного копирования файлов')
    for i,(number,header,detail) in enumerate([
        ('01','Данные','Загрузка и журнал ошибок'),
        ('02','Обучение','3 модели + validation / holdout'),
        ('03','Прогноз','7 дней, разрез по зонам'),
        ('04','Контроль','Объяснение, PSI, реестр')]):
        y=1.9+i*1.11
        panel(s,.85,y,11.5,.86,'17314A')
        txt(s,number,1.12,y+.15,1.0,.45,23,'0CB7A5',True)
        txt(s,header,2.05,y+.17,2.55,.44,20,'FFFFFF',True)
        txt(s,detail,5.02,y+.19,7.0,.45,16,'CFD9E5')
    txt(s,'Тесты: 7 / 7 успешно · команда: make all',.92,6.68,11,.3,16,'F0AD4E',True)

    s=slide_start(prs,'10  Выводы и честные ограничения','Прототип готов для автономной демонстрации')
    bullet(s,[
        'Реализован полный цикл от источника до аналитического решения',
        'Есть три модели, интерпретация, мониторинг и повторение экспериментов',
        f'На реальных данных выбрана {WINNER}, WAPE holdout {percentage(WINNER_TEST["wape"])}',
        'Загружены 6 месяцев TLC, погода и геоданные зон',
        'Ограничения: 30 зон, 6 месяцев, рекурсивный прогноз и будущая погода'],.86,2.05,11.65,18,.78)
    prs.save(DOCS/'Презентация_Вариант16.pptx')


if __name__=='__main__':
    DOCS.mkdir(parents=True,exist_ok=True)
    build_architecture()
    build_metrics_chart()
    build_docx()
    build_pptx()
    print('Created architecture.png, metrics_comparison.png, report.docx and slides.pptx')
