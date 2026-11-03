# Архитектура TaxiDemand NYC

```mermaid
flowchart TB
  TLC[Официальные поездки TLC / Parquet] --> EX[Сбор данных, контроль ошибок, инкрементальность]
  MET[Open-Meteo / JSON] --> EX
  GEO[NYC Taxi Zones / GeoJSON + CSV] --> EX
  EX --> RAW[Исходные файлы: data/raw]
  EX --> SQL[(SQLite: trips / zones / weather / ingestion_log)]
  SQL --> QC[Контроль качества, пропуски, аномалии]
  QC --> FT[Витрина признаков: лаги / окна / календарь / погода]
  FT --> M[Naive / Ridge / HistGradientBoosting]
  M --> REG[(Модели и журнал экспериментов)]
  M --> EXP[Интерпретация: permutation, ablation, PDP, локальный вклад]
  M --> MON[Мониторинг: PSI + WAPE + правила отката]
  REG --> FC[Рекурсивный прогноз на 7 дней]
  FC --> UI[Streamlit + Plotly]
  EXP --> UI
  MON --> UI
  SQL --> UI
```

Потоки в одной системе, без искусственного создания микросервисов. Формат хранения исходных файлов сохраняется, показатели обрабатываются в SQLite. Признаки во время обучения и прогнозирования формируются одной функцией `build_features`.
