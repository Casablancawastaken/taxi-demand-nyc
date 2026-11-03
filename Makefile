.PHONY: install demo real train forecast monitor drift test app reset all
install:
	python -m pip install -r requirements.txt
demo:
	python -m taxi.cli demo --reset
real:
	python -m taxi.cli sync --start 2024-01 --months 6
train:
	python -m taxi.cli train
forecast:
	python -m taxi.cli forecast --days 7
monitor:
	python -m taxi.cli monitor
drift:
	python -m taxi.cli monitor --simulate-drift
test:
	python -m pytest -q
app:
	streamlit run app.py
reset:
	python -m taxi.cli reset
all: demo train forecast monitor test
