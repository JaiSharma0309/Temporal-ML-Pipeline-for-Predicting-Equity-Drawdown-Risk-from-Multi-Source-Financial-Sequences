.PHONY: setup pipeline train train-sequence visuals clean-results

setup:
	python -m pip install -r requirements.txt

pipeline:
	python src/download_yfinance_prices.py
	python src/build_modeling_dataset.py
	python src/fetch_short_interest.py
	python src/fetch_fundamentals.py --overwrite
	python src/train_drawdown_risk_models.py
	python src/model_visualizations.py

train:
	python src/train_drawdown_risk_models.py

train-sequence:
	python src/train_sequence_model_pytorch.py

visuals:
	python src/model_visualizations.py

clean-results:
	rm -f results/stage1/tables/*_predictions.csv
