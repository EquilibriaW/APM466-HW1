# APM466 HW1

This repo contains the scraping and analysis code for APM466 HW1, plus the collected data artifacts.

## Contents
- `scrape.py`: scrape and assemble bond data
- `analysis.py`: compute YTM/spot/forward curves, covariance matrices, PCA
- `scraped_data/`: raw/clean data matrices
- `analysis_outputs/`: generated curves, covariance matrices, PCA outputs, plots

## Run
```bash
python analysis.py --data scraped_data/bonds_final_price_matrix.csv
```
