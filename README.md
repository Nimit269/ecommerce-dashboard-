# E-commerce & Finance Intelligence Platform (Prototype)

## Run
    pip install -r requirements.txt
    streamlit run app.py

(`prophet` and `xgboost` are optional - the app falls back automatically if they fail to install.)

## Files
- `engine.py` - data generation, cleaning, features, ML ensemble, cash-flow, stock alerts, report
- `app.py`    - Streamlit dashboard

## Mapping to your slides
| Slide section        | Where in the prototype |
|----------------------|------------------------|
| Data sources         | Sidebar: sample data or upload orders / expenses / inventory CSV |
| Data processing      | `clean_data`, `daily_summary`, `make_features` (day, month, season, lags, moving averages) |
| ML engine            | `train_and_evaluate` (70/30 split, MAE & RMSE), `forecast_future` (XGBoost, RF, Linear Reg, Prophet averaged) |
| Cash-flow prediction | `cash_projection` + Cash Flow tab |
| Stock-out risk       | `stock_risk` + Stock tab |
| Dashboard            | Sales & profit charts, alerts, monthly report download |

## Using real data (UCI Online Retail / Olist)
Rename columns to date, product, quantity, price, customer_id and upload via the sidebar.
