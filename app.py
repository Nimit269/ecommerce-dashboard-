"""Streamlit dashboard: E-commerce & Finance Intelligence Platform (prototype)
Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st
import engine as E

st.set_page_config(page_title="E-commerce & Finance Intelligence", page_icon="📈", layout="wide")
st.title("📈 E-commerce & Finance Intelligence Platform")
st.caption("Sales forecast • Cash-flow prediction • Stock-out alerts — for small sellers")

# ---------------- Sidebar: 1. DATA SOURCES ----------------
st.sidebar.header("1. Data Sources")
source = st.sidebar.radio("Choose data", ["Use sample data (clothing seller)", "Upload my CSV files"])
if source.startswith("Use sample"):
    data = E.generate_sample_data()
    orders_raw, exp_raw, inv, customers = data["orders"], data["expenses"], data["inventory"], data["customers"]
else:
    st.sidebar.caption("orders.csv: date, product, quantity, price, customer_id\n\n"
                       "expenses.csv: date, category, amount\n\n"
                       "inventory.csv: product, stock, reorder_level, lead_time_days")
    f_o = st.sidebar.file_uploader("Orders CSV", type="csv")
    f_e = st.sidebar.file_uploader("Expenses CSV", type="csv")
    f_i = st.sidebar.file_uploader("Inventory CSV", type="csv")
    if not (f_o and f_e and f_i):
        st.info("Upload all three CSV files in the sidebar, or switch to the sample data.")
        st.stop()
    orders_raw, exp_raw, inv = pd.read_csv(f_o), pd.read_csv(f_e), pd.read_csv(f_i)
    if "customer_id" not in orders_raw: orders_raw["customer_id"] = "NA"

opening_cash = st.sidebar.number_input("Opening cash (Rs)", value=150000, step=10000)
safety_cash = st.sidebar.number_input("Cash safety limit (Rs)", value=50000, step=10000,
                                      help="You get an alert if projected cash falls below this.")
horizon = st.sidebar.slider("Forecast days", 14, 60, 30)

# ---------------- 2. DATA PROCESSING + 3. ML ENGINE ----------------
with st.spinner("Cleaning data and training models..."):
    orders, expenses = E.clean_data(orders_raw, exp_raw)
    daily = E.daily_summary(orders, expenses, opening_cash)
    metrics, y_test, ens_pred = E.train_and_evaluate(daily["revenue"])
    forecast = E.forecast_future(daily["revenue"], horizon)
    cash_df = E.cash_projection(daily, forecast, safety_cash)
    inv_risk = E.stock_risk(orders, inv, forecast, daily)
    alerts = E.build_alerts(cash_df, inv_risk, safety_cash)

# ---------------- 4. DASHBOARD ----------------
last30 = daily.tail(30)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Revenue (30d)", f"Rs {last30.revenue.sum():,.0f}")
c2.metric("Profit (30d)", f"Rs {last30.profit.sum():,.0f}")
c3.metric("Forecast revenue (next %d d)" % horizon, f"Rs {forecast.sum():,.0f}",
          f"{(forecast.head(30).sum() / last30.revenue.sum() - 1) * 100:+.1f}% vs last 30d")
c4.metric("Cash today", f"Rs {daily.cash.iloc[-1]:,.0f}")

st.subheader("🔔 Smart Alerts")
for level, msg in alerts:
    {"critical": st.error, "warning": st.warning, "ok": st.success}[level](msg)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Sales & Profit", "Sales Forecast", "Cash Flow", "Stock", "Model Accuracy"])

with tab1:
    weekly = daily[["revenue", "expenses", "profit"]].resample("W").sum().tail(26)
    st.markdown("**Weekly revenue, expenses and profit (last 26 weeks)**")
    st.line_chart(weekly)
    a, b = st.columns(2)
    a.markdown("**Top products by revenue (last 90 days)**")
    top = orders[orders.date > orders.date.max() - pd.Timedelta(days=90)].groupby("product")["revenue"].sum()
    a.bar_chart(top.sort_values(ascending=False))
    b.markdown("**Expenses by category (last 90 days)**")
    ex = expenses[expenses.date > expenses.date.max() - pd.Timedelta(days=90)].groupby("category")["amount"].sum()
    b.bar_chart(ex.sort_values(ascending=False))
    st.markdown("**Top customers**")
    cust = orders.groupby("customer_id").agg(orders=("date", "count"), spent=("revenue", "sum"))
    st.dataframe(cust.sort_values("spent", ascending=False).head(10), use_container_width=True)

with tab2:
    st.markdown("**Actual vs Forecast daily sales** (ensemble = average of all models)")
    actual = daily["revenue"].tail(60).rename("Actual")
    chart = pd.concat([actual, forecast.rename("Forecast")], axis=1)
    st.line_chart(chart)
    st.markdown("**Weekly forecast**")
    st.dataframe(forecast.resample("W").sum().rename("Forecast revenue (Rs)").round(0), use_container_width=True)

with tab3:
    st.markdown("**Projected cash balance**")
    proj = cash_df[["cash"]].rename(columns={"cash": "Projected cash"})
    proj["Safety limit"] = safety_cash
    st.line_chart(proj)
    wk = cash_df.groupby("week").agg(inflow=("revenue", "sum"), outflow=("expenses", "sum"),
                                     lowest_cash=("cash", "min")).round(0)
    st.dataframe(wk, use_container_width=True)

with tab4:
    st.markdown("**Stock-out risk** (days of stock left vs supplier lead time, using forecast demand)")
    show = inv_risk[["product", "stock", "daily_demand", "days_left", "lead_time_days", "risk", "reorder_qty"]].round(1)
    st.dataframe(show.style.map(lambda v: {"HIGH": "background-color:#f8c9c9", "MEDIUM": "background-color:#fbe8b0",
                                           "LOW": "background-color:#cfeccf"}.get(v, ""), subset=["risk"]),
                 use_container_width=True)
    st.bar_chart(inv_risk.set_index("product")["days_left"])

with tab5:
    st.markdown("**70% train / 30% test — lower MAE and RMSE is better**")
    st.dataframe(metrics.round(0), use_container_width=True)
    st.line_chart(pd.concat([y_test.rename("Actual"), ens_pred.rename("Ensemble")], axis=1))
    st.caption("Models: " + ", ".join(["XGBoost" if E.HAS_XGB else "GradientBoosting (XGBoost fallback)",
               "Random Forest", "Linear Regression", "Prophet" if E.HAS_PROPHET else "Prophet (not installed - skipped)"]))

st.divider()
report = E.monthly_report(daily, forecast, cash_df, inv_risk, alerts)
st.download_button("⬇️ Download monthly report", report, file_name="monthly_report.txt")
