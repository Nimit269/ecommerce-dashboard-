"""Core logic for the E-commerce & Finance Intelligence Platform (prototype).
Pipeline: data -> clean -> daily summary -> features -> ensemble ML -> alerts
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

PRODUCTS = {  # name: (price, base daily units, unit cost)
    "T-Shirt": (499, 5, 220), "Jeans": (1299, 3, 650), "Hoodie": (1599, 2, 800),
    "Kurta": (899, 4, 400), "Sneakers": (2199, 1.5, 1200), "Cap": (299, 3, 110),
}

# ---------------------------------------------------------------- 1. DATA
def generate_sample_data(days=420, seed=42):
    """Sample small clothing seller: orders, expenses, inventory, customers."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    dates = pd.date_range(end=end, periods=days)
    rows = []
    for d in dates:
        season = 1 + 0.25 * np.sin(2 * np.pi * (d.dayofyear - 80) / 365)
        weekend = 1.25 if d.dayofweek >= 5 else 1.0
        trend = 1 + 0.0008 * (d - dates[0]).days
        festive = 1.5 if d.month in (10, 11) else 1.0
        for p, (price, base, _) in PRODUCTS.items():
            n = rng.poisson(base * season * weekend * trend * festive)
            if n:
                rows.append((d, p, n, price, f"C{rng.integers(1, 400):04d}"))
    orders = pd.DataFrame(rows, columns=["date", "product", "quantity", "price", "customer_id"])
    orders["revenue"] = orders["quantity"] * orders["price"]

    exp = []
    daily_rev = orders.groupby("date")["revenue"].sum().reindex(dates, fill_value=0)
    for d in dates:
        exp.append((d, "Ads & Marketing", abs(rng.normal(900, 250))))
        exp.append((d, "Shipping", abs(rng.normal(700, 200))))
        if d.day == 1:
            exp.append((d, "Rent", 20000)); exp.append((d, "Salaries", 35000))
        if d.day == 25:  # owner drawings + loan EMI
            prev30 = daily_rev[(daily_rev.index < d) & (daily_rev.index >= d - pd.Timedelta(days=30))].sum()
            exp.append((d, "Owner Drawings & Loan EMI", prev30 * 0.30))
        if d.day in (5, 20):  # supplier payments = ~48% of the last 15 days' sales
            prev = daily_rev[(daily_rev.index < d) & (daily_rev.index >= d - pd.Timedelta(days=15))].sum()
            exp.append((d, "Inventory Purchase", prev * rng.normal(0.52, 0.04)))
    expenses = pd.DataFrame(exp, columns=["date", "category", "amount"])

    inventory = pd.DataFrame({
        "product": list(PRODUCTS),
        "stock": [int(v[1] * k) for v, k in zip(PRODUCTS.values(), [30, 26, 12, 20, 9, 40])],
        "reorder_level": [int(v[1] * 10) for v in PRODUCTS.values()],
        "lead_time_days": [7, 10, 10, 7, 14, 5],
        "unit_cost": [v[2] for v in PRODUCTS.values()],
    })
    customers = orders.groupby("customer_id").agg(
        orders=("date", "count"), spent=("revenue", "sum"), last_order=("date", "max")).reset_index()
    return {"orders": orders, "expenses": expenses, "inventory": inventory, "customers": customers}

def _classify(df):
    """Identify a table (orders / expenses / inventory) from its column names."""
    cols = set(df.columns)
    if {"date", "category", "amount"} <= cols: return "expenses"
    if {"product", "stock"} <= cols: return "inventory"
    if {"date", "product", "quantity"} <= cols: return "orders"
    return None

def load_combined(files):
    """One upload -> tables found. Accepts an Excel workbook (3 sheets) and/or several CSVs;
    each table is detected by its column names, so file/sheet names do not matter."""
    found = {}
    for f in files:
        f.seek(0)  # safe on Streamlit reruns
        if f.name.lower().endswith((".xlsx", ".xls")):
            frames = pd.read_excel(f, sheet_name=None)
        else:
            frames = {f.name: pd.read_csv(f)}
        for df in frames.values():
            df.columns = [str(c).strip().lower() for c in df.columns]
            kind = _classify(df)
            if kind: found[kind] = df
    missing = [k for k in ("orders", "expenses", "inventory") if k not in found]
    return found, missing

def build_template(rows=30):
    """Excel workbook with orders / expenses / inventory sheets (sample rows) to use as a template."""
    import io
    d = generate_sample_data(days=90)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        d["orders"][["date", "product", "quantity", "price", "customer_id"]].head(rows).to_excel(w, sheet_name="orders", index=False)
        d["expenses"].head(rows).to_excel(w, sheet_name="expenses", index=False)
        d["inventory"][["product", "stock", "reorder_level", "lead_time_days"]].to_excel(w, sheet_name="inventory", index=False)
    return buf.getvalue()

# ---------------------------------------------------------------- 2. PROCESSING
def clean_data(orders, expenses):
    o = orders.copy(); e = expenses.copy()
    o["date"] = pd.to_datetime(o["date"], errors="coerce")
    e["date"] = pd.to_datetime(e["date"], errors="coerce")
    o = o.drop_duplicates().dropna(subset=["date", "product"])
    e = e.drop_duplicates().dropna(subset=["date"])
    o["quantity"] = o["quantity"].fillna(o["quantity"].median()).clip(lower=0)
    o["price"] = o["price"].fillna(o.groupby("product")["price"].transform("median"))
    if "revenue" not in o:
        o["revenue"] = o["quantity"] * o["price"]
    e["amount"] = e["amount"].fillna(0).clip(lower=0)
    return o, e

def daily_summary(orders, expenses, opening_cash=150000):
    rev = orders.groupby("date").agg(revenue=("revenue", "sum"), units=("quantity", "sum"),
                                     orders=("quantity", "count"))
    exp = expenses.groupby("date")["amount"].sum().rename("expenses")
    df = rev.join(exp, how="outer").fillna(0)
    df = df.reindex(pd.date_range(df.index.min(), df.index.max()), fill_value=0)
    df.index.name = "date"
    df["profit"] = df["revenue"] - df["expenses"]
    df["cash"] = opening_cash + df["profit"].cumsum()
    return df

def make_features(s):
    """s = daily revenue series -> feature table (day, month, season, lags, moving averages)."""
    f = pd.DataFrame(index=s.index)
    f["dow"] = s.index.dayofweek
    f["day"] = s.index.day
    f["month"] = s.index.month
    f["season"] = (s.index.month % 12) // 3
    f["lag1"] = s.shift(1); f["lag7"] = s.shift(7)
    f["ma7"] = s.shift(1).rolling(7).mean(); f["ma14"] = s.shift(1).rolling(14).mean()
    return f

# ---------------------------------------------------------------- 3. ML ENGINE
def _models():
    m = {"Random Forest": RandomForestRegressor(200, random_state=1),
         "Linear Regression": LinearRegression()}
    m["XGBoost"] = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.08) if HAS_XGB \
        else GradientBoostingRegressor(random_state=1)  # fallback if xgboost not installed
    return m

def _prophet_fit(train):
    d = pd.DataFrame({"ds": train.index, "y": train.values})
    p = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    p.fit(d); return p

def train_and_evaluate(series):
    """70/30 time split, MAE & RMSE for every model + averaged ensemble."""
    feats = make_features(series).dropna(); y = series.loc[feats.index]
    cut = int(len(feats) * 0.7)
    Xtr, Xte, ytr, yte = feats.iloc[:cut], feats.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    preds = {}
    for name, m in _models().items():
        m.fit(Xtr, ytr); preds[name] = pd.Series(m.predict(Xte), index=yte.index)
    if HAS_PROPHET:
        p = _prophet_fit(ytr)
        fc = p.predict(pd.DataFrame({"ds": yte.index}))
        preds["Prophet"] = pd.Series(fc["yhat"].values, index=yte.index)
    preds["Ensemble (average)"] = pd.concat(preds.values(), axis=1).mean(axis=1)
    rows = [{"Model": n, "MAE": mean_absolute_error(yte, p),
             "RMSE": float(np.sqrt(mean_squared_error(yte, p)))} for n, p in preds.items()]
    table = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return table, yte, preds["Ensemble (average)"]

def forecast_future(series, horizon=30):
    """Retrain on all data, predict the next `horizon` days (recursive for lag features)."""
    feats = make_features(series).dropna(); y = series.loc[feats.index]
    models = _models()
    for m in models.values(): m.fit(feats, y)
    prophet = _prophet_fit(series) if HAS_PROPHET else None
    hist = series.copy(); out = []
    future_idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
    pf = prophet.predict(pd.DataFrame({"ds": future_idx}))["yhat"].values if prophet else None
    for i, d in enumerate(future_idx):
        tmp = pd.concat([hist, pd.Series([np.nan], index=[d])])
        row = make_features(tmp).iloc[[-1]]
        ps = [m.predict(row)[0] for m in models.values()]
        if pf is not None: ps.append(pf[i])
        val = max(float(np.mean(ps)), 0)
        out.append(val); hist.loc[d] = val
    return pd.Series(out, index=future_idx, name="forecast")

# ---------------------------------------------------------------- 4. CASH FLOW & STOCK
def cash_projection(daily, rev_forecast, safety_cash=50000, recurring=None):
    """Projected cash = current cash + forecast revenue - expected expenses."""
    # expected expenses: average of the same day-of-month over the last 3 months
    # (captures rent, salaries and supplier payments), floored by recent daily average
    hist = daily["expenses"].tail(95)
    by_dom = hist.groupby(hist.index.day).mean()
    idx = rev_forecast.index
    exp_f = pd.Series([by_dom.get(d.day, hist.mean()) for d in idx], index=idx)
    out = pd.DataFrame({"revenue": rev_forecast, "expenses": exp_f})
    out["cash"] = daily["cash"].iloc[-1] + (out["revenue"] - out["expenses"]).cumsum()
    out["week"] = ((np.arange(len(out)) // 7) + 1)
    out["below_safety"] = out["cash"] < safety_cash
    return out

def stock_risk(orders, inventory, rev_forecast, daily):
    """Days of stock left vs supplier lead time, scaled by the sales forecast."""
    last28 = orders[orders.date > orders.date.max() - pd.Timedelta(days=28)]
    avg_units = last28.groupby("product")["quantity"].sum() / 28
    recent_rev = daily["revenue"].tail(28).mean()
    growth = (rev_forecast.head(28).mean() / recent_rev) if recent_rev else 1
    inv = inventory.copy()
    inv["daily_demand"] = inv["product"].map(avg_units).fillna(0) * growth
    inv["days_left"] = np.where(inv["daily_demand"] > 0, inv["stock"] / inv["daily_demand"], 999)
    def level(r):
        if r.days_left <= r.lead_time_days: return "HIGH"
        if r.days_left <= r.lead_time_days + 7 or r.stock <= r.reorder_level: return "MEDIUM"
        return "LOW"
    inv["risk"] = inv.apply(level, axis=1)
    inv["reorder_qty"] = np.ceil(inv["daily_demand"] * (inv["lead_time_days"] + 30) - inv["stock"]).clip(lower=0)
    return inv.sort_values("days_left")

def build_alerts(cash_df, inv, safety_cash):
    alerts = []
    for _, r in inv.iterrows():
        if r.risk == "HIGH":
            alerts.append(("critical", f"Stock: {r['product']} will run out in ~{r.days_left:.0f} days, "
                           f"supplier needs {r.lead_time_days} days. Reorder ~{int(r.reorder_qty)} units now."))
        elif r.risk == "MEDIUM":
            alerts.append(("warning", f"Stock: {r['product']} is getting low (~{r.days_left:.0f} days left)."))
    low = cash_df[cash_df.below_safety]
    if not low.empty:
        alerts.append(("critical", f"Cash: balance expected to fall below Rs {safety_cash:,.0f} in week "
                       f"{int(low.week.iloc[0])} (lowest ~Rs {cash_df.cash.min():,.0f})."))
    else:
        alerts.append(("ok", "Cash: no cash shortage expected in the next 30 days."))
    return alerts

def monthly_report(daily, forecast, cash_df, inv, alerts):
    last30 = daily.tail(30)
    lines = ["MONTHLY REPORT", "=" * 40,
             f"Revenue (last 30 days): Rs {last30.revenue.sum():,.0f}",
             f"Expenses (last 30 days): Rs {last30.expenses.sum():,.0f}",
             f"Profit (last 30 days): Rs {last30.profit.sum():,.0f}",
             f"Forecast revenue (next 30 days): Rs {forecast.sum():,.0f}",
             f"Projected cash after 30 days: Rs {cash_df.cash.iloc[-1]:,.0f}", "", "ALERTS"]
    lines += [f"- [{lvl.upper()}] {msg}" for lvl, msg in alerts]
    lines += ["", "STOCK STATUS"]
    lines += [f"- {r['product']}: {int(r.stock)} units, {r.days_left:.0f} days left, risk {r.risk}"
              for _, r in inv.iterrows()]
    return "\n".join(lines)
