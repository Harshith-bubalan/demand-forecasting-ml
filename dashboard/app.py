# ============================================================
# PRODUCT DEMAND FORECASTING — ADVANCED STREAMLIT DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DemandIQ | Product Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM UI
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f6f8fb;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #0b1220;
}

[data-testid="stSidebar"] * {
    color: #e8eef8 !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: #2563eb;
    border: none;
    color: white !important;
    font-weight: 700;
    border-radius: 10px;
}

.hero {
    background: linear-gradient(135deg, #0b1220 0%, #172554 55%, #2563eb 100%);
    padding: 28px 32px;
    border-radius: 20px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 12px 35px rgba(15, 23, 42, .14);
}

.hero h1 {
    margin: 0;
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -1px;
}

.hero p {
    margin: 8px 0 0 0;
    color: #cbd5e1;
    font-size: 15px;
}

.badge {
    display: inline-block;
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.18);
    padding: 5px 10px;
    border-radius: 999px;
    font-size: 12px;
    margin-bottom: 12px;
}

.kpi {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 18px 20px;
    min-height: 112px;
    box-shadow: 0 5px 18px rgba(15, 23, 42, .05);
}

.kpi-label {
    color: #64748b;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .06em;
}

.kpi-value {
    color: #0f172a;
    font-size: 28px;
    font-weight: 800;
    margin-top: 5px;
}

.kpi-sub {
    color: #64748b;
    font-size: 12px;
    margin-top: 4px;
}

.section-title {
    color: #0f172a;
    font-size: 20px;
    font-weight: 800;
    margin: 25px 0 10px;
}

.insight {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 10px;
}

.insight strong {
    color: #0f172a;
}

.small-note {
    color: #64748b;
    font-size: 12px;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e5e7eb;
    padding: 15px;
    border-radius: 14px;
    box-shadow: 0 5px 18px rgba(15, 23, 42, .04);
}

div[data-testid="stTabs"] button {
    font-weight: 700;
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent

DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
MODEL_PATH = PROJECT_ROOT / "notebooks" / "demand_forecasting_model.pkl"

# ============================================================
# EXPECTED FEATURES
# ============================================================

FEATURES_V2 = [
    "store_id", "sku_id", "total_price", "base_price",
    "discount_pct", "price_ratio", "is_featured_sku",
    "is_display_sku", "year", "month", "week_of_year",
    "lag_1", "lag_2", "lag_4", "lag_8", "lag_13", "lag_26",
    "rolling_mean_4", "rolling_mean_8", "rolling_std_4",
    "rolling_std_8", "demand_trend"
]

# ============================================================
# DATA + MODEL
# ============================================================

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_model(path):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_data(path):
    data = pd.read_csv(path)

    data["week"] = pd.to_datetime(data["week"], dayfirst=True)

    data["total_price"] = (
        data["total_price"]
        .fillna(data["total_price"].median())
    )

    data = data.sort_values(
        ["store_id", "sku_id", "week"]
    ).reset_index(drop=True)

    return data


@st.cache_data(show_spinner=False)
def prepare_data(data):
    data = data.copy()

    data["year"] = data["week"].dt.year
    data["month"] = data["week"].dt.month
    data["week_of_year"] = (
        data["week"].dt.isocalendar().week.astype(int)
    )

    group = data.groupby(["store_id", "sku_id"])["units_sold"]

    data["lag_1"] = group.shift(1)
    data["lag_2"] = group.shift(2)
    data["lag_4"] = group.shift(4)
    data["lag_8"] = group.shift(8)
    data["lag_13"] = group.shift(13)
    data["lag_26"] = group.shift(26)

    grouped = data.groupby(["store_id", "sku_id"])["units_sold"]

    data["rolling_mean_4"] = grouped.transform(
        lambda x: x.shift(1).rolling(4).mean()
    )
    data["rolling_mean_8"] = grouped.transform(
        lambda x: x.shift(1).rolling(8).mean()
    )
    data["rolling_std_4"] = grouped.transform(
        lambda x: x.shift(1).rolling(4).std()
    )
    data["rolling_std_8"] = grouped.transform(
        lambda x: x.shift(1).rolling(8).std()
    )

    data["demand_trend"] = data["lag_1"] - data["lag_4"]

    data["discount_pct"] = np.where(
        data["base_price"] != 0,
        ((data["base_price"] - data["total_price"])
         / data["base_price"]) * 100,
        0,
    )

    data["price_ratio"] = np.where(
        data["base_price"] != 0,
        data["total_price"] / data["base_price"],
        1,
    )

    return data


try:
    MODEL_PACKAGE = load_model(MODEL_PATH)
except Exception as e:
    st.error("Failed to load the trained model.")
    st.exception(e)
    st.stop()

if isinstance(MODEL_PACKAGE, dict):
    if "model" not in MODEL_PACKAGE:
        st.error("The saved model file does not contain a 'model' key.")
        st.stop()
    model = MODEL_PACKAGE["model"]
else:
    model = MODEL_PACKAGE

try:
    model_feature_names = list(model.feature_names_in_)
    if model_feature_names != FEATURES_V2:
        st.error("The saved model does not match the expected V2 feature set.")
        st.write("Model features:", model_feature_names)
        st.write("Expected features:", FEATURES_V2)
        st.stop()
except AttributeError:
    pass

df = prepare_data(load_data(DATA_PATH))

# ============================================================
# FORECAST ENGINE
# ============================================================

def forecast_demand(
    store_id,
    sku_id,
    weeks_ahead,
    price_multiplier=1.0,
    featured_override=None,
    display_override=None,
):
    history = df[
        (df["store_id"] == store_id) &
        (df["sku_id"] == sku_id)
    ].sort_values("week").copy()

    if len(history) < 26:
        raise ValueError(
            "Not enough historical data for this Store-SKU combination."
        )

    forecasts = []

    for _ in range(weeks_ahead):
        next_week = history["week"].max() + pd.Timedelta(weeks=1)

        demand = history["units_sold"].astype(float).values

        lag_1 = demand[-1]
        lag_2 = demand[-2]
        lag_4 = demand[-4]
        lag_8 = demand[-8]
        lag_13 = demand[-13]
        lag_26 = demand[-26]

        rolling_mean_4 = np.mean(demand[-4:])
        rolling_mean_8 = np.mean(demand[-8:])
        rolling_std_4 = np.std(demand[-4:], ddof=1)
        rolling_std_8 = np.std(demand[-8:], ddof=1)

        demand_trend = lag_1 - lag_4

        latest = history.iloc[-1]

        base_price = float(latest["base_price"])
        total_price = float(latest["total_price"]) * price_multiplier

        is_featured = (
            latest["is_featured_sku"]
            if featured_override is None
            else int(featured_override)
        )

        is_display = (
            latest["is_display_sku"]
            if display_override is None
            else int(display_override)
        )

        if base_price != 0:
            discount_pct = (
                (base_price - total_price) / base_price
            ) * 100
            price_ratio = total_price / base_price
        else:
            discount_pct = 0
            price_ratio = 1

        future_row = pd.DataFrame([{
            "store_id": store_id,
            "sku_id": sku_id,
            "total_price": total_price,
            "base_price": base_price,
            "discount_pct": discount_pct,
            "price_ratio": price_ratio,
            "is_featured_sku": is_featured,
            "is_display_sku": is_display,
            "year": next_week.year,
            "month": next_week.month,
            "week_of_year": int(next_week.isocalendar().week),
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_4": lag_4,
            "lag_8": lag_8,
            "lag_13": lag_13,
            "lag_26": lag_26,
            "rolling_mean_4": rolling_mean_4,
            "rolling_mean_8": rolling_mean_8,
            "rolling_std_4": rolling_std_4,
            "rolling_std_8": rolling_std_8,
            "demand_trend": demand_trend,
        }])

        future_X = future_row[FEATURES_V2]
        prediction = float(model.predict(future_X)[0])
        prediction = max(0, prediction)

        forecasts.append({
            "week": next_week,
            "predicted_demand": prediction,
        })

        new_row = latest.copy()
        new_row["week"] = next_week
        new_row["units_sold"] = prediction
        new_row["total_price"] = total_price
        new_row["is_featured_sku"] = is_featured
        new_row["is_display_sku"] = is_display

        history = pd.concat(
            [history, pd.DataFrame([new_row])],
            ignore_index=True,
        )

    return pd.DataFrame(forecasts)


# ============================================================
# HELPERS
# ============================================================

def fmt_num(value):
    return f"{value:,.0f}"


def pct_change(new, old):
    if old == 0:
        return 0
    return ((new - old) / old) * 100


def plot_layout(fig, height=440):
    fig.update_layout(
        height=height,
        template="plotly_white",
        margin=dict(l=10, r=10, t=45, b=10),
        font=dict(family="Inter, sans-serif", color="#334155"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e5e7eb")
    return fig


# ============================================================
# SIDEBAR
# ============================================================

stores = sorted(df["store_id"].dropna().unique())

with st.sidebar:
    st.markdown("## 📦 DemandIQ")
    st.caption("AI-powered product demand intelligence")
    st.divider()

    page = st.radio(
        "Workspace",
        ["Executive Dashboard", "Forecast Studio", "Demand Analytics", "Inventory Planner"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### Forecast Controls")

    selected_store = st.selectbox(
        "Store",
        stores,
        format_func=lambda x: f"Store {x}",
    )

    available_skus = sorted(
        df[df["store_id"] == selected_store]["sku_id"]
        .dropna()
        .unique()
    )

    selected_sku = st.selectbox(
        "SKU",
        available_skus,
        format_func=lambda x: f"SKU {x}",
    )

    weeks_ahead = st.slider(
        "Forecast horizon",
        min_value=1,
        max_value=13,
        value=4,
    )

    generate = st.button(
        "🔮 Generate Forecast",
        use_container_width=True,
    )

    st.divider()

    st.caption(
        f"Dataset: {len(df):,} rows\n\n"
        f"Stores: {df['store_id'].nunique()}\n\n"
        f"SKUs: {df['sku_id'].nunique()}"
    )

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <div class="badge">MACHINE LEARNING • DEMAND INTELLIGENCE</div>
    <h1>Product Demand Intelligence</h1>
    <p>Forecast future demand, understand demand behaviour, and turn predictions into inventory decisions.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# CURRENT SELECTION DATA
# ============================================================

selection_history = df[
    (df["store_id"] == selected_store) &
    (df["sku_id"] == selected_sku)
].sort_values("week").copy()

if len(selection_history) < 26:
    st.error("This Store-SKU combination has fewer than 26 historical observations.")
    st.stop()

recent_8 = selection_history["units_sold"].tail(8)
recent_26 = selection_history["units_sold"].tail(26)

latest_demand = float(selection_history["units_sold"].iloc[-1])
recent_avg = float(recent_8.mean())
long_avg = float(recent_26.mean())
volatility = float(recent_26.std())
trend_pct = pct_change(recent_avg, long_avg)

# Generate on demand, but also keep result in session state.
if generate:
    try:
        forecast_result = forecast_demand(
            selected_store,
            selected_sku,
            weeks_ahead,
        )
        st.session_state["forecast_result"] = forecast_result
        st.session_state["forecast_store"] = selected_store
        st.session_state["forecast_sku"] = selected_sku
    except Exception as e:
        st.error("Forecast generation failed.")
        st.exception(e)

forecast = st.session_state.get("forecast_result")

# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if page == "Executive Dashboard":

    st.markdown('<div class="section-title">Business Snapshot</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">Latest Demand</div>
            <div class="kpi-value">{fmt_num(latest_demand)}</div>
            <div class="kpi-sub">units in latest observed week</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">8-Week Average</div>
            <div class="kpi-value">{fmt_num(recent_avg)}</div>
            <div class="kpi-sub">recent baseline demand</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        trend_sign = "+" if trend_pct >= 0 else ""
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">Demand Trend</div>
            <div class="kpi-value">{trend_sign}{trend_pct:.1f}%</div>
            <div class="kpi-sub">recent vs 26-week average</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">Volatility</div>
            <div class="kpi-value">{fmt_num(volatility)}</div>
            <div class="kpi-sub">26-week demand standard deviation</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Demand Overview</div>',
                unsafe_allow_html=True)

    chart_data = selection_history.tail(52)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chart_data["week"],
        y=chart_data["units_sold"],
        mode="lines",
        name="Historical Demand",
        line=dict(width=3),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.08)",
    ))

    fig.add_trace(go.Scatter(
        x=chart_data["week"],
        y=chart_data["units_sold"].rolling(8).mean(),
        mode="lines",
        name="8-Week Moving Average",
        line=dict(width=2, dash="dash"),
    ))

    fig.update_layout(
        title=f"Store {selected_store} • SKU {selected_sku}",
        xaxis_title="Week",
        yaxis_title="Units Sold",
    )

    st.plotly_chart(plot_layout(fig), use_container_width=True)

    if forecast is not None:
        total_forecast = float(forecast["predicted_demand"].sum())
        avg_forecast = float(forecast["predicted_demand"].mean())
        forecast_change = pct_change(avg_forecast, recent_avg)

        st.markdown('<div class="section-title">Latest Forecast</div>',
                    unsafe_allow_html=True)

        a, b, c = st.columns(3)
        a.metric("Forecast / Week", fmt_num(avg_forecast))
        b.metric(
            f"Next {weeks_ahead} Weeks",
            fmt_num(total_forecast),
        )
        b.caption(f"{forecast_change:+.1f}% vs recent 8-week average")
        c.metric(
            "Forecast End",
            forecast["week"].max().strftime("%d %b %Y"),
        )

        st.dataframe(
            forecast.assign(
                week=forecast["week"].dt.strftime("%d %b %Y"),
                predicted_demand=forecast["predicted_demand"].round(0),
            ).rename(columns={
                "week": "Week",
                "predicted_demand": "Predicted Demand",
            }),
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# FORECAST STUDIO
# ============================================================

elif page == "Forecast Studio":

    st.markdown('<div class="section-title">Forecast Studio</div>',
                unsafe_allow_html=True)

    st.info(
        "Generate a baseline forecast and test simple business scenarios "
        "without changing the trained model."
    )

    if forecast is None:
        st.warning("Choose your Store, SKU and horizon, then click Generate Forecast.")
    else:
        left, right = st.columns([1, 2])

        with left:
            st.markdown("#### Scenario Planner")

            price_change = st.slider(
                "Future price change",
                min_value=-30,
                max_value=30,
                value=0,
                step=5,
                format="%d%%",
            )

            featured = st.checkbox(
                "Featured SKU",
                value=bool(selection_history["is_featured_sku"].iloc[-1]),
            )

            display = st.checkbox(
                "Display promotion",
                value=bool(selection_history["is_display_sku"].iloc[-1]),
            )

            scenario_forecast = forecast_demand(
                selected_store,
                selected_sku,
                weeks_ahead,
                price_multiplier=1 + price_change / 100,
                featured_override=int(featured),
                display_override=int(display),
            )

            baseline_total = forecast["predicted_demand"].sum()
            scenario_total = scenario_forecast["predicted_demand"].sum()
            scenario_delta = pct_change(scenario_total, baseline_total)

            st.metric(
                "Scenario demand",
                f"{scenario_total:,.0f} units",
                f"{scenario_delta:+.1f}% vs baseline",
            )

            st.caption(
                "Scenario output is a model-based what-if estimate using "
                "the same trained model and altered future inputs."
            )

        with right:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=forecast["week"],
                y=forecast["predicted_demand"],
                mode="lines+markers",
                name="Baseline",
                line=dict(width=3),
            ))

            fig.add_trace(go.Scatter(
                x=scenario_forecast["week"],
                y=scenario_forecast["predicted_demand"],
                mode="lines+markers",
                name="Scenario",
                line=dict(width=3, dash="dash"),
            ))

            fig.update_layout(
                title="Baseline vs Scenario Forecast",
                xaxis_title="Week",
                yaxis_title="Predicted Units",
            )

            st.plotly_chart(plot_layout(fig, 480), use_container_width=True)

        st.markdown('<div class="section-title">Forecast Detail</div>',
                    unsafe_allow_html=True)

        comparison = pd.DataFrame({
            "Week": forecast["week"].dt.strftime("%d %b %Y"),
            "Baseline": forecast["predicted_demand"].round(0),
            "Scenario": scenario_forecast["predicted_demand"].round(0),
        })

        st.dataframe(comparison, use_container_width=True, hide_index=True)

        csv = comparison.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Forecast CSV",
            csv,
            file_name=f"forecast_store_{selected_store}_sku_{selected_sku}.csv",
            mime="text/csv",
        )

# ============================================================
# DEMAND ANALYTICS
# ============================================================

elif page == "Demand Analytics":

    st.markdown('<div class="section-title">Demand Analytics</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    c1.metric("Historical Records", f"{len(selection_history):,}")
    c2.metric("26-Week Mean", f"{long_avg:,.0f}")
    c3.metric("26-Week Std. Dev.", f"{volatility:,.0f}")

    tab1, tab2, tab3 = st.tabs(
        ["Seasonality", "Price & Promotions", "Demand Distribution"]
    )

    with tab1:
        monthly = (
            selection_history
            .assign(month=selection_history["week"].dt.month)
            .groupby("month")["units_sold"]
            .mean()
            .reset_index()
        )

        fig = go.Figure(go.Bar(
            x=monthly["month"],
            y=monthly["units_sold"],
            text=monthly["units_sold"].round(0),
            textposition="outside",
        ))

        fig.update_layout(
            title="Average Demand by Calendar Month",
            xaxis_title="Month",
            yaxis_title="Average Units",
        )

        st.plotly_chart(plot_layout(fig, 420), use_container_width=True)

        st.caption(
            "This view describes historical monthly demand behaviour; "
            "it is not a separate seasonal model."
        )

    with tab2:
        price_data = selection_history.copy()
        price_data["discount_pct"] = np.where(
            price_data["base_price"] != 0,
            ((price_data["base_price"] - price_data["total_price"])
             / price_data["base_price"]) * 100,
            0,
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=price_data["discount_pct"],
            y=price_data["units_sold"],
            mode="markers",
            marker=dict(size=8, opacity=.65),
            name="Observed Weeks",
        ))

        fig.update_layout(
            title="Observed Discount vs Demand",
            xaxis_title="Discount (%)",
            yaxis_title="Units Sold",
        )

        st.plotly_chart(plot_layout(fig, 420), use_container_width=True)

        promo_rate = (
            selection_history["is_featured_sku"].mean() * 100
        )

        st.metric(
            "Featured-SKU occurrence",
            f"{promo_rate:.1f}%",
        )

    with tab3:
        fig = go.Figure(go.Histogram(
            x=selection_history["units_sold"],
            nbinsx=25,
        ))

        fig.update_layout(
            title="Historical Demand Distribution",
            xaxis_title="Units Sold",
            yaxis_title="Frequency",
        )

        st.plotly_chart(plot_layout(fig, 420), use_container_width=True)

# ============================================================
# INVENTORY PLANNER
# ============================================================

else:

    st.markdown('<div class="section-title">Inventory Planner</div>',
                unsafe_allow_html=True)

    st.info(
        "This section converts the forecast into planning indicators. "
        "The inventory values are heuristic planning estimates, not "
        "outputs directly learned by the forecasting model."
    )

    lead_time = st.slider(
        "Supplier lead time",
        min_value=1,
        max_value=12,
        value=2,
        help="Estimated replenishment lead time in weeks.",
    )

    service_factor = st.slider(
        "Safety-stock factor",
        min_value=1.0,
        max_value=2.5,
        value=1.65,
        step=0.05,
        help="Approximate safety-stock multiplier.",
    )

    if forecast is None:
        st.warning("Generate a forecast first to populate inventory planning.")
    else:
        avg_future = float(forecast["predicted_demand"].mean())
        forecast_std = float(forecast["predicted_demand"].std())

        if np.isnan(forecast_std):
            forecast_std = 0

        safety_stock = service_factor * max(volatility, forecast_std)
        reorder_point = avg_future * lead_time + safety_stock
        suggested_cycle_stock = avg_future * 4
        projected_horizon = float(forecast["predicted_demand"].sum())

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Avg Forecast / Week", f"{avg_future:,.0f}")
        c2.metric("Safety Stock", f"{safety_stock:,.0f}")
        c3.metric("Reorder Point", f"{reorder_point:,.0f}")
        c4.metric("4-Week Cycle Stock", f"{suggested_cycle_stock:,.0f}")

        st.markdown('<div class="section-title">Planning View</div>',
                    unsafe_allow_html=True)

        planning = pd.DataFrame({
            "Planning Metric": [
                "Average forecast / week",
                "Forecast horizon demand",
                "Estimated safety stock",
                "Estimated reorder point",
                "Suggested 4-week cycle stock",
            ],
            "Units": [
                round(avg_future),
                round(projected_horizon),
                round(safety_stock),
                round(reorder_point),
                round(suggested_cycle_stock),
            ],
        })

        st.dataframe(
            planning,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            f"""
            <div class="insight">
                <strong>Inventory interpretation</strong><br>
                With a {lead_time}-week supplier lead time, the estimated
                reorder point is approximately <b>{reorder_point:,.0f} units</b>.
                This combines expected lead-time demand with a volatility-based
                safety-stock buffer.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "DemandIQ • ML-based demand forecasting dashboard • "
    "Forecasts should be combined with business and inventory context."
)
