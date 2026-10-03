# ============================================================
# PRODUCT DEMAND FORECASTING DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Product Demand Forecaster",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

# app.py is inside:
# DEMAND-FORECASTING-ML/dashboard/app.py

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent

DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
MODEL_PATH = PROJECT_ROOT / "notebooks" / "demand_forecasting_model.pkl"


# ============================================================
# CHECK FILES
# ============================================================

if not DATA_PATH.exists():

    st.error(
        f"Dataset not found:\n\n{DATA_PATH}"
    )

    st.stop()


if not MODEL_PATH.exists():

    st.error(
        f"Model not found:\n\n{MODEL_PATH}"
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model_package = joblib.load(MODEL_PATH)

except Exception as e:

    st.error("Failed to load the trained model.")
    st.exception(e)
    st.stop()


# ============================================================
# EXTRACT MODEL
# ============================================================

if isinstance(model_package, dict):

    if "model" not in model_package:

        st.error(
            "The saved model file does not contain a 'model' key."
        )

        st.stop()

    model = model_package["model"]

    saved_features = model_package.get(
        "features",
        None
    )

else:

    # In case the pkl directly contains the model
    model = model_package
    saved_features = None


# ============================================================
# EXPECTED V2 FEATURES
# ============================================================

features_v2 = [
    "store_id",
    "sku_id",
    "total_price",
    "base_price",
    "discount_pct",
    "price_ratio",
    "is_featured_sku",
    "is_display_sku",
    "year",
    "month",
    "week_of_year",
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "lag_13",
    "lag_26",
    "rolling_mean_4",
    "rolling_mean_8",
    "rolling_std_4",
    "rolling_std_8",
    "demand_trend"
]


# ============================================================
# MODEL FEATURE CHECK
# ============================================================

try:

    model_feature_names = list(
        model.feature_names_in_
    )

except AttributeError:

    model_feature_names = None


if model_feature_names is not None:

    if model_feature_names != features_v2:

        st.error(
            "The saved model does not match the V2 feature set."
        )

        st.write(
            "Model features:",
            model_feature_names
        )

        st.write(
            "Expected V2 features:",
            features_v2
        )

        st.stop()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data(path):

    data = pd.read_csv(path)

    # Convert date
    data["week"] = pd.to_datetime(
        data["week"],
        dayfirst=True
    )

    # Handle missing price
    data["total_price"] = (
        data["total_price"]
        .fillna(data["total_price"].median())
    )

    # Sort
    data = data.sort_values(
        ["store_id", "sku_id", "week"]
    ).reset_index(drop=True)

    return data


df = load_data(DATA_PATH)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

@st.cache_data
def prepare_data(data):

    data = data.copy()

    # -----------------------------
    # Calendar features
    # -----------------------------

    data["year"] = data["week"].dt.year

    data["month"] = data["week"].dt.month

    data["week_of_year"] = (
        data["week"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    # -----------------------------
    # Group
    # -----------------------------

    group = (
        data
        .groupby(["store_id", "sku_id"])
        ["units_sold"]
    )

    # -----------------------------
    # Lag features
    # -----------------------------

    data["lag_1"] = group.shift(1)

    data["lag_2"] = group.shift(2)

    data["lag_4"] = group.shift(4)

    data["lag_8"] = group.shift(8)

    data["lag_13"] = group.shift(13)

    data["lag_26"] = group.shift(26)

    # -----------------------------
    # Rolling mean
    # -----------------------------

    data["rolling_mean_4"] = (
        data
        .groupby(["store_id", "sku_id"])["units_sold"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(4)
            .mean()
        )
    )

    data["rolling_mean_8"] = (
        data
        .groupby(["store_id", "sku_id"])["units_sold"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(8)
            .mean()
        )
    )

    # -----------------------------
    # Rolling std
    # -----------------------------

    data["rolling_std_4"] = (
        data
        .groupby(["store_id", "sku_id"])["units_sold"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(4)
            .std()
        )
    )

    data["rolling_std_8"] = (
        data
        .groupby(["store_id", "sku_id"])["units_sold"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(8)
            .std()
        )
    )

    # -----------------------------
    # Demand trend
    # -----------------------------

    data["demand_trend"] = (
        data["lag_1"]
        - data["lag_4"]
    )

    # -----------------------------
    # Price features
    # -----------------------------

    data["discount_pct"] = np.where(
        data["base_price"] != 0,
        (
            (data["base_price"] - data["total_price"])
            / data["base_price"]
        ) * 100,
        0
    )

    data["price_ratio"] = np.where(
        data["base_price"] != 0,
        data["total_price"]
        / data["base_price"],
        1
    )

    return data


df = prepare_data(df)


# ============================================================
# FORECAST FUNCTION
# ============================================================

def forecast_demand(
    store_id,
    sku_id,
    weeks_ahead
):

    # ---------------------------------------------
    # Get Store-SKU history
    # ---------------------------------------------

    history = df[
        (df["store_id"] == store_id) &
        (df["sku_id"] == sku_id)
    ].sort_values("week").copy()

    if len(history) < 26:

        raise ValueError(
            "Not enough historical data for this "
            "Store-SKU combination."
        )

    forecasts = []

    # ---------------------------------------------
    # Recursive forecasting
    # ---------------------------------------------

    for _ in range(weeks_ahead):

        # Next week
        next_week = (
            history["week"].max()
            + pd.Timedelta(weeks=1)
        )

        # Demand history
        demand = (
            history["units_sold"]
            .astype(float)
            .values
        )

        # -----------------------------------------
        # Lag features
        # -----------------------------------------

        lag_1 = demand[-1]

        lag_2 = demand[-2]

        lag_4 = demand[-4]

        lag_8 = demand[-8]

        lag_13 = demand[-13]

        lag_26 = demand[-26]

        # -----------------------------------------
        # Rolling features
        # -----------------------------------------

        rolling_mean_4 = np.mean(
            demand[-4:]
        )

        rolling_mean_8 = np.mean(
            demand[-8:]
        )

        rolling_std_4 = np.std(
            demand[-4:],
            ddof=1
        )

        rolling_std_8 = np.std(
            demand[-8:],
            ddof=1
        )

        # -----------------------------------------
        # Trend
        # -----------------------------------------

        demand_trend = (
            lag_1
            - lag_4
        )

        # -----------------------------------------
        # Latest known information
        # -----------------------------------------

        latest = history.iloc[-1]

        total_price = float(
            latest["total_price"]
        )

        base_price = float(
            latest["base_price"]
        )

        is_featured_sku = latest[
            "is_featured_sku"
        ]

        is_display_sku = latest[
            "is_display_sku"
        ]

        # -----------------------------------------
        # Price features
        # -----------------------------------------

        if base_price != 0:

            discount_pct = (
                (base_price - total_price)
                / base_price
            ) * 100

            price_ratio = (
                total_price
                / base_price
            )

        else:

            discount_pct = 0
            price_ratio = 1

        # -----------------------------------------
        # Create future feature row
        # -----------------------------------------

        future_row = pd.DataFrame([{

            "store_id": store_id,

            "sku_id": sku_id,

            "total_price": total_price,

            "base_price": base_price,

            "discount_pct": discount_pct,

            "price_ratio": price_ratio,

            "is_featured_sku": (
                is_featured_sku
            ),

            "is_display_sku": (
                is_display_sku
            ),

            "year": next_week.year,

            "month": next_week.month,

            "week_of_year": int(
                next_week.isocalendar().week
            ),

            "lag_1": lag_1,

            "lag_2": lag_2,

            "lag_4": lag_4,

            "lag_8": lag_8,

            "lag_13": lag_13,

            "lag_26": lag_26,

            "rolling_mean_4": (
                rolling_mean_4
            ),

            "rolling_mean_8": (
                rolling_mean_8
            ),

            "rolling_std_4": (
                rolling_std_4
            ),

            "rolling_std_8": (
                rolling_std_8
            ),

            "demand_trend": (
                demand_trend
            )

        }])

        # -----------------------------------------
        # Make sure feature order is correct
        # -----------------------------------------

        future_X = future_row[
            features_v2
        ].copy()

        # -----------------------------------------
        # Prediction
        # -----------------------------------------

        prediction = model.predict(
            future_X
        )[0]

        # Prevent negative demand
        prediction = max(
            0,
            float(prediction)
        )

        # -----------------------------------------
        # Save forecast
        # -----------------------------------------

        forecasts.append({

            "week": next_week,

            "predicted_demand": prediction

        })

        # -----------------------------------------
        # Add predicted demand into history
        # -----------------------------------------

        new_row = latest.copy()

        new_row["week"] = next_week

        new_row["units_sold"] = prediction

        history = pd.concat(
            [
                history,
                pd.DataFrame([new_row])
            ],
            ignore_index=True
        )

    return pd.DataFrame(
        forecasts
    )



# ============================================================
# PREMIUM DASHBOARD UI
# ============================================================

# ---------- Visual system ----------
st.markdown("""
<style>
    /* App background */
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(99,102,241,.08), transparent 30%),
            radial-gradient(circle at 90% 10%, rgba(14,165,233,.07), transparent 28%),
            #f7f8fc;
    }

    /* Main width */
    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #101827;
        border-right: 1px solid rgba(255,255,255,.08);
    }
    section[data-testid="stSidebar"] * {
        color: #f8fafc;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #cbd5e1;
    }

    /* Hero */
    .hero {
        padding: 28px 32px;
        border-radius: 24px;
        background: linear-gradient(135deg, #111827 0%, #1e293b 58%, #312e81 100%);
        box-shadow: 0 18px 45px rgba(15,23,42,.16);
        color: white;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        background: rgba(99,102,241,.20);
        right: -90px;
        top: -120px;
    }
    .hero h1 {
        margin: 0 0 8px 0;
        font-size: 2.35rem;
        letter-spacing: -1.2px;
        position: relative;
        z-index: 1;
    }
    .hero p {
        margin: 0;
        color: #cbd5e1;
        font-size: 1rem;
        position: relative;
        z-index: 1;
    }
    .hero-badge {
        display: inline-block;
        margin-bottom: 14px;
        padding: 5px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,.10);
        color: #c7d2fe;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .6px;
        text-transform: uppercase;
        position: relative;
        z-index: 1;
    }

    /* KPI cards */
    .kpi {
        background: rgba(255,255,255,.92);
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 18px 20px;
        min-height: 112px;
        box-shadow: 0 8px 28px rgba(15,23,42,.06);
    }
    .kpi-label {
        color: #64748b;
        font-size: .78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .7px;
    }
    .kpi-value {
        color: #0f172a;
        font-size: 1.72rem;
        font-weight: 800;
        margin-top: 5px;
        letter-spacing: -.5px;
    }
    .kpi-sub {
        color: #94a3b8;
        font-size: .78rem;
        margin-top: 3px;
    }

    /* Section headings */
    .section-title {
        color: #0f172a;
        font-size: 1.22rem;
        font-weight: 800;
        margin: 28px 0 10px 0;
        letter-spacing: -.2px;
    }
    .section-caption {
        color: #64748b;
        font-size: .88rem;
        margin-bottom: 14px;
    }

    /* Insight card */
    .insight {
        border-radius: 18px;
        padding: 18px 20px;
        background: white;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 28px rgba(15,23,42,.05);
        height: 100%;
    }
    .insight-title {
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 7px;
    }
    .insight-text {
        color: #64748b;
        font-size: .9rem;
        line-height: 1.55;
    }

    /* Pills */
    .pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: #eef2ff;
        color: #4338ca;
        font-size: .75rem;
        font-weight: 800;
        margin-right: 6px;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        font-weight: 800;
        min-height: 44px;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 700;
    }

    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------- Header ----------
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI • Demand Intelligence</div>
    <h1>Product Demand Forecaster</h1>
    <p>Forecast future Store–SKU demand using historical sales patterns, pricing signals and time-series features.</p>
</div>
""", unsafe_allow_html=True)


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ⚙️ Forecast Controls")
    st.caption("Configure the product and planning horizon.")

    stores = sorted(df["store_id"].dropna().unique())
    selected_store = st.selectbox(
        "Store",
        stores,
        format_func=lambda x: f"Store {x}"
    )

    available_skus = sorted(
        df[df["store_id"] == selected_store]["sku_id"].dropna().unique()
    )
    selected_sku = st.selectbox(
        "Product / SKU",
        available_skus,
        format_func=lambda x: f"SKU {x}"
    )

    weeks_ahead = st.select_slider(
        "Forecast horizon",
        options=[1, 4, 8],
        value=4,
        format_func=lambda x: f"{x} week" if x == 1 else f"{x} weeks"
    )

    st.markdown("---")
    forecast_button = st.button(
        "🔮 Generate Forecast",
        use_container_width=True,
        type="primary"
    )

    st.markdown("---")
    st.markdown("**Model inputs**")
    st.caption("Historical demand • price • discount • promotion/display • seasonal features • lagged demand")
    st.caption(f"Dataset: {len(df):,} records")
    st.caption(f"Stores: {df['store_id'].nunique():,}  •  SKUs: {df['sku_id'].nunique():,}")


# ---------- Selection context ----------
context_col1, context_col2, context_col3 = st.columns([1, 1, 1])
with context_col1:
    st.markdown(f'<span class="pill">STORE {selected_store}</span>', unsafe_allow_html=True)
with context_col2:
    st.markdown(f'<span class="pill">SKU {selected_sku}</span>', unsafe_allow_html=True)
with context_col3:
    st.markdown(f'<span class="pill">{weeks_ahead}-WEEK HORIZON</span>', unsafe_allow_html=True)


# ---------- Forecast ----------
if forecast_button:
    try:
        with st.spinner("Generating forecast..."):
            forecast = forecast_demand(
                store_id=selected_store,
                sku_id=selected_sku,
                weeks_ahead=weeks_ahead
            )

        history = df[
            (df["store_id"] == selected_store) &
            (df["sku_id"] == selected_sku)
        ].sort_values("week")

        # Core KPIs
        recent_8 = history["units_sold"].tail(8)
        avg_demand = recent_8.mean()
        total_forecast = forecast["predicted_demand"].sum()
        avg_forecast = forecast["predicted_demand"].mean()
        latest_demand = history["units_sold"].iloc[-1]

        forecast_change = (
            ((avg_forecast - avg_demand) / avg_demand) * 100
            if avg_demand != 0 else 0
        )

        forecast_min = forecast["predicted_demand"].min()
        forecast_max = forecast["predicted_demand"].max()

        # ---------- KPI row ----------
        st.markdown('<div class="section-title">Forecast at a glance</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">A compact view of current demand and the selected planning horizon.</div>',
            unsafe_allow_html=True
        )

        k1, k2, k3, k4 = st.columns(4)

        def kpi_card(container, label, value, sub):
            container.markdown(
                f"""
                <div class="kpi">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        kpi_card(k1, "Latest demand", f"{latest_demand:,.0f}", "units in latest observed week")
        kpi_card(k2, "Recent average", f"{avg_demand:,.0f}", "8-week historical average")
        kpi_card(k3, "Forecast / week", f"{avg_forecast:,.0f}", f"{forecast_change:+.1f}% vs recent average")
        kpi_card(k4, f"{weeks_ahead}-week demand", f"{total_forecast:,.0f}", f"range: {forecast_min:,.0f}–{forecast_max:,.0f} units")

        # ---------- Main chart ----------
        st.markdown('<div class="section-title">Demand trajectory</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Recent historical demand followed by the model-generated forecast.</div>',
            unsafe_allow_html=True
        )

        chart_col, insight_col = st.columns([2.5, 1])

        with chart_col:
            history_chart = history.tail(52)

            fig, ax = plt.subplots(figsize=(14, 5.5))
            fig.patch.set_alpha(0)
            ax.set_facecolor("#ffffff")

            ax.plot(
                history_chart["week"],
                history_chart["units_sold"],
                linewidth=2.4,
                label="Historical demand"
            )

            ax.plot(
                forecast["week"],
                forecast["predicted_demand"],
                marker="o",
                markersize=6,
                linewidth=2.8,
                label="Forecast"
            )

            ax.axvline(
                history["week"].max(),
                linestyle="--",
                linewidth=1.3,
                alpha=.7,
                label="Forecast starts"
            )

            ax.fill_between(
                forecast["week"],
                forecast["predicted_demand"],
                alpha=.08
            )

            ax.set_xlabel("")
            ax.set_ylabel("Units sold", fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_alpha(.18)
            ax.spines["bottom"].set_alpha(.18)
            ax.grid(axis="y", alpha=.16)
            ax.legend(frameon=False, loc="upper left")
            fig.autofmt_xdate()
            plt.tight_layout()

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with insight_col:
            if forecast_change > 5:
                direction = "Increasing"
                headline = "Demand is trending upward"
                detail = f"The forecast average is {forecast_change:.1f}% above the recent 8-week average."
                icon = "↗️"
            elif forecast_change < -5:
                direction = "Decreasing"
                headline = "Demand is trending downward"
                detail = f"The forecast average is {abs(forecast_change):.1f}% below the recent 8-week average."
                icon = "↘️"
            else:
                direction = "Stable"
                headline = "Demand is relatively stable"
                detail = f"The forecast average is within ±5% of the recent 8-week average."
                icon = "→"

            st.markdown(
                f"""
                <div class="insight">
                    <div class="insight-title">{icon} {headline}</div>
                    <div class="insight-text">{detail}</div>
                    <br>
                    <div class="insight-title">Planning view</div>
                    <div class="insight-text">
                        Expected weekly demand: <b>{avg_forecast:,.0f}</b> units.<br>
                        Total horizon demand: <b>{total_forecast:,.0f}</b> units.<br>
                        Forecast range: <b>{forecast_min:,.0f}–{forecast_max:,.0f}</b>.
                    </div>
                    <br>
                    <div class="insight-title">Model context</div>
                    <div class="insight-text">
                        Forecasts are recursive: each predicted week is fed back into the demand history for the next prediction.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ---------- Forecast table + trend metrics ----------
        table_col, detail_col = st.columns([1.55, 1])

        with table_col:
            st.markdown('<div class="section-title">Weekly forecast</div>', unsafe_allow_html=True)

            display_forecast = forecast.copy()
            display_forecast["week"] = display_forecast["week"].dt.strftime("%d %b %Y")
            display_forecast["predicted_demand"] = display_forecast["predicted_demand"].round(0).astype(int)
            display_forecast["change_vs_previous"] = (
                display_forecast["predicted_demand"].pct_change() * 100
            ).round(1)
            display_forecast["change_vs_previous"] = display_forecast["change_vs_previous"].fillna(0)

            display_forecast.columns = [
                "Week",
                "Predicted Demand",
                "WoW Change %"
            ]

            st.dataframe(
                display_forecast,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Predicted Demand": st.column_config.NumberColumn(
                        "Predicted Demand",
                        format="%d units"
                    ),
                    "WoW Change %": st.column_config.NumberColumn(
                        "WoW Change",
                        format="%+.1f%%"
                    )
                }
            )

        with detail_col:
            st.markdown('<div class="section-title">Forecast diagnostics</div>', unsafe_allow_html=True)

            volatility = history["units_sold"].tail(8).std()
            cv = (volatility / avg_demand * 100) if avg_demand else 0

            d1, d2 = st.columns(2)
            d1.metric("Recent volatility", f"{volatility:,.1f}")
            d2.metric("Demand CV", f"{cv:.1f}%")

            latest_row = history.iloc[-1]
            discount = (
                ((latest_row["base_price"] - latest_row["total_price"]) /
                 latest_row["base_price"]) * 100
                if latest_row["base_price"] else 0
            )

            st.markdown(
                f"""
                <div class="insight" style="margin-top:12px;">
                    <div class="insight-title">Latest commercial signals</div>
                    <div class="insight-text">
                        Current price: <b>{latest_row["total_price"]:.2f}</b><br>
                        Base price: <b>{latest_row["base_price"]:.2f}</b><br>
                        Discount: <b>{discount:.1f}%</b><br>
                        Featured SKU: <b>{"Yes" if latest_row["is_featured_sku"] else "No"}</b><br>
                        Display SKU: <b>{"Yes" if latest_row["is_display_sku"] else "No"}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ---------- History detail ----------
        with st.expander("📊 View historical demand data"):
            hist_view = history[["week", "units_sold", "total_price", "base_price",
                                 "is_featured_sku", "is_display_sku"]].tail(52).copy()
            hist_view.columns = [
                "Week", "Units Sold", "Selling Price", "Base Price",
                "Featured", "Display"
            ]
            st.dataframe(
                hist_view.sort_values("Week", ascending=False),
                use_container_width=True,
                hide_index=True
            )

        # ---------- Assumptions ----------
        st.markdown(
            """
            <div class="section-title">Forecast assumptions</div>
            <div class="section-caption">
                Future price, promotion and display conditions are held at their latest observed values.
                The model uses lagged demand, rolling statistics, calendar features and price-related features.
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error("Forecast generation failed.")
        st.exception(e)

else:
    # ---------- Empty state ----------
    st.markdown(
        """
        <div class="insight" style="margin-top:28px; padding:30px;">
            <div class="insight-title" style="font-size:1.35rem;">Ready to forecast</div>
            <div class="insight-text">
                Choose a store, select a SKU, set the forecast horizon and click
                <b>Generate Forecast</b>. The dashboard will calculate future demand
                and show the forecast trajectory, planning metrics and diagnostics.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Dataset overview before a forecast is generated
    st.markdown('<div class="section-title">Dataset overview</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)

    a.metric("Records", f"{len(df):,}")
    b.metric("Stores", f"{df['store_id'].nunique():,}")
    c.metric("SKUs", f"{df['sku_id'].nunique():,}")
    d.metric("Store–SKU pairs", f"{df[['store_id','sku_id']].drop_duplicates().shape[0]:,}")

    st.caption(
        f"Historical coverage: {df['week'].min().strftime('%d %b %Y')} → {df['week'].max().strftime('%d %b %Y')}"
    )
