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
# DASHBOARD HEADER
# ============================================================

st.title(
    "📈 Product Demand Forecaster"
)

st.write(
    "Machine-learning based demand forecasting "
    "using historical Store-SKU sales data."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Forecast Settings"
)


# ============================================================
# STORE SELECTION
# ============================================================

stores = sorted(
    df["store_id"]
    .dropna()
    .unique()
)


selected_store = st.sidebar.selectbox(
    "Select Store",
    stores
)


# ============================================================
# SKU SELECTION
# ============================================================

available_skus = sorted(
    df[
        df["store_id"] == selected_store
    ]["sku_id"]
    .dropna()
    .unique()
)


selected_sku = st.sidebar.selectbox(
    "Select SKU",
    available_skus
)


# ============================================================
# FORECAST HORIZON
# ============================================================

weeks_ahead = st.sidebar.selectbox(
    "Forecast Horizon",
    [1, 4, 8],
    index=1
)


# ============================================================
# FORECAST BUTTON
# ============================================================

forecast_button = st.sidebar.button(
    "🔮 Generate Forecast",
    use_container_width=True
)


# ============================================================
# MAIN FORECAST
# ============================================================

if forecast_button:

    try:

        # -----------------------------------------
        # Generate forecast
        # -----------------------------------------

        forecast = forecast_demand(
            store_id=selected_store,
            sku_id=selected_sku,
            weeks_ahead=weeks_ahead
        )

        # -----------------------------------------
        # Historical data
        # -----------------------------------------

        history = df[
            (df["store_id"] == selected_store) &
            (df["sku_id"] == selected_sku)
        ].sort_values("week")

        # -----------------------------------------
        # Metrics
        # -----------------------------------------

        avg_demand = (
            history[
                "units_sold"
            ]
            .tail(8)
            .mean()
        )

        total_forecast = (
            forecast[
                "predicted_demand"
            ]
            .sum()
        )

        avg_forecast = (
            forecast[
                "predicted_demand"
            ]
            .mean()
        )

        latest_demand = (
            history[
                "units_sold"
            ]
            .iloc[-1]
        )

        # -----------------------------------------
        # KPI CARDS
        # -----------------------------------------

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        col1.metric(
            "Latest Demand",
            f"{latest_demand:.0f} units"
        )

        col2.metric(
            "Recent Avg Demand",
            f"{avg_demand:.0f} units"
        )

        col3.metric(
            "Forecast / Week",
            f"{avg_forecast:.0f} units"
        )

        col4.metric(
            f"Forecast ({weeks_ahead} Weeks)",
            f"{total_forecast:.0f} units"
        )

        st.divider()

        # -----------------------------------------
        # FORECAST TABLE
        # -----------------------------------------

        st.subheader(
            "Future Demand Forecast"
        )

        display_forecast = forecast.copy()

        display_forecast[
            "predicted_demand"
        ] = (
            display_forecast[
                "predicted_demand"
            ].round(0)
        )

        display_forecast.columns = [
            "Week",
            "Predicted Demand"
        ]

        st.dataframe(
            display_forecast,
            use_container_width=True,
            hide_index=True
        )

        # -----------------------------------------
        # CHART
        # -----------------------------------------

        st.subheader(
            "Historical Demand & Forecast"
        )

        fig, ax = plt.subplots(
            figsize=(14, 6)
        )

        # Show last 52 weeks
        history_chart = (
            history.tail(52)
        )

        ax.plot(
            history_chart["week"],
            history_chart["units_sold"],
            label="Historical Demand"
        )

        ax.plot(
            forecast["week"],
            forecast["predicted_demand"],
            marker="o",
            label="Forecast"
        )

        ax.axvline(
            history["week"].max(),
            linestyle="--",
            label="Forecast Start"
        )

        ax.set_xlabel(
            "Week"
        )

        ax.set_ylabel(
            "Units Sold"
        )

        ax.set_title(
            f"Store {selected_store} | "
            f"SKU {selected_sku}"
        )

        ax.legend()

        ax.grid(True)

        fig.autofmt_xdate()

        st.pyplot(
            fig,
            use_container_width=True
        )

        # -----------------------------------------
        # FORECAST SUMMARY
        # -----------------------------------------

        st.subheader(
            "Forecast Summary"
        )

        if avg_forecast > avg_demand * 1.05:

            st.success(
                "The forecast indicates demand may "
                "increase compared with recent demand."
            )

        elif avg_forecast < avg_demand * 0.95:

            st.warning(
                "The forecast indicates demand may "
                "decrease compared with recent demand."
            )

        else:

            st.info(
                "The forecast indicates demand is "
                "relatively stable."
            )

        st.caption(
            "Forecast assumption: future price, promotion "
            "and display conditions remain at their latest "
            "observed values."
        )

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            "Forecast generation failed."
        )

        st.exception(e)


# ============================================================
# DEFAULT SCREEN
# ============================================================

else:

    st.info(
        "Select a Store, SKU and forecast horizon, "
        "then click **Generate Forecast**."
    )