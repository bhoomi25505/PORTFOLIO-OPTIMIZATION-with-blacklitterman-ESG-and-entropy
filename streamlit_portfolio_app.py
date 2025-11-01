
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from datetime import datetime
import yfinance as yf

# Portfolio optimization libs
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
    from pypfopt import plotting as pyp_plotting
    from pypfopt.black_litterman import BlackLittermanModel
except Exception as e:
    st.error("PyPortfolioOpt is required. Run `pip install pyportfolioopt` in your environment.")
    raise e

st.set_page_config(layout="wide", page_title="Portfolio Optimization Dashboard")

st.title("Portfolio Optimization Dashboard")
st.markdown("Upload a screener CSV (or use the provided dataset), select tickers and optimize a portfolio using mean-variance or Black-Litterman.")

# Load default screener dataset bundled with the app
@st.cache_data
def load_screener(path="finance_screener_output.csv"):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["symbol","companyName","industry","marketCap","revenue","netIncome","ebitda","price","exchange"])
    return df

screener_df = load_screener("finance_screener_output.csv")

# Allow user upload to override
uploaded = st.file_uploader("Upload your screener CSV (optional)", type=["csv"])
if uploaded is not None:
    screener_df = pd.read_csv(uploaded)

with st.expander("Screener dataset (first 50 rows)"):
    st.dataframe(screener_df.head(50))

# Basic filters
st.sidebar.header("Universe selection")
industries = ["All"] + sorted(screener_df["industry"].dropna().unique().tolist())
industry = st.sidebar.selectbox("Industry", industries)
if industry != "All":
    universe = screener_df[screener_df["industry"] == industry]
else:
    universe = screener_df.copy()

# Allow manual ticker selection
symbols = universe["symbol"].dropna().unique().tolist()
selected = st.sidebar.multiselect("Select tickers (symbols)", symbols, default=symbols[:6])

# Historical price input options
st.sidebar.header("Price data")
period = st.sidebar.selectbox("Period", ["1y","2y","3y","5y","10y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"], index=0)
use_upload_prices = st.sidebar.checkbox("Upload historical prices CSV instead of fetching (optional)")

price_df = None
if use_upload_prices:
    uploaded_prices = st.file_uploader("Upload a CSV with Date column and ticker columns (wide format)")
    if uploaded_prices is not None:
        price_df = pd.read_csv(uploaded_prices, parse_dates=["Date"], index_col="Date")
        st.success("Uploaded price data loaded.")
else:
    if len(selected) == 0:
        st.warning("Please select at least one ticker in the sidebar to fetch price history.")
    else:
        with st.spinner("Downloading price data from yfinance..."):
            price_data = yf.download(selected, period=period, interval=interval, progress=False, threads=True)
            if price_data.empty:
                st.error("Failed to download price data — check ticker symbols or your internet connection.")
            else:
                # If multi-level columns, take 'Adj Close' if available else 'Close'
                if ("Adj Close" in price_data.columns.get_level_values(0)):
                    price_df = price_data["Adj Close"].copy()
                elif ("Adj Close" in price_data.columns):
                    price_df = price_data["Adj Close"].copy()
                elif ("Close" in price_data.columns.get_level_values(0)):
                    price_df = price_data["Close"].copy()
                else:
                    price_df = price_data.copy()

if price_df is not None:
    st.subheader("Price data preview")
    st.dataframe(price_df.tail())

# Compute returns and risk model
def compute_returns_and_cov(price_df, returns_type="log"):
    if returns_type == "log":
        rets = np.log(price_df / price_df.shift(1)).dropna()
    else:
        rets = price_df.pct_change().dropna()
    mu = expected_returns.mean_historical_return(price_df, returns_data=True) if False else expected_returns.mean_historical_return(price_df)
    S = risk_models.sample_cov(price_df)
    return rets, mu, S

if price_df is not None and not price_df.empty:
    returns_type = st.sidebar.selectbox("Returns type", ["log", "simple"], index=0)
    rets, mu, S = compute_returns_and_cov(price_df, returns_type=returns_type)

    st.sidebar.header("Optimization settings")
    opt_method = st.sidebar.selectbox("Method", ["Max Sharpe (risk-free=0)","Min Volatility","Mean-Variance (target return)","Black-Litterman (views)"])
    weight_bounds = st.sidebar.slider("Max weight per asset", 0.05, 1.0, 0.4, step=0.05)
    allow_short = st.sidebar.checkbox("Allow shorting (only for advanced users)", value=False)

    if opt_method == "Mean-Variance (target return)":
        target_return = st.sidebar.number_input("Target annual return (fraction, e.g. 0.12)", value=0.12, step=0.01, format="%.4f")

    if st.sidebar.button("Run optimization"):
        try:
            if opt_method == "Black-Litterman (views)":
                # Build prior (market-implied) using market caps if available
                try:
                    market_caps = screener_df.set_index("symbol")["marketCap"].dropna().astype(float)
                    weights_market = market_caps.loc[selected].values / market_caps.loc[selected].sum()
                except Exception:
                    weights_market = None

                # Convert returns to expected returns using Black-Litterman
                tau = 0.05
                if weights_market is None:
                    st.warning("Market cap weights not available for BL; falling back to sample expected returns.")
                    bl_mu = mu
                else:
                    # Let user input simple views
                    st.info("Enter views as ticker1,ticker2,...:alpha (alpha is view return in decimal), e.g. 'AAPL:0.05' or 'AAPL,MSFT:0.02'")
                    views_text = st.text_input("Views (one per line)", value="")
                    # parse views
                    P = []
                    Q = []
                    for line in views_text.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        if ":" not in line:
                            continue
                        left, right = line.split(":")
                        tickers_in_view = [t.strip() for t in left.split(",") if t.strip()]
                        alpha = float(right.strip())
                        p_row = [1 if t in tickers_in_view else 0 for t in selected]
                        P.append(p_row)
                        Q.append(alpha)
                    if len(P) == 0:
                        st.warning("No valid views provided; using historical expected returns instead.")
                        bl_mu = mu
                    else:
                        P = np.array(P)
                        Q = np.array(Q)
                        try:
                            bl = BlackLittermanModel(S, pi=weights_market, market_caps=weights_market, tau=tau, absolute_views=True, P=P, Q=Q)
                            bl_mu = bl.bl_returns()
                            bl_cov = bl.bl_cov()
                        except Exception as e:
                            st.error(f"Black-Litterman failed: {e}")
                            bl_mu = mu
                            bl_cov = S
                # use EfficientFrontier with bl_mu and bl_cov
                ef = EfficientFrontier(bl_mu, bl_cov, weight_bounds=(None if allow_short else 0, 1))
            else:
                # regular mean-variance
                ef = EfficientFrontier(mu, S, weight_bounds=(None if allow_short else 0, weight_bounds))

            if opt_method == "Max Sharpe (risk-free=0)":
                raw_weights = ef.max_sharpe(risk_free_rate=0.0)
            elif opt_method == "Min Volatility":
                raw_weights = ef.min_volatility()
            elif opt_method == "Mean-Variance (target return)":
                ef.add_objective(objective_functions.L2_reg)
                raw_weights = ef.efficient_return(target_return)
            else:
                raw_weights = ef.max_sharpe(risk_free_rate=0.0)

            cleaned = ef.clean_weights()
            st.subheader("Optimized weights")
            ws = pd.Series(cleaned).sort_values(ascending=False)
            st.dataframe(ws.to_frame("weight"))

            # Show pie chart
            st.subheader("Weights pie chart")
            fig1, ax1 = plt.subplots(figsize=(6,6))
            ws_nonzero = ws[ws>0]
            ax1.pie(ws_nonzero, labels=ws_nonzero.index, autopct="%1.1f%%", startangle=90)
            ax1.axis("equal")
            st.pyplot(fig1)

            # Efficient frontier plot (sampled)
            try:
                fig2, ax2 = plt.subplots(figsize=(8,5))
                pyp_plotting.plot_efficient_frontier(ef, ax=ax2, show_assets=False)
                st.pyplot(fig2)
            except Exception:
                st.info("Could not plot efficient frontier with the current objects (this can happen for BL).")

            # Compute portfolio expected performance
            # portfolio performance using pypfopt helper (if available)
            try:
                from pypfopt import expected_returns as exp, risk_models as rm, objective_functions as objf
                latest_weights = np.array([cleaned.get(t,0) for t in selected])
                port_ret = np.dot(latest_weights, mu.loc[selected])
                port_vol = np.sqrt(np.dot(latest_weights.T, np.dot(S.loc[selected, selected], latest_weights)))
                st.metric("Expected annual return (approx)", f"{port_ret:.2%}")
                st.metric("Expected annual volatility (approx)", f"{port_vol:.2%}")
            except Exception:
                pass

            # Show weights CSV download
            csv_out = ws.to_csv(index=True)
            st.download_button("Download weights CSV", data=csv_out, file_name="optimized_weights.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Optimization failed: {e}")

st.markdown("---")
st.markdown("Built with PyPortfolioOpt + Streamlit. This app fetches price data using yfinance on your machine.")
