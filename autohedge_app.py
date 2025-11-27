import os
import json
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

# ================================
# Env / secrets handling
# ================================
# Load local .env when running on your laptop
load_dotenv()

# Also support Streamlit Cloud secrets
for key in [
    "OPENAI_API_KEY",
    "WORKSPACE_DIR",
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "APCA_API_BASE_URL",
]:
    try:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = st.secrets[key]
    except Exception:
        # st.secrets not available when running as plain script
        pass

# ⬇️ only now import AutoFund so it sees OPENAI_API_KEY
from autohedge import AutoFund

# Alpaca
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

RESULTS_FILE = "autohedge_runs.jsonl"


# =========================================================
# Alpaca Helpers (Paper Trading)
# =========================================================

def get_alpaca_client() -> TradingClient | None:
    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not api_secret:
        return None

    # paper=True uses paper trading environment
    client = TradingClient(api_key, api_secret, paper=True)
    return client


def alpaca_place_market_order(symbol: str, side: str, qty: float):
    """
    Place a simple market order via Alpaca (paper).
    """
    client = get_alpaca_client()
    if client is None:
        raise RuntimeError("Alpaca API keys not set.")

    side_enum = OrderSide.BUY if side.upper() in ["BUY", "LONG"] else OrderSide.SELL

    order_req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side_enum,
        time_in_force=TimeInForce.DAY,
    )

    order = client.submit_order(order_req)
    return order


def alpaca_get_positions():
    client = get_alpaca_client()
    if client is None:
        return []
    return client.get_all_positions()


def alpaca_get_account():
    client = get_alpaca_client()
    if client is None:
        return None
    return client.get_account()


# =========================================================
# Custom Risk Rules
# =========================================================

def apply_custom_risk_rules(order: dict, capital: float,
                            max_position_pct: float = 0.1,
                            min_trade_notional: float = 1000.0) -> dict:
    """
    Apply simple custom risk rules:
      - Max position size as % of capital
      - Minimum notional
      - Only allow BUY / SELL / LONG / SHORT
    Returns a dict with:
      - approved: bool
      - adjusted_order: dict | None
      - reason: str
    """
    if not order or not isinstance(order, dict):
        return {
            "approved": False,
            "adjusted_order": None,
            "reason": "No valid order provided by AutoHedge.",
        }

    side = str(order.get("side") or order.get("action") or "").upper()
    price = order.get("entry_price") or order.get("price")
    quantity = order.get("quantity") or order.get("qty")

    if side not in {"BUY", "SELL", "LONG", "SHORT"}:
        return {
            "approved": False,
            "adjusted_order": None,
            "reason": f"Unsupported side '{side}'.",
        }

    if not price or price <= 0:
        return {
            "approved": False,
            "adjusted_order": None,
            "reason": "Missing or invalid price in order.",
        }

    max_notional = capital * max_position_pct

    if not quantity or quantity <= 0:
        quantity = max_notional / price

    notional = quantity * price

    if notional < min_trade_notional:
        return {
            "approved": False,
            "adjusted_order": None,
            "reason": f"Trade notional {notional:.2f} below minimum {min_trade_notional:.2f}.",
        }

    if notional > max_notional:
        quantity = max_notional / price
        notional = quantity * price

    adjusted_order = dict(order)
    adjusted_order["side"] = side
    adjusted_order["quantity"] = quantity
    adjusted_order["notional"] = notional

    return {
        "approved": True,
        "adjusted_order": adjusted_order,
        "reason": f"Order capped at {max_position_pct*100:.1f}% of capital.",
    }


# =========================================================
# AutoHedge Run + Save
# =========================================================

def save_result(autohedge_result, capital_assumed: float = 100_000.0):
    """
    Save AutoHedgeOutput + custom risk info into JSONL file.
    """

    # ---------- Handle AutoFund return types ----------
    if isinstance(autohedge_result, str):
        # AutoFund returned a JSON string – try to parse it
        try:
            parsed = json.loads(autohedge_result)
            data = parsed
        except Exception:
            # If parsing fails, at least keep the raw text
            data = {"raw_output": autohedge_result}
    else:
        if hasattr(autohedge_result, "model_dump"):
            data = autohedge_result.model_dump()
        elif hasattr(autohedge_result, "dict"):
            data = autohedge_result.dict()
        else:
            data = (
                autohedge_result.__dict__
                if hasattr(autohedge_result, "__dict__")
                else {"result": autohedge_result}
            )


    # ---------- Custom Risk Rules ----------
    order = data.get("order") if isinstance(data.get("order"), dict) else {}
    risk_eval = apply_custom_risk_rules(order, capital=capital_assumed)

    data["custom_risk_approved"] = risk_eval["approved"]
    data["custom_risk_reason"] = risk_eval["reason"]
    data["custom_risk_capital_assumed"] = capital_assumed
    data["custom_risk_order"] = risk_eval["adjusted_order"]

    adj_order = risk_eval["adjusted_order"] or {}

    data["order_side"] = (
        adj_order.get("side")
        or order.get("side")
        or order.get("action")
    )
    data["order_quantity"] = (
        adj_order.get("quantity")
        or order.get("quantity")
        or order.get("qty")
    )
    data["order_entry_price"] = (
        adj_order.get("entry_price")
        or adj_order.get("price")
        or order.get("entry_price")
        or order.get("price")
    )
    data["order_notional"] = adj_order.get("notional")

    # ---------- Add timestamp ----------
    data["run_time"] = datetime.utcnow().isoformat()

    # ---------- Save to JSONL ----------
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")

    return data




# =========================================================
# Load Runs + Price Data
# =========================================================

def load_runs() -> pd.DataFrame:
    path = Path(RESULTS_FILE)
    if not path.exists():
        return pd.DataFrame()

    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Parse run_time
    if "run_time" in df.columns:
        df["run_time"] = pd.to_datetime(df["run_time"], errors="coerce")

    # Flatten stocks list
    if "stocks" in df.columns:
        df["stocks_str"] = df["stocks"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else str(x)
        )

    for col in ["order_side", "order_quantity", "order_entry_price", "order_notional"]:
        if col not in df.columns:
            df[col] = None

    return df


def get_yfinance_price_series(ticker: str, days_back: int = 60):
    end = datetime.now()
    start = end - timedelta(days=days_back)
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        return None
    return df


# =========================================================
# Backtest Engine
# =========================================================

def get_next_trading_day(dt: datetime) -> datetime:
    candidate = dt + timedelta(days=1)
    while candidate.weekday() >= 5:  # 5=Sat, 6=Sun
        candidate += timedelta(days=1)
    return candidate


def run_simple_backtest(
    df_runs: pd.DataFrame,
    ticker: str,
    start_capital: float = 100_000.0,
    risk_per_trade_pct: float = 0.05,
    holding_period_days: int = 5,
):
    """
    Simple long-only backtest:
      - Uses BUY/LONG signals
      - Enter at next day's open
      - Exit after holding_period_days at close
    """
    if df_runs.empty:
        return None, None, None

    df = df_runs.copy()
    df = df[(df["order_side"].astype(str).str.upper().isin(["BUY", "LONG"]))]
    if df.empty:
        return None, None, None

    df = df.sort_values("run_time")

    trades = []
    capital = start_capital
    equity = [capital]
    equity_dates = []

    for _, row in df.iterrows():
        run_time = row["run_time"]
        if pd.isna(run_time):
            continue

        entry_day = get_next_trading_day(run_time)
        exit_day = entry_day + timedelta(days=holding_period_days)

        prices = yf.download(ticker, start=entry_day, end=exit_day + timedelta(days=1))
        if prices.empty:
            continue

        entry_ts = prices.index[0]
        entry_price = float(prices["Open"].iloc[0])

        exit_ts = prices.index[-1]
        exit_price = float(prices["Close"].iloc[-1])

        position_notional = capital * risk_per_trade_pct
        if position_notional <= 0:
            continue

        quantity = position_notional / entry_price
        pnl = quantity * (exit_price - entry_price)
        capital_after = capital + pnl

        trades.append(
            {
                "run_time": run_time,
                "entry_time": entry_ts,
                "exit_time": exit_ts,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "position_notional": position_notional,
                "pnl": pnl,
                "capital_before": capital,
                "capital_after": capital_after,
            }
        )

        capital = capital_after
        equity.append(capital)
        equity_dates.append(exit_ts)

    if not trades:
        return None, None, None

    trades_df = pd.DataFrame(trades)
    equity_series = pd.Series(equity[1:], index=equity_dates, name="equity")

    # Metrics
    total_return = capital / start_capital - 1.0
    start_date = trades_df["entry_time"].min()
    end_date = trades_df["exit_time"].max()
    days = (end_date - start_date).days if start_date and end_date else 0
    years = days / 365.25 if days > 0 else None
    cagr = (capital / start_capital) ** (1 / years) - 1 if years and years > 0 else None

    wins = (trades_df["pnl"] > 0).sum()
    losses = (trades_df["pnl"] < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else None

    avg_win = trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()
    avg_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean()
    gross_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
    gross_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else None

    equity_values = equity_series.values
    running_max = []
    max_val = -float("inf")
    for v in equity_values:
        if v > max_val:
            max_val = v
        running_max.append(max_val)
    dd = equity_values / (pd.Series(running_max).values) - 1.0
    max_drawdown = dd.min() if len(dd) > 0 else None

    metrics = {
        "start_capital": start_capital,
        "end_capital": capital,
        "total_return": total_return,
        "cagr": cagr,
        "num_trades": len(trades_df),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
    }

    return metrics, trades_df, equity_series

def run_autohedge(
    stocks,
    allocation_usd: float,
    strategy_type: str = "momentum",
    risk_level: int = 5,
):
    """
    Run the AutoFund (AutoHedge) pipeline for given stocks and allocation,
    then save the result with our custom risk wrapper.
    """
    trading_system = AutoFund(stocks)
    tickers = ", ".join(stocks)
    # Use the first stock name for a more natural sentence
    stock = stocks[0] if stocks else tickers

    task = (
        f"Analyze {stock} and tell me whether to BUY, HOLD, or SELL. "
        "You MUST return a valid JSON object with fields: "
        "'thesis', 'quant_analysis', 'risk_assessment', and 'order'. "
        "The 'order' field must itself be a JSON object with keys: "
        "'side' ('buy' or 'sell'), 'quantity' (int), 'entry_price' (float), "
        "'stop_loss' (float), 'take_profit' (float). "
        f"We have ${allocation_usd:,.0f} allocation with a {strategy_type} style and "
        f"risk level {risk_level}/10."
    )

    autohedge_result = trading_system.run(task=task)
    saved = save_result(autohedge_result, capital_assumed=allocation_usd)
    return saved


# =========================================================
# Streamlit App
# =========================================================

st.set_page_config(page_title="AutoHedge + Alpaca Trading Dashboard", layout="wide")

st.title("📈 AutoHedge Trading Dashboard (Single File)")

st.markdown(
    """
This app lets you:

- Run **AutoHedge** (multi-agent AI) to generate trade ideas  
- Apply **your own risk rules** (max position, min notional)  
- Optionally send orders to **Alpaca paper trading**  
- See **thesis**, **risk**, **orders**  
- Run a **backtest** from AutoHedge signals (P&L, win rate, max drawdown, equity curve)  
"""
)

df = load_runs()

# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filters")

if df.empty:
    st.sidebar.info("No runs yet. Use the form at the bottom to run AutoHedge.")
    selected_stock = None
else:
    if "current_stock" in df.columns:
        available_stocks = sorted(s for s in df["current_stock"].dropna().unique())
    elif "stocks_str" in df.columns:
        available_stocks = sorted(
            set(
                df["stocks_str"].dropna().apply(
                    lambda s: str(s).split(",")[0].strip()
                )
            )
        )
    else:
        available_stocks = []

    selected_stock = st.sidebar.selectbox(
        "Select stock", options=available_stocks if available_stocks else ["(none)"]
    )

    if "run_time" in df.columns and not df["run_time"].isna().all():
        min_date = df["run_time"].min()
        max_date = df["run_time"].max()
        if pd.notna(min_date) and pd.notna(max_date):
            start_date, end_date = st.sidebar.date_input(
                "Run date range",
                value=[min_date.date(), max_date.date()],
            )
            df = df[
                (df["run_time"].dt.date >= start_date)
                & (df["run_time"].dt.date <= end_date)
            ]

    if selected_stock and selected_stock != "(none)":
        if "current_stock" in df.columns:
            df = df[df["current_stock"] == selected_stock]
        elif "stocks_str" in df.columns:
            df = df[df["stocks_str"].str.contains(selected_stock)]

# ---------------- Main Sections ----------------
if df.empty:
    st.warning("No runs available with current filters.")
else:
    latest = df.sort_values("run_time", ascending=False).iloc[0]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🧠 Latest AutoHedge View")
        st.markdown(f"**Run Time (UTC):** {latest['run_time']}")
        st.markdown(f"**Stock:** `{latest.get('current_stock', 'N/A')}`")
        st.markdown(f"**Task:** {latest.get('task', 'N/A')}")

        st.markdown("### Thesis")
        st.write(latest.get("thesis", "No thesis available"))

        st.markdown("### Risk Assessment")
        st.write(latest.get("risk_assessment", "No risk assessment available"))

    with col2:
        st.subheader("📋 Suggested Order (After Custom Risk)")
        st.markdown(f"**Custom risk approved:** {latest.get('custom_risk_approved', 'N/A')}")
        st.markdown(f"**Reason:** {latest.get('custom_risk_reason', 'N/A')}")

        order_fields = {
            "Side": latest.get("order_side"),
            "Quantity": latest.get("order_quantity"),
            "Entry Price": latest.get("order_entry_price"),
            "Notional": latest.get("order_notional"),
        }
        for k, v in order_fields.items():
            st.markdown(f"**{k}:** {v if v is not None else 'N/A'}")

        # Optional: send to Alpaca
        st.markdown("### 🚀 Send Order to Alpaca (Paper)")
        if st.button("Send latest order to Alpaca"):
            if not latest.get("order_side") or not latest.get("order_quantity"):
                st.error("Order side or quantity missing.")
            elif not selected_stock or selected_stock == "(none)":
                st.error("Select a stock in sidebar first.")
            else:
                try:
                    placed = alpaca_place_market_order(
                        symbol=selected_stock,
                        side=str(latest["order_side"]),
                        qty=float(latest["order_quantity"]),
                    )
                    st.success(f"Order sent to Alpaca. ID: {placed.id}")
                    st.json(dict(placed))
                except Exception as e:
                    st.error(f"Error placing order: {e}")

    # Historical runs
    st.subheader("📜 Historical Runs")

    display_cols = [
        col
        for col in [
            "run_time", "current_stock", "stocks_str",
            "task", "thesis", "risk_assessment",
            "order_side", "order_quantity", "order_entry_price", "order_notional",
            "custom_risk_approved",
        ]
        if col in df.columns
    ]

    st.dataframe(
        df[display_cols].sort_values("run_time", ascending=False),
        use_container_width=True,
    )

    # Price chart
    if selected_stock and selected_stock != "(none)":
        st.subheader(f"💹 Recent Price – {selected_stock}")
        days_back = st.slider(
            "Days back for price chart", min_value=10, max_value=180, value=60, step=10
        )
        price_df = get_yfinance_price_series(selected_stock, days_back=days_back)
        if price_df is not None:
            st.line_chart(price_df["Close"])
        else:
            st.info("No price data available for this ticker.")

    # Backtest
    st.markdown("---")
    st.header("📊 Backtest From AutoHedge Signals")

    if not selected_stock or selected_stock == "(none)":
        st.info("Select a stock in the sidebar to run a backtest.")
    else:
        col_bt1, col_bt2, col_bt3 = st.columns(3)
        with col_bt1:
            start_capital = st.number_input(
                "Starting capital (USD)", min_value=1_000.0, value=100_000.0, step=1_000.0
            )
        with col_bt2:
            risk_pct = st.slider(
                "Risk per trade (% of capital)",
                min_value=1.0, max_value=50.0, value=5.0, step=1.0,
            )
        with col_bt3:
            holding_days = st.slider(
                "Holding period (days)", min_value=1, max_value=30, value=5, step=1
            )

        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                metrics, trades_df, equity_series = run_simple_backtest(
                    df_runs=df,
                    ticker=selected_stock,
                    start_capital=start_capital,
                    risk_per_trade_pct=risk_pct / 100.0,
                    holding_period_days=holding_days,
                )

            if metrics is None:
                st.warning("No valid BUY/LONG signals to backtest.")
            else:
                st.subheader("Backtest Metrics")
                m = metrics

                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric(
                    "End capital",
                    f"${m['end_capital']:,.2f}",
                    f"{m['total_return']*100:.1f}%",
                )
                col_m2.metric(
                    "CAGR", f"{m['cagr']*100:.2f}%" if m["cagr"] is not None else "N/A"
                )
                col_m3.metric(
                    "Win rate",
                    f"{m['win_rate']*100:.1f}%" if m["win_rate"] is not None else "N/A",
                )

                col_m4, col_m5, col_m6 = st.columns(3)
                col_m4.metric(
                    "Max drawdown",
                    f"{m['max_drawdown']*100:.1f}%" if m["max_drawdown"] is not None else "N/A",
                )
                col_m5.metric(
                    "Profit factor",
                    f"{m['profit_factor']:.2f}" if m["profit_factor"] is not None else "N/A",
                )
                col_m6.metric("Trades", m["num_trades"])

                st.subheader("Equity Curve")
                st.line_chart(equity_series)

                st.subheader("Trades Detail")
                st.dataframe(trades_df, use_container_width=True)

# Alpaca account/positions section
st.markdown("---")
st.header("📟 Alpaca Account & Positions (Paper)")

client = get_alpaca_client()
if client is None:
    st.info("Alpaca API keys not set or .env not loaded. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY to see account/positions.")
else:
    try:
        acct = alpaca_get_account()
    except Exception as e:
        st.error(f"Could not fetch Alpaca account (likely unauthorized). Please double-check your Paper API keys in .env. Error: {e}")
        acct = None

    if acct:
        col_a1, col_a2, col_a3 = st.columns(3)
        col_a1.metric("Equity", f"${float(acct.equity):,.2f}")
        col_a2.metric("Cash", f"${float(acct.cash):,.2f}")
        col_a3.metric("Portfolio value", f"${float(acct.portfolio_value):,.2f}")

        positions = alpaca_get_positions()
        if positions:
            pos_data = []
            for p in positions:
                pos_data.append(
                    {
                        "symbol": p.symbol,
                        "qty": float(p.qty),
                        "avg_entry_price": float(p.avg_entry_price),
                        "market_value": float(p.market_value),
                        "unrealized_pl": float(p.unrealized_pl),
                    }
                )
            st.subheader("Open Positions")
            st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
        else:
            st.info("No open positions in Alpaca paper account.")


# Run new AutoHedge
st.markdown("---")
st.header("⚙️ Run New AutoHedge Analysis")

with st.form("new_run_form"):
    tickers_input = st.text_input("Tickers (comma-separated)", value="NVDA")
    allocation = st.number_input(
        "Allocation (USD)", min_value=1000.0, value=50_000.0, step=1000.0
    )
    strategy = st.selectbox(
        "Strategy type", ["momentum", "mean-reversion", "trend-following", "value"]
    )
    risk_level = st.slider("Risk level (1–10)", min_value=1, max_value=10, value=5)
    submitted = st.form_submit_button("Run AutoHedge")

if submitted:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if not tickers:
        st.error("Please provide at least one valid ticker.")
    else:
        with st.spinner("Running AutoHedge..."):
            try:
                result = run_autohedge(
                    stocks=tickers,
                    allocation_usd=allocation,
                    strategy_type=strategy,
                    risk_level=risk_level,
                )
                st.success("Run completed and saved. Adjust filters to see it above.")
                st.json(result)
            except Exception as e:
                st.error(f"Error running AutoHedge: {e}")
