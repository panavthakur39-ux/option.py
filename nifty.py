"""
Streamlit app: Simulate an "algo pushes then dumps" option-market scenario

Save this file as `streamlit_app.py` (or the name you prefer) and push to GitHub.
Run locally with:
    pip install streamlit pandas numpy
    streamlit run streamlit_app.py

This single-file app provides interactive controls in the sidebar to configure
fair price, initial quotes, human order schedule, algo behavior and then
runs a simplified discrete-time market simulation. Outputs (no graphs):
 - Price history table (downloadable CSV)
 - Trade log (downloadable CSV)
 - Positions summary

This is an educational toy model — not a market microstructure-accurate
simulation. Use it for demonstration and learning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="Algo Push-and-Dump Simulator (No Graphs)", layout="wide")

st.title("Algo Push-and-Dump — Option Market Simulator (Tables Only)")
st.markdown(
    """
    This toy simulator models a simplified scenario where an algorithm posts
    wide passive quotes, buys into the market to push the price when a human
    buyer shows interest, then dumps inventory to that buyer once price
    reaches a threshold (e.g., 20% above fair value). The algo then returns
    to passive quotes. Use controls on the left to change parameters.

    This version intentionally does not plot any graphs — it only shows tables
    and CSV downloads suitable for pushing to GitHub or analysis in a spreadsheet.
    """
)

# --- Sidebar controls ---
st.sidebar.header("Simulation parameters")
fair_price = st.sidebar.number_input("Fair price (option)", value=40.0, step=1.0)
initial_bid = st.sidebar.number_input("Algo passive bid", value=20.0, step=1.0)
initial_ask = st.sidebar.number_input("Algo passive ask", value=100.0, step=1.0)
threshold_pct = st.sidebar.slider("Dump threshold (% above fair)", 0, 200, 20)
threshold = fair_price * (1 + threshold_pct / 100.0)

time_steps = st.sidebar.number_input("Time steps (discrete)", min_value=5, max_value=1000, value=40)

st.sidebar.markdown("---")
st.sidebar.header("Algo behavior")
algo_buy_step = st.sidebar.number_input("Algo buy price step (impact per buy)", value=1.0, step=0.1)
algo_buy_size = st.sidebar.number_input("Algo buy size per push", value=20, step=1)
algo_sell_size = st.sidebar.number_input("Algo dump sell size", value=100, step=1)

st.sidebar.markdown("---")
st.sidebar.header("Human orders (comma-separated list)
Format: t:size:limit  (e.g. 1:10:21,3:15:25)")
human_orders_text = st.sidebar.text_area("Human order list", value="1:10:21,3:15:25,6:20:35,10:30:60")

st.sidebar.markdown("---")
if st.sidebar.button("Run simulation"):
    run_simulation = True
else:
    run_simulation = False

# --- Helper: parse human orders ---

def parse_human_orders(text):
    orders = []
    if not text.strip():
        return orders
    parts = [p.strip() for p in text.split(",") if p.strip()]
    for p in parts:
        try:
            t_str, size_str, limit_str = p.split(":")
            orders.append({"t": int(t_str), "size": int(size_str), "type": "buy", "price_limit": float(limit_str)})
        except Exception:
            # ignore malformed entries
            continue
    return orders

human_orders = parse_human_orders(human_orders_text)

# --- Simulation function ---

def simulate(fair_price, initial_bid, initial_ask, threshold, time_steps, human_orders,
             algo_buy_step, algo_buy_size, algo_sell_size):
    price = fair_price
    current_quotes = {"bid": initial_bid, "ask": initial_ask}

    price_history = []
    trade_log = []
    human_inventory = 0
    human_cash = 0.0
    algo_inventory = 0
    algo_cash = 0.0

    def record_trade(t, buyer, seller, price_, size, reason):
        trade_log.append({"time": t, "buyer": buyer, "seller": seller, "price": price_, "size": size, "reason": reason})

    for t in range(time_steps + 1):
        price_history.append({"time": t, "price": price, "bid": current_quotes["bid"], "ask": current_quotes["ask"]})
        matching_orders = [o for o in human_orders if o["t"] == t]
        for order in matching_orders:
            if order["type"] == "buy":
                # initial fill at current price or at human limit, whichever is higher
                fill_price = max(price, order["price_limit"]) if order["price_limit"] > price else price
                remaining = order["size"]
                immediate_fill = min(remaining, max(1, int(order["size"] * 0.2)))
                human_inventory += immediate_fill
                human_cash -= immediate_fill * fill_price
                record_trade(t, "human", "algo", fill_price, immediate_fill, "human initial fill")
                remaining -= immediate_fill

                # algo pushes price up by buying into market until threshold or human filled
                while remaining > 0 and price < threshold:
                    buy_size = min(remaining, algo_buy_size)
                    price = price + algo_buy_step
                    algo_inventory += buy_size
                    algo_cash -= buy_size * price
                    record_trade(t, "algo", "hidden", price, buy_size, "algo pushes price up")

                    chase_fill = min(remaining, int(max(1, buy_size * 0.5)))
                    human_inventory += chase_fill
                    human_cash -= chase_fill * price
                    record_trade(t, "human", "algo", price, chase_fill, "human chases higher price")
                    remaining -= chase_fill

                # If threshold reached, dump
                if price >= threshold and remaining > 0:
                    dump_price = threshold
                    sell_size = min(remaining, algo_sell_size)
                    algo_inventory -= sell_size
                    algo_cash += sell_size * dump_price
                    human_inventory += sell_size
                    human_cash -= sell_size * dump_price
                    record_trade(t, "algo", "human", dump_price, sell_size, "algo dumps at inflated price")
                    remaining -= sell_size
                    current_quotes["bid"] = initial_bid
                    current_quotes["ask"] = initial_ask
                    price = dump_price

                # any remaining fills at current price
                if remaining > 0 and price < threshold:
                    human_inventory += remaining
                    human_cash -= remaining * price
                    record_trade(t, "human", "hidden", price, remaining, "remaining filled at current price")
                    remaining = 0

        dumped_this_t = any(tr["time"] == t and tr["reason"] == "algo dumps at inflated price" for tr in trade_log)
        if not dumped_this_t:
            price += (fair_price - price) * 0.03

    # post-simulation revert toward fair (illustrative)
    for _ in range(6):
        price += (fair_price - price) * 0.25
        price_history.append({"time": time_steps + 1 + _, "price": price, "bid": current_quotes["bid"], "ask": current_quotes["ask"]})

    df_prices = pd.DataFrame(price_history)
    df_trades = pd.DataFrame(trade_log)

    human_unrealized = human_inventory * fair_price + human_cash
    summary = pd.DataFrame([
        {"actor": "human", "inventory": human_inventory, "cash": human_cash, "market_value_at_fair": human_inventory * fair_price, "unrealized_PnL_if_price_returns_to_fair": human_unrealized},
        {"actor": "algo", "inventory": algo_inventory, "cash": algo_cash, "mark_to_market_at_fair": algo_inventory * fair_price + algo_cash}
    ])

    return df_prices, df_trades, summary

# --- Execute simulation if requested ---
if run_simulation := run_simulation:
    with st.spinner("Running simulation..."):
        df_prices, df_trades, summary = simulate(
            fair_price, initial_bid, initial_ask, threshold, int(time_steps), human_orders,
            algo_buy_step, int(algo_buy_size), int(algo_sell_size)
        )

    col1, col2 = st.columns((2, 1))

    with col1:
        st.subheader("Price history (table)")
        st.dataframe(df_prices)
        # download
        csv_buf_prices = io.StringIO()
        df_prices.to_csv(csv_buf_prices, index=False)
        st.download_button("Download price history CSV", csv_buf_prices.getvalue(), file_name="price_history.csv", mime="text/csv")

    with col2:
        st.subheader("Trade log")
        if df_trades.empty:
            st.info("No trades executed with the current parameters. Try adjusting human orders or algo behavior.")
        else:
            st.dataframe(df_trades)
            csv_buf = io.StringIO()
            df_trades.to_csv(csv_buf, index=False)
            st.download_button("Download trade log CSV", csv_buf.getvalue(), file_name="trade_log.csv", mime="text/csv")

        st.subheader("Positions summary")
        st.dataframe(summary)

    st.markdown("---")
    st.caption("Notes: toy model. Real markets are more complex (order book depth, multiple participants, latency, exchange rules, fees). Use responsibly.")

else:
    st.info("Adjust parameters in the sidebar and click 'Run simulation' to execute.")
    st.caption("Suggested human orders example: 1:10:21,3:15:25,6:20:35,10:30:60")

# EOF
