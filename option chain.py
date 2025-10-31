import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------
# App Title
# -------------------------------
st.set_page_config(page_title="Option Manipulation Simulation", layout="centered")
st.title("📈 Non-Liquid Option Algo Simulation (No Graph)")
st.markdown("""
This simulation demonstrates how an **algorithm (acting as both buyer and seller)**  
can manipulate prices in a **non-liquid option** scenario.

**Process Overview:**
- Algo starts as Buyer @ ₹20 and Seller @ ₹80  
- Fair Price = ₹40  
- Human places a Buy @ ₹21  
- Algo reacts and moves bid to ₹22  
- Momentum pushes price up above fair value  
- Once 20% above fair (₹48), Algo sells at inflated price to the human  
- Algo then resets back to ₹20/₹80 — leaving the human at a loss.
""")

# -------------------------------
# Sidebar Parameters
# -------------------------------
st.sidebar.header("🔧 Simulation Parameters")
fair_price = st.sidebar.number_input("Fair Value of Option", 20.0, 200.0, 40.0, step=1.0)
initial_bid = st.sidebar.number_input("Initial Algo Bid", 1.0, 200.0, 20.0, step=1.0)
initial_ask = st.sidebar.number_input("Initial Algo Ask", 1.0, 200.0, 80.0, step=1.0)
human_limit_buy = st.sidebar.number_input("Human Buy Order Price", 1.0, 200.0, 21.0, step=1.0)
algo_step = st.sidebar.slider("Algo Aggressive Step", 0.5, 5.0, 1.0, 0.5)
max_steps = st.sidebar.slider("Simulation Steps", 10, 100, 30, 5)

# -------------------------------
# Simulation Logic
# -------------------------------
bid = initial_bid
ask = initial_ask
reset_bid, reset_ask = initial_bid, initial_ask
fair_threshold = fair_price * 1.20

logs = []
human_position = 0
human_avg_price = None

for t in range(max_steps):
    event = ""
    mid = (bid + ask) / 2
    logs.append({"t": t, "bid": bid, "ask": ask, "mid": mid, "event": event})

    # Step 1: Human posts buy order
    if t == 1:
        event += f"Human posts buy @ ₹{human_limit_buy:.2f}. "

    # Step 2: Algo reacts
    if t >= 1 and human_limit_buy > bid:
        bid = human_limit_buy + algo_step
        ask = max(ask, bid + 2)
        event += f"Algo bumps bid to ₹{bid:.2f}. "

    # Step 3: Random market momentum
    if t >= 2:
        bid += np.random.choice([0.5, 1.0, 2.0])
        ask = max(ask, bid + np.random.choice([2.0, 3.0, 5.0]))
        event += "Momentum buyers push price up. "

    # Step 4: Price exceeds 20% above fair
    if (bid + ask) / 2 >= fair_threshold:
        sell_price = max(ask, fair_threshold)
        if human_position == 0:
            human_position = 1
            human_avg_price = sell_price
            event += f"Algo sells to human @ ₹{sell_price:.2f}. "
        bid, ask = reset_bid, reset_ask
        logs.append({
            "t": t + 0.1,
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask)/2,
            "event": f"Algo resets quotes to ₹{bid:.2f}/₹{ask:.2f}"
        })
        break

    logs[-1]["event"] = event

# -------------------------------
# Display Results
# -------------------------------
df = pd.DataFrame(logs)
df["mid"] = (df["bid"] + df["ask"]) / 2

st.subheader("📜 Market Simulation Log")
st.dataframe(df, use_container_width=True)

if human_position > 0:
    pnl = (fair_price - human_avg_price) * human_position
    st.subheader("💰 Human Trade Summary")
    st.write(f"**Human Buy Price:** ₹{human_avg_price:.2f}")
    st.write(f"**Fair Value:** ₹{fair_price:.2f}")
    st.write(f"**Unrealized P&L:** ₹{pnl:.2f} (Loss if negative)")
else:
    st.info("No trade executed with the human yet.")

# -------------------------------
# Download Option
# -------------------------------
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Simulation Log (CSV)", csv, "simulation_log.csv", "text/csv")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("🧠 *Educational use only — demonstrates illiquid option manipulation by algorithms.*")
