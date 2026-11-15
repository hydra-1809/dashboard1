import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import math

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Q12.18 | VLE Dashboard",
    page_icon="⚗️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp { background: #0b1220; color: #e5e7eb; }
.block-container { padding-top: 2rem; max-width: 1400px; }
h1, h2, h3 { color: #e2e8f0 !important; }
div[data-testid="stMetric"] {
    background: #172338;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #293b56;
}
section[data-testid="stSidebar"] { background: #111c30; }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("⚗️ VLE Thermodynamics Laboratory")
st.caption("ASSIGNMENT 3  /  QUESTION 12.18  /  WILSON MODEL")

st.markdown("""
BUBL T, DEW T, P,T-flash and azeotrope calculations for a binary
mixture at constant pressure using the Wilson activity coefficient model.
""")

st.divider()

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.title("⚙️ Calculation Settings")

st.sidebar.subheader("System Selection")
system = st.sidebar.selectbox("Binary system", ["Acetone – Water"])

st.sidebar.subheader("Operating Conditions")
P = st.sidebar.number_input(
    "Pressure (kPa)", min_value=10.0, max_value=500.0, value=101.33, step=1.0
)

x1_bubl = st.sidebar.number_input("(a) BUBL T: x1", min_value=0.001, max_value=0.999, value=0.3, step=0.05)
y1_dew = st.sidebar.number_input("(b) DEW T: y1", min_value=0.001, max_value=0.999, value=0.3, step=0.05)
z1_flash = st.sidebar.number_input("(c) Flash: z1", min_value=0.001, max_value=0.999, value=0.3, step=0.05)

st.sidebar.subheader("Antoine Constants")

st.sidebar.markdown("**Component 1: Acetone**")
A1 = st.sidebar.number_input("A1", value=14.3145, format="%.5f")
B1 = st.sidebar.number_input("B1", value=2756.22)
C1 = st.sidebar.number_input("C1", value=228.060)

st.sidebar.markdown("**Component 2: Water**")
A2 = st.sidebar.number_input("A2", value=16.3872, format="%.5f")
B2 = st.sidebar.number_input("B2", value=3885.70)
C2 = st.sidebar.number_input("C2", value=230.170)

st.sidebar.subheader("Wilson Parameters")
V1 = st.sidebar.number_input("V1 (cm³/mol)", value=74.05)
V2 = st.sidebar.number_input("V2 (cm³/mol)", value=18.07)
a12 = st.sidebar.number_input("a12 (cal/mol)", value=291.27)
a21 = st.sidebar.number_input("a21 (cal/mol)", value=1448.01)

# ---------------- CONSTANTS ----------------
R = 1.987204  # cal/(mol K)

# ---------------- EQUATIONS ----------------
def antoine(A, B, C, temp_C):
    return math.exp(A - B / (temp_C + C))

def antoine_T(A, B, C, Pressure):
    return B / (A - math.log(Pressure)) - C

def wilson(x1, T):
    x2 = 1.0 - x1
    L12 = (V2 / V1) * math.exp(-a12 / (R * T))
    L21 = (V1 / V2) * math.exp(-a21 / (R * T))
    term = L12 / (x1 + L12 * x2) - L21 / (x2 + L21 * x1)
    ln_g1 = -math.log(x1 + L12 * x2) + x2 * term
    ln_g2 = -math.log(x2 + L21 * x1) - x1 * term
    return math.exp(ln_g1), math.exp(ln_g2)

def bisect(f, low, high, n=100):
    for _ in range(n):
        mid = 0.5 * (low + high)
        if f(low) * f(mid) <= 0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)

# ---------------- (a) BUBL T ----------------
def bubl_t(x1):
    x2 = 1.0 - x1

    def f(T_C):
        g1, g2 = wilson(x1, T_C + 273.15)
        return (x1 * g1 * antoine(A1, B1, C1, T_C)
                + x2 * g2 * antoine(A2, B2, C2, T_C)) - P

    T_C = bisect(f, 0.0, 200.0)
    g1, g2 = wilson(x1, T_C + 273.15)
    y1 = x1 * g1 * antoine(A1, B1, C1, T_C) / P
    return T_C, y1, g1, g2

# ---------------- (b) DEW T (nested iteration) ----------------
def dew_t(y1):
    y2 = 1.0 - y1
    x1 = y1
    T_C = 50.0
    for _ in range(500):
        def g(Tc):
            g1, g2 = wilson(x1, Tc + 273.15)
            return (y1 * P / (g1 * antoine(A1, B1, C1, Tc))
                    + y2 * P / (g2 * antoine(A2, B2, C2, Tc))) - 1.0

        T_C = bisect(g, 0.0, 200.0)
        g1, g2 = wilson(x1, T_C + 273.15)
        x1_new = y1 * P / (g1 * antoine(A1, B1, C1, T_C))
        x2_new = y2 * P / (g2 * antoine(A2, B2, C2, T_C))
        x1_new = x1_new / (x1_new + x2_new)
        if abs(x1_new - x1) < 1e-12:
            x1 = x1_new
            break
        x1 = x1_new
    g1, g2 = wilson(x1, T_C + 273.15)
    return T_C, x1, g1, g2

# ---------------- (c) P,T-FLASH ----------------
def flash(z1, T_C):
    z2 = 1.0 - z1
    T = T_C + 273.15
    P1s = antoine(A1, B1, C1, T_C)
    P2s = antoine(A2, B2, C2, T_C)
    x1 = z1
    y1 = z1
    V = 0.5
    for _ in range(1000):
        g1, g2 = wilson(x1, T)
        K1 = g1 * P1s / P
        K2 = g2 * P2s / P

        def rr(v):
            return z1 * (K1 - 1) / (1 + v * (K1 - 1)) + z2 * (K2 - 1) / (1 + v * (K2 - 1))

        if rr(0.0) <= 0:
            V = 0.0
        elif rr(1.0) >= 0:
            V = 1.0
        else:
            V = bisect(rr, 0.0, 1.0)

        x1_new = z1 / (1 + V * (K1 - 1))
        x2_new = z2 / (1 + V * (K2 - 1))
        s = x1_new + x2_new
        x1_new, x2_new = x1_new / s, x2_new / s
        y1_new = K1 * x1_new
        y2_new = K2 * x2_new
        y1_new = y1_new / (y1_new + y2_new)

        if abs(x1_new - x1) < 1e-12:
            x1, y1 = x1_new, y1_new
            break
        x1, y1 = x1_new, y1_new
    return V, x1, y1

# ---------------- (d) AZEOTROPE ----------------
def az_function(x1):
    return bubl_t(x1)[1] - x1   # y1 - x1 along the bubble curve

def find_azeotrope():
    xs = np.linspace(0.001, 0.999, 400)
    fs = [az_function(x) for x in xs]
    for i in range(len(xs) - 1):
        if fs[i] * fs[i + 1] < 0:
            x_az = bisect(az_function, xs[i], xs[i + 1], 60)
            T_az = bubl_t(x_az)[0]
            return x_az, T_az
    return None

# ---------------- RUN CALCULATIONS ----------------
T_a, y1_a, g1_a, g2_a = bubl_t(x1_bubl)
T_b, x1_b, g1_b, g2_b = dew_t(y1_dew)

T_bub_z = bubl_t(z1_flash)[0]
T_dew_z = dew_t(z1_flash)[0]
T_flash = 0.5 * (T_bub_z + T_dew_z)
V_f, x1_f, y1_f = flash(z1_flash, T_flash)

az = find_azeotrope()

Tb1 = antoine_T(A1, B1, C1, P)
Tb2 = antoine_T(A2, B2, C2, P)

# ---------------- METRICS ----------------
st.subheader("01 | System Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Pressure", f"{P:.2f} kPa")
c2.metric("Acetone boiling point", f"{Tb1:.2f} °C")
c3.metric("Water boiling point", f"{Tb2:.2f} °C")

st.divider()

# ---------------- RESULTS ----------------
st.subheader("02 | Results")

st.markdown(f"**(a) BUBL T** at P = {P:.2f} kPa, x1 = {x1_bubl}")
a1, a2, a3, a4 = st.columns(4)
a1.metric("Bubble T", f"{T_a + 273.15:.3f} K ({T_a:.2f} °C)")
a2.metric("y1", f"{y1_a:.5f}")
a3.metric("γ1", f"{g1_a:.5f}")
a4.metric("γ2", f"{g2_a:.5f}")

st.markdown(f"**(b) DEW T** at P = {P:.2f} kPa, y1 = {y1_dew}")
b1, b2, b3, b4 = st.columns(4)
b1.metric("Dew T", f"{T_b + 273.15:.3f} K ({T_b:.2f} °C)")
b2.metric("x1", f"{x1_b:.5f}")
b3.metric("γ1", f"{g1_b:.5f}")
b4.metric("γ2", f"{g2_b:.5f}")

st.markdown(f"**(c) P,T-flash** at P = {P:.2f} kPa, z1 = {z1_flash}, T = ½(T_bubble + T_dew)")
c1_, c2_, c3_, c4_ = st.columns(4)
c1_.metric("Flash T", f"{T_flash + 273.15:.3f} K ({T_flash:.2f} °C)")
c2_.metric("Vapor fraction V", f"{V_f:.5f}")
c3_.metric("x1 (liquid)", f"{x1_f:.5f}")
c4_.metric("y1 (vapor)", f"{y1_f:.5f}")

st.markdown(f"**(d) Azeotrope** at P = {P:.2f} kPa")
if az is None:
    st.info("No azeotrope exists at this pressure for the Wilson model "
            "(y1 − x1 does not change sign along the bubble curve for 0 < x1 < 1).")
else:
    d1, d2 = st.columns(2)
    d1.metric("x1_az = y1_az", f"{az[0]:.5f}")
    d2.metric("T_az", f"{az[1] + 273.15:.3f} K ({az[1]:.2f} °C)")

st.divider()

# ---------------- T-X-Y GRAPH ----------------
st.subheader("03 | Temperature – Composition Diagram")

x_values = np.linspace(0.001, 0.999, 101)
bt, vc = [], []
for x in x_values:
    T_, y_, _, _ = bubl_t(x)
    bt.append(T_)
    vc.append(y_)

plot_df = pd.concat([
    pd.DataFrame({
        "Mole fraction of acetone": x_values,
        "Temperature (°C)": bt,
        "Curve": "Bubble Point (T-x1)"
    }),
    pd.DataFrame({
        "Mole fraction of acetone": vc,
        "Temperature (°C)": bt,
        "Curve": "Dew Point (T-y1)"
    })
], ignore_index=True)

fig = px.line(
    plot_df,
    x="Mole fraction of acetone",
    y="Temperature (°C)",
    color="Curve",
    color_discrete_map={
        "Bubble Point (T-x1)": "#38bdf8",
        "Dew Point (T-y1)": "#fb923c"
    },
    template="plotly_dark"
)
fig.update_traces(line=dict(width=3))
fig.update_layout(
    height=600,
    legend=dict(orientation="h", y=1.12, x=0, title_text=""),
    paper_bgcolor="#0b1220",
    plot_bgcolor="#111c30",
    margin=dict(l=40, r=30, t=50, b=50)
)
fig.update_xaxes(range=[0, 1], gridcolor="#293b56")
fig.update_yaxes(gridcolor="#293b56")

st.plotly_chart(fig, use_container_width=True)

# ---------------- SUMMARY TABLE ----------------
st.subheader("04 | Summary Table")

summary = pd.DataFrame({
    "Part": ["(a) BUBL T", "(b) DEW T", "(c) P,T-flash"],
    "T (K)": [T_a + 273.15, T_b + 273.15, T_flash + 273.15],
    "T (°C)": [T_a, T_b, T_flash],
    "x1": [x1_bubl, x1_b, x1_f],
    "y1": [y1_a, y1_dew, y1_f]
})
st.dataframe(summary.round(5), use_container_width=True)

csv = summary.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Results CSV",
    data=csv,
    file_name="Q12_18_Acetone_Water.csv",
    mime="text/csv"
)

# ---------------- FORMULAS ----------------
with st.expander("View equations and calculation methodology"):
    st.markdown("""
    **BUBL T:** find T such that x1 γ1 P1sat(T) + x2 γ2 P2sat(T) = P
    (bisection; γ recomputed at every trial T). Then y1 = x1 γ1 P1sat / P.

    **DEW T:** find T such that y1 P/(γ1 P1sat) + y2 P/(γ2 P2sat) = 1, with
    xi = yi P/(γi Pisat). γ depends on x and T, so x is updated and the
    T-search is repeated until x converges.

    **P,T-flash:** T = ½(T_bubble + T_dew) at z1. Ki = γi Pisat / P.
    The Rachford-Rice equation is solved for V by bisection, then
    xi = zi/(1+V(Ki−1)) and yi = Ki xi, iterating until x converges.

    **Azeotrope:** at x1 = y1, γ1 P1sat = γ2 P2sat. The code scans x1 along
    the bubble-T curve for a sign change of (y1 − x1) and refines it by bisection.

    **Wilson equation:** Λ12 = (V2/V1) exp(−a12/RT), Λ21 = (V1/V2) exp(−a21/RT)

    **Gas constant:** R = 1.987204 cal/(mol K)
    """)

st.divider()

st.markdown("""
<div style="text-align:center;color:#94a3b8;">
VLE Thermodynamics Laboratory | Assignment 3 | Question 12.18
<br>
Wilson Equation | Acetone – Water
</div>
""", unsafe_allow_html=True)
