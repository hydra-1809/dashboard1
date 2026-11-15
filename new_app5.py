import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import math

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Q12.17 | VLE Dashboard",
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
st.caption("ASSIGNMENT 3  /  QUESTION 12.17  /  NRTL MODEL")

st.markdown("""
BUBL P, DEW P, P,T-flash and azeotrope calculations for a binary
mixture at constant temperature using the NRTL activity coefficient model.
""")

st.divider()

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.title("⚙️ Calculation Settings")

st.sidebar.subheader("System Selection")
system = st.sidebar.selectbox("Binary system", ["Acetone – Water"])

st.sidebar.subheader("Operating Conditions")
T_C = st.sidebar.number_input(
    "Temperature (°C)", min_value=0.0, max_value=150.0, value=60.0, step=1.0
)
T = T_C + 273.15

x1_bubl = st.sidebar.number_input("(a) BUBL P: x1", min_value=0.001, max_value=0.999, value=0.3, step=0.05)
y1_dew = st.sidebar.number_input("(b) DEW P: y1", min_value=0.001, max_value=0.999, value=0.3, step=0.05)
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

st.sidebar.subheader("NRTL Parameters")
b12 = st.sidebar.number_input("b12 (cal/mol)", value=631.05)
b21 = st.sidebar.number_input("b21 (cal/mol)", value=1197.41)
alpha = st.sidebar.number_input("alpha (α12)", value=0.5343, format="%.4f")

# ---------------- CONSTANTS ----------------
R = 1.987204  # cal/(mol K)

# ---------------- EQUATIONS ----------------
def antoine(A, B, C, temp_C):
    return math.exp(A - B / (temp_C + C))

def nrtl(x1, T):
    x2 = 1.0 - x1
    tau12 = b12 / (R * T)
    tau21 = b21 / (R * T)
    G12 = math.exp(-alpha * tau12)
    G21 = math.exp(-alpha * tau21)
    ln_g1 = x2**2 * (
        tau21 * (G21 / (x1 + x2 * G21))**2
        + tau12 * G12 / (x2 + x1 * G12)**2
    )
    ln_g2 = x1**2 * (
        tau12 * (G12 / (x2 + x1 * G12))**2
        + tau21 * G21 / (x1 + x2 * G21)**2
    )
    return math.exp(ln_g1), math.exp(ln_g2)

P1_sat = antoine(A1, B1, C1, T_C)
P2_sat = antoine(A2, B2, C2, T_C)

# ---------------- (a) BUBL P ----------------
def bubl_p(x1):
    x2 = 1.0 - x1
    g1, g2 = nrtl(x1, T)
    P = x1 * g1 * P1_sat + x2 * g2 * P2_sat
    y1 = x1 * g1 * P1_sat / P
    return P, y1, g1, g2

# ---------------- (b) DEW P (iterative) ----------------
def dew_p(y1):
    y2 = 1.0 - y1
    x1 = y1
    P = 1.0
    for _ in range(500):
        g1, g2 = nrtl(x1, T)
        P = 1.0 / (y1 / (g1 * P1_sat) + y2 / (g2 * P2_sat))
        x1_new = y1 * P / (g1 * P1_sat)
        x2_new = y2 * P / (g2 * P2_sat)
        x1_new = x1_new / (x1_new + x2_new)
        if abs(x1_new - x1) < 1e-12:
            x1 = x1_new
            break
        x1 = x1_new
    g1, g2 = nrtl(x1, T)
    return P, x1, g1, g2

# ---------------- (c) P,T-FLASH ----------------
def flash(z1, P):
    z2 = 1.0 - z1
    x1 = z1
    y1 = z1
    V = 0.5
    for _ in range(1000):
        g1, g2 = nrtl(x1, T)
        K1 = g1 * P1_sat / P
        K2 = g2 * P2_sat / P

        def rr(v):
            return z1 * (K1 - 1) / (1 + v * (K1 - 1)) + z2 * (K2 - 1) / (1 + v * (K2 - 1))

        if rr(0.0) <= 0:
            V = 0.0
        elif rr(1.0) >= 0:
            V = 1.0
        else:
            lo, hi = 0.0, 1.0
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                if rr(lo) * rr(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            V = 0.5 * (lo + hi)

        x1_new = z1 / (1 + V * (K1 - 1))
        x2_new = z2 / (1 + V * (K2 - 1))
        s = x1_new + x2_new
        x1_new, x2_new = x1_new / s, x2_new / s
        y1_new = K1 * x1_new
        y2_new = K2 * x2_new
        s = y1_new + y2_new
        y1_new = y1_new / s

        if abs(x1_new - x1) < 1e-12:
            x1, y1 = x1_new, y1_new
            break
        x1, y1 = x1_new, y1_new
    return V, x1, y1

# ---------------- (d) AZEOTROPE ----------------
def az_function(x1):
    g1, g2 = nrtl(x1, T)
    return math.log(g1 * P1_sat) - math.log(g2 * P2_sat)

def find_azeotrope():
    xs = np.linspace(0.0001, 0.9999, 2001)
    fs = [az_function(x) for x in xs]
    for i in range(len(xs) - 1):
        if fs[i] * fs[i + 1] < 0:
            lo, hi = xs[i], xs[i + 1]
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                if az_function(lo) * az_function(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            x_az = 0.5 * (lo + hi)
            P_az = bubl_p(x_az)[0]
            return x_az, P_az
    return None

# ---------------- RUN CALCULATIONS ----------------
P_a, y1_a, g1_a, g2_a = bubl_p(x1_bubl)
P_b, x1_b, g1_b, g2_b = dew_p(y1_dew)

P_bub_z = bubl_p(z1_flash)[0]
P_dew_z = dew_p(z1_flash)[0]
P_flash = 0.5 * (P_bub_z + P_dew_z)
V_f, x1_f, y1_f = flash(z1_flash, P_flash)

az = find_azeotrope()

# ---------------- METRICS ----------------
st.subheader("01 | System Overview")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Temperature", f"{T:.2f} K")
with c2:
    st.metric("Acetone Psat", f"{P1_sat:.3f} kPa")
with c3:
    st.metric("Water Psat", f"{P2_sat:.3f} kPa")

st.divider()

# ---------------- RESULTS ----------------
st.subheader("02 | Results")

st.markdown(f"**(a) BUBL P** at T = {T:.2f} K, x1 = {x1_bubl}")
a1, a2, a3, a4 = st.columns(4)
a1.metric("Bubble pressure", f"{P_a:.4f} kPa")
a2.metric("y1", f"{y1_a:.5f}")
a3.metric("γ1", f"{g1_a:.5f}")
a4.metric("γ2", f"{g2_a:.5f}")

st.markdown(f"**(b) DEW P** at T = {T:.2f} K, y1 = {y1_dew}")
b1, b2, b3, b4 = st.columns(4)
b1.metric("Dew pressure", f"{P_b:.4f} kPa")
b2.metric("x1", f"{x1_b:.5f}")
b3.metric("γ1", f"{g1_b:.5f}")
b4.metric("γ2", f"{g2_b:.5f}")

st.markdown(f"**(c) P,T-flash** at T = {T:.2f} K, z1 = {z1_flash}, P = ½(P_bubble + P_dew)")
c1_, c2_, c3_, c4_ = st.columns(4)
c1_.metric("Flash pressure", f"{P_flash:.4f} kPa")
c2_.metric("Vapor fraction V", f"{V_f:.5f}")
c3_.metric("x1 (liquid)", f"{x1_f:.5f}")
c4_.metric("y1 (vapor)", f"{y1_f:.5f}")

st.markdown(f"**(d) Azeotrope** at T = {T:.2f} K")
if az is None:
    st.info("No azeotrope exists at this temperature for the NRTL model "
            "(ln(γ1·P1sat / γ2·P2sat) does not change sign for 0 < x1 < 1).")
else:
    d1, d2 = st.columns(2)
    d1.metric("x1_az = y1_az", f"{az[0]:.5f}")
    d2.metric("P_az", f"{az[1]:.4f} kPa")

st.divider()

# ---------------- P-X-Y GRAPH ----------------
st.subheader("03 | Pressure – Composition Diagram")

x_values = np.linspace(0.0001, 0.9999, 201)
bp, vc = [], []
for x in x_values:
    P_, y_, _, _ = bubl_p(x)
    bp.append(P_)
    vc.append(y_)

plot_df = pd.concat([
    pd.DataFrame({
        "Mole fraction of acetone": x_values,
        "Pressure (kPa)": bp,
        "Curve": "Bubble Point (P-x1)"
    }),
    pd.DataFrame({
        "Mole fraction of acetone": vc,
        "Pressure (kPa)": bp,
        "Curve": "Dew Point (P-y1)"
    })
], ignore_index=True)

fig = px.line(
    plot_df,
    x="Mole fraction of acetone",
    y="Pressure (kPa)",
    color="Curve",
    color_discrete_map={
        "Bubble Point (P-x1)": "#38bdf8",
        "Dew Point (P-y1)": "#fb923c"
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

# ---------------- NRTL PARAMETERS ----------------
st.subheader("04 | NRTL Parameters")

tau12 = b12 / (R * T)
tau21 = b21 / (R * T)
p1, p2, p3, p4 = st.columns(4)
p1.metric("τ12", f"{tau12:.5f}")
p2.metric("τ21", f"{tau21:.5f}")
p3.metric("G12", f"{math.exp(-alpha * tau12):.5f}")
p4.metric("G21", f"{math.exp(-alpha * tau21):.5f}")

# ---------------- SUMMARY TABLE ----------------
st.subheader("05 | Summary Table")

summary = pd.DataFrame({
    "Part": ["(a) BUBL P", "(b) DEW P", "(c) P,T-flash"],
    "P (kPa)": [P_a, P_b, P_flash],
    "x1": [x1_bubl, x1_b, x1_f],
    "y1": [y1_a, y1_dew, y1_f]
})
st.dataframe(summary.round(5), use_container_width=True)

csv = summary.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Results CSV",
    data=csv,
    file_name="Q12_17_Acetone_Water_NRTL.csv",
    mime="text/csv"
)

# ---------------- FORMULAS ----------------
with st.expander("View equations and calculation methodology"):
    st.markdown("""
    **BUBL P:** P = x1 γ1 P1sat + x2 γ2 P2sat,  y1 = x1 γ1 P1sat / P

    **DEW P:** 1/P = y1/(γ1 P1sat) + y2/(γ2 P2sat), with xi = yi P/(γi Pisat).
    γ depends on x, so x is iterated until it converges.

    **P,T-flash:** P = ½(P_bubble + P_dew) at z1. Ki = γi Pisat / P.
    The Rachford-Rice equation is solved for V by bisection, then
    xi = zi/(1+V(Ki−1)) and yi = Ki xi, iterating until x converges.

    **Azeotrope:** x1 = y1 requires γ1 P1sat = γ2 P2sat. The code scans x1
    for a sign change and refines it by bisection.

    **NRTL equation**

    τ12 = b12/(RT),  τ21 = b21/(RT)

    G12 = exp(-α τ12),  G21 = exp(-α τ21)

    ln γ1 = x2² [ τ21 (G21/(x1 + x2 G21))² + τ12 G12/(x2 + x1 G12)² ]

    ln γ2 = x1² [ τ12 (G12/(x2 + x1 G12))² + τ21 G21/(x1 + x2 G21)² ]

    **Gas constant:** R = 1.987204 cal/(mol K)
    """)

st.divider()

st.markdown("""
<div style="text-align:center;color:#94a3b8;">
VLE Thermodynamics Laboratory | Assignment 3 | Question 12.17
<br>
NRTL Equation | Acetone – Water
</div>
""", unsafe_allow_html=True)
