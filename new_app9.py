import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import math

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Q12.21 | VLE Dashboard",
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
st.caption("ASSIGNMENT 3  /  QUESTION 12.21  /  NRTL MODEL  /  TERNARY")

st.markdown("""
BUBL P, DEW P and P,T-flash calculations for the
**acetone(1) / methanol(2) / water(3)** system at constant temperature
using the multicomponent NRTL equation.
""")

st.divider()

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.title("⚙️ Calculation Settings")

st.sidebar.subheader("Operating Conditions")
T = st.sidebar.number_input(
    "Temperature (K)", min_value=273.15, max_value=400.0, value=338.15, step=1.0
)
T_C = T - 273.15

st.sidebar.markdown("**(a) BUBL P liquid composition**")
xa1 = st.sidebar.number_input("x1 (acetone)", 0.001, 0.998, 0.3, 0.05, key="xa1")
xa2 = st.sidebar.number_input("x2 (methanol)", 0.001, 0.998, 0.4, 0.05, key="xa2")

st.sidebar.markdown("**(b) DEW P vapor composition**")
yb1 = st.sidebar.number_input("y1 (acetone)", 0.001, 0.998, 0.3, 0.05, key="yb1")
yb2 = st.sidebar.number_input("y2 (methanol)", 0.001, 0.998, 0.4, 0.05, key="yb2")

st.sidebar.markdown("**(c) Flash feed composition**")
zc1 = st.sidebar.number_input("z1 (acetone)", 0.001, 0.998, 0.3, 0.05, key="zc1")
zc2 = st.sidebar.number_input("z2 (methanol)", 0.001, 0.998, 0.4, 0.05, key="zc2")

st.sidebar.subheader("Antoine Constants")
st.sidebar.markdown("**Acetone (1)**")
A1 = st.sidebar.number_input("A1", value=14.3145, format="%.5f")
B1 = st.sidebar.number_input("B1", value=2756.22)
C1 = st.sidebar.number_input("C1", value=228.060)
st.sidebar.markdown("**Methanol (2)**")
A2 = st.sidebar.number_input("A2", value=16.5785, format="%.5f")
B2 = st.sidebar.number_input("B2", value=3638.27)
C2 = st.sidebar.number_input("C2", value=239.500)
st.sidebar.markdown("**Water (3)**")
A3 = st.sidebar.number_input("A3", value=16.3872, format="%.5f")
B3 = st.sidebar.number_input("B3", value=3885.70)
C3 = st.sidebar.number_input("C3", value=230.170)

st.sidebar.subheader("NRTL b_ij (cal/mol)")
b12 = st.sidebar.number_input("b12 (acetone/methanol)", value=184.70)
b21 = st.sidebar.number_input("b21 (methanol/acetone)", value=222.64)
b13 = st.sidebar.number_input("b13 (acetone/water)", value=631.05)
b31 = st.sidebar.number_input("b31 (water/acetone)", value=1197.41)
b23 = st.sidebar.number_input("b23 (methanol/water)", value=-253.88)
b32 = st.sidebar.number_input("b32 (water/methanol)", value=845.21)

st.sidebar.subheader("NRTL α_ij")
al12 = st.sidebar.number_input("α12 (acetone/methanol)", value=0.3084, format="%.4f")
al13 = st.sidebar.number_input("α13 (acetone/water)", value=0.5343, format="%.4f")
al23 = st.sidebar.number_input("α23 (methanol/water)", value=0.2994, format="%.4f")

# ---------------- CONSTANTS & DATA ----------------
R = 1.987204  # cal/(mol K)
names = ["Acetone", "Methanol", "Water"]

b = np.array([
    [0.0, b12, b13],
    [b21, 0.0, b23],
    [b31, b32, 0.0]
])
alpha = np.array([
    [0.0, al12, al13],
    [al12, 0.0, al23],
    [al13, al23, 0.0]
])

# ---------------- EQUATIONS ----------------
def antoine(A, B, C, temp_C):
    return math.exp(A - B / (temp_C + C))

Psat = np.array([
    antoine(A1, B1, C1, T_C),
    antoine(A2, B2, C2, T_C),
    antoine(A3, B3, C3, T_C)
])

# NRTL matrices: tau_ij = b_ij/(RT), G_ij = exp(-alpha_ij tau_ij)
tau = b / (R * T)
G = np.exp(-alpha * tau)
np.fill_diagonal(tau, 0.0)
np.fill_diagonal(G, 1.0)

def nrtl(x):
    """ln gamma_i = sum_j tau_ji G_ji x_j / sum_k G_ki x_k
       + sum_j [ x_j G_ij / sum_k G_kj x_k ] * ( tau_ij - sum_m x_m tau_mj G_mj / sum_k G_kj x_k )"""
    S = G.T @ x                         # S_j = sum_k G_kj x_k
    Q = (tau * G).T @ x                 # Q_j = sum_m x_m tau_mj G_mj
    term1 = Q / S
    term2 = np.zeros(3)
    for i in range(3):
        for j in range(3):
            term2[i] += (x[j] * G[i, j] / S[j]) * (tau[i, j] - Q[j] / S[j])
    return np.exp(term1 + term2)

# ---------------- (a) BUBL P ----------------
def bubl_p(x):
    g = nrtl(x)
    P = np.sum(x * g * Psat)
    y = x * g * Psat / P
    return P, y, g

# ---------------- (b) DEW P ----------------
def dew_p(y):
    x = y.copy()
    P = 1.0
    for _ in range(1000):
        g = nrtl(x)
        P = 1.0 / np.sum(y / (g * Psat))
        x_new = y * P / (g * Psat)
        x_new = x_new / np.sum(x_new)
        if np.max(np.abs(x_new - x)) < 1e-12:
            x = x_new
            break
        x = x_new
    g = nrtl(x)
    return P, x, g

# ---------------- (c) P,T-FLASH ----------------
def flash(z, P):
    x = z.copy()
    V_frac = 0.5
    for _ in range(2000):
        g = nrtl(x)
        K = g * Psat / P

        def rr(v):
            return np.sum(z * (K - 1) / (1 + v * (K - 1)))

        if rr(0.0) <= 0:
            V_frac = 0.0
        elif rr(1.0) >= 0:
            V_frac = 1.0
        else:
            lo, hi = 0.0, 1.0
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                if rr(lo) * rr(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            V_frac = 0.5 * (lo + hi)

        x_new = z / (1 + V_frac * (K - 1))
        x_new = x_new / np.sum(x_new)
        if np.max(np.abs(x_new - x)) < 1e-12:
            x = x_new
            break
        x = x_new
    g = nrtl(x)
    K = g * Psat / P
    y = K * x
    y = y / np.sum(y)
    return V_frac, x, y

# ---------------- COMPOSITIONS ----------------
x_a = np.array([xa1, xa2, 1.0 - xa1 - xa2])
y_b = np.array([yb1, yb2, 1.0 - yb1 - yb2])
z_c = np.array([zc1, zc2, 1.0 - zc1 - zc2])

valid = True
for label, comp in [("(a) x", x_a), ("(b) y", y_b), ("(c) z", z_c)]:
    if comp[2] <= 0:
        st.error(f"{label}: composition of water (3) is ≤ 0. Make sure x1 + x2 < 1.")
        valid = False
if not valid:
    st.stop()

# ---------------- RUN CALCULATIONS ----------------
P_a, y_a, g_a = bubl_p(x_a)
P_b, x_b, g_b = dew_p(y_b)

P_bub_z = bubl_p(z_c)[0]
P_dew_z = dew_p(z_c)[0]
P_flash = 0.5 * (P_bub_z + P_dew_z)
V_f, x_f, y_f = flash(z_c, P_flash)

# ---------------- METRICS ----------------
st.subheader("01 | System Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Temperature", f"{T:.2f} K ({T_C:.2f} °C)")
c2.metric("Acetone Psat", f"{Psat[0]:.3f} kPa")
c3.metric("Methanol Psat", f"{Psat[1]:.3f} kPa")
c4.metric("Water Psat", f"{Psat[2]:.3f} kPa")

st.divider()

# ---------------- RESULTS ----------------
st.subheader("02 | Results")

st.markdown(f"**(a) BUBL P** at T = {T:.2f} K, x = ({x_a[0]:.3f}, {x_a[1]:.3f}, {x_a[2]:.3f})")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Bubble pressure", f"{P_a:.4f} kPa")
m2.metric("y1 (acetone)", f"{y_a[0]:.5f}")
m3.metric("y2 (methanol)", f"{y_a[1]:.5f}")
m4.metric("y3 (water)", f"{y_a[2]:.5f}")

st.markdown(f"**(b) DEW P** at T = {T:.2f} K, y = ({y_b[0]:.3f}, {y_b[1]:.3f}, {y_b[2]:.3f})")
n1, n2, n3, n4 = st.columns(4)
n1.metric("Dew pressure", f"{P_b:.4f} kPa")
n2.metric("x1 (acetone)", f"{x_b[0]:.5f}")
n3.metric("x2 (methanol)", f"{x_b[1]:.5f}")
n4.metric("x3 (water)", f"{x_b[2]:.5f}")

st.markdown(
    f"**(c) P,T-flash** at T = {T:.2f} K, z = ({z_c[0]:.3f}, {z_c[1]:.3f}, {z_c[2]:.3f}), "
    f"P = ½(P_bubble + P_dew)"
)
f1, f2, f3 = st.columns(3)
f1.metric("Flash pressure", f"{P_flash:.4f} kPa")
f2.metric("Vapor fraction V", f"{V_f:.5f}")
f3.metric("Liquid fraction L", f"{1 - V_f:.5f}")

f4, f5, f6 = st.columns(3)
f4.metric("x1 / y1 (acetone)", f"{x_f[0]:.4f} / {y_f[0]:.4f}")
f5.metric("x2 / y2 (methanol)", f"{x_f[1]:.4f} / {y_f[1]:.4f}")
f6.metric("x3 / y3 (water)", f"{x_f[2]:.4f} / {y_f[2]:.4f}")

st.divider()

# ---------------- COMPOSITION CHART ----------------
st.subheader("03 | Phase Compositions")

chart_df = pd.DataFrame({
    "Component": names * 6,
    "Mole fraction": list(x_a) + list(y_a) + list(x_b) + list(y_b) + list(x_f) + list(y_f),
    "Stream": (["(a) liquid x"] * 3 + ["(a) vapor y"] * 3
               + ["(b) liquid x"] * 3 + ["(b) vapor y"] * 3
               + ["(c) liquid x"] * 3 + ["(c) vapor y"] * 3)
})

fig = px.bar(
    chart_df,
    x="Stream",
    y="Mole fraction",
    color="Component",
    barmode="stack",
    template="plotly_dark",
    color_discrete_map={
        "Acetone": "#38bdf8",
        "Methanol": "#fb923c",
        "Water": "#4ade80"
    }
)
fig.update_layout(
    height=500,
    paper_bgcolor="#0b1220",
    plot_bgcolor="#111c30",
    legend=dict(orientation="h", y=1.12, x=0, title_text=""),
    margin=dict(l=40, r=30, t=50, b=50)
)
fig.update_xaxes(gridcolor="#293b56")
fig.update_yaxes(gridcolor="#293b56", range=[0, 1])

st.plotly_chart(fig, use_container_width=True)

# ---------------- NRTL PARAMETERS ----------------
st.subheader("04 | NRTL τ and G Matrices")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**τij = bij / (RT)**")
    st.dataframe(pd.DataFrame(tau, index=names, columns=names).round(5), use_container_width=True)
with col_b:
    st.markdown("**Gij = exp(−αij τij)**")
    st.dataframe(pd.DataFrame(G, index=names, columns=names).round(5), use_container_width=True)

# ---------------- SUMMARY TABLE ----------------
st.subheader("05 | Summary Table")

summary = pd.DataFrame({
    "Part": ["(a) BUBL P", "(b) DEW P", "(c) P,T-flash"],
    "P (kPa)": [P_a, P_b, P_flash],
    "x1": [x_a[0], x_b[0], x_f[0]],
    "x2": [x_a[1], x_b[1], x_f[1]],
    "x3": [x_a[2], x_b[2], x_f[2]],
    "y1": [y_a[0], y_b[0], y_f[0]],
    "y2": [y_a[1], y_b[1], y_f[1]],
    "y3": [y_a[2], y_b[2], y_f[2]],
})
st.dataframe(summary.round(5), use_container_width=True)

csv = summary.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Results CSV",
    data=csv,
    file_name="Q12_21_Ternary_NRTL.csv",
    mime="text/csv"
)

# ---------------- FORMULAS ----------------
with st.expander("View equations and calculation methodology"):
    st.markdown("""
    **Antoine equation:** ln(Psat/kPa) = A − B/(T°C + C)

    **Multicomponent NRTL equation**

    τij = bij/(RT),  Gij = exp(−αij τij),  αij = αji

    ln γi = Σj τji Gji xj / Σk Gki xk
    + Σj [ xj Gij / Σk Gkj xk ] · ( τij − Σm xm τmj Gmj / Σk Gkj xk )

    **BUBL P:** P = Σ xi γi Pisat,  yi = xi γi Pisat / P

    **DEW P:** 1/P = Σ yi/(γi Pisat), with xi = yi P/(γi Pisat).
    γ depends on x, so x is iterated until it converges.

    **P,T-flash:** P = ½(P_bubble + P_dew) at z. Ki = γi Pisat / P.
    The Rachford-Rice equation is solved for V by bisection, then
    xi = zi/(1+V(Ki−1)) and yi = Ki xi, iterating until x converges.

    **Gas constant:** R = 1.987204 cal/(mol K)
    """)

st.divider()

st.markdown("""
<div style="text-align:center;color:#94a3b8;">
VLE Thermodynamics Laboratory | Assignment 3 | Question 12.21
<br>
NRTL Equation | Acetone – Methanol – Water
</div>
""", unsafe_allow_html=True)
