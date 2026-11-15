import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import math

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Q12.22 | VLE Dashboard",
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
st.caption("ASSIGNMENT 3  /  QUESTION 12.22  /  WILSON MODEL  /  TERNARY")

st.markdown("""
BUBL T, DEW T and P,T-flash calculations for the
**acetone(1) / methanol(2) / water(3)** system at constant pressure
using the multicomponent Wilson equation.
""")

st.divider()

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.title("⚙️ Calculation Settings")

st.sidebar.subheader("Operating Conditions")
P = st.sidebar.number_input(
    "Pressure (kPa)", min_value=10.0, max_value=500.0, value=101.33, step=1.0
)

st.sidebar.markdown("**(a) BUBL T liquid composition**")
xa1 = st.sidebar.number_input("x1 (acetone)", 0.001, 0.998, 0.3, 0.05, key="xa1")
xa2 = st.sidebar.number_input("x2 (methanol)", 0.001, 0.998, 0.4, 0.05, key="xa2")

st.sidebar.markdown("**(b) DEW T vapor composition**")
yb1 = st.sidebar.number_input("y1 (acetone)", 0.001, 0.998, 0.3, 0.05, key="yb1")
yb2 = st.sidebar.number_input("y2 (methanol)", 0.001, 0.998, 0.4, 0.05, key="yb2")

st.sidebar.markdown("**(c) Flash feed composition**")
zc1 = st.sidebar.number_input("z1 (acetone)", 0.001, 0.998, 0.3, 0.05, key="zc1")
zc2 = st.sidebar.number_input("z2 (methanol)", 0.001, 0.998, 0.2, 0.05, key="zc2")

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

st.sidebar.subheader("Molar Volumes (cm³/mol)")
V1 = st.sidebar.number_input("V1 (acetone)", value=74.05)
V2 = st.sidebar.number_input("V2 (methanol)", value=40.73)
V3 = st.sidebar.number_input("V3 (water)", value=18.07)

st.sidebar.subheader("Wilson a_ij (cal/mol)")
a12 = st.sidebar.number_input("a12 (acetone/methanol)", value=-161.88)
a21 = st.sidebar.number_input("a21 (methanol/acetone)", value=583.11)
a13 = st.sidebar.number_input("a13 (acetone/water)", value=291.27)
a31 = st.sidebar.number_input("a31 (water/acetone)", value=1448.01)
a23 = st.sidebar.number_input("a23 (methanol/water)", value=107.38)
a32 = st.sidebar.number_input("a32 (water/methanol)", value=469.55)

# ---------------- CONSTANTS & DATA ----------------
R = 1.987204  # cal/(mol K)
names = ["Acetone", "Methanol", "Water"]
Vm = np.array([V1, V2, V3])
a = np.array([
    [0.0, a12, a13],
    [a21, 0.0, a23],
    [a31, a32, 0.0]
])
ANT = [(A1, B1, C1), (A2, B2, C2), (A3, B3, C3)]

# ---------------- EQUATIONS ----------------
def psat_vec(T_C):
    return np.array([math.exp(A - B / (T_C + C)) for A, B, C in ANT])

def wilson(x, T):
    Lam = np.ones((3, 3))
    for i in range(3):
        for j in range(3):
            if i != j:
                Lam[i, j] = (Vm[j] / Vm[i]) * math.exp(-a[i, j] / (R * T))
    S = Lam @ x
    ln_g = 1.0 - np.log(S) - (Lam.T @ (x / S))
    return np.exp(ln_g), Lam

def bisect(f, low, high, n=100):
    for _ in range(n):
        mid = 0.5 * (low + high)
        if f(low) * f(mid) <= 0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)

# ---------------- (a) BUBL T ----------------
def bubl_t(x):
    def f(T_C):
        g, _ = wilson(x, T_C + 273.15)
        return np.sum(x * g * psat_vec(T_C)) - P

    T_C = bisect(f, 0.0, 200.0)
    g, _ = wilson(x, T_C + 273.15)
    y = x * g * psat_vec(T_C) / P
    return T_C, y, g

# ---------------- (b) DEW T (nested iteration) ----------------
def dew_t(y):
    x = y.copy()
    T_C = 50.0
    for _ in range(500):
        def f(Tc):
            g, _ = wilson(x, Tc + 273.15)
            return np.sum(y * P / (g * psat_vec(Tc))) - 1.0

        T_C = bisect(f, 0.0, 200.0)
        g, _ = wilson(x, T_C + 273.15)
        x_new = y * P / (g * psat_vec(T_C))
        x_new = x_new / np.sum(x_new)
        if np.max(np.abs(x_new - x)) < 1e-12:
            x = x_new
            break
        x = x_new
    g, _ = wilson(x, T_C + 273.15)
    return T_C, x, g

# ---------------- (c) P,T-FLASH ----------------
def flash(z, T_C):
    T = T_C + 273.15
    Ps = psat_vec(T_C)
    x = z.copy()
    V_frac = 0.5
    for _ in range(2000):
        g, _ = wilson(x, T)
        K = g * Ps / P

        def rr(v):
            return np.sum(z * (K - 1) / (1 + v * (K - 1)))

        if rr(0.0) <= 0:
            V_frac = 0.0
        elif rr(1.0) >= 0:
            V_frac = 1.0
        else:
            V_frac = bisect(rr, 0.0, 1.0)

        x_new = z / (1 + V_frac * (K - 1))
        x_new = x_new / np.sum(x_new)
        if np.max(np.abs(x_new - x)) < 1e-12:
            x = x_new
            break
        x = x_new
    g, _ = wilson(x, T)
    K = g * Ps / P
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
        st.error(f"{label}: composition of water (3) is ≤ 0. Make sure the first two add to less than 1.")
        valid = False
if not valid:
    st.stop()

# ---------------- RUN CALCULATIONS ----------------
T_a, y_a, g_a = bubl_t(x_a)
T_b, x_b, g_b = dew_t(y_b)

T_bub_z = bubl_t(z_c)[0]
T_dew_z = dew_t(z_c)[0]
T_flash = 0.5 * (T_bub_z + T_dew_z)
V_f, x_f, y_f = flash(z_c, T_flash)

# ---------------- METRICS ----------------
st.subheader("01 | System Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Pressure", f"{P:.2f} kPa")
c2.metric("Feed z (flash)", f"({z_c[0]:.2f}, {z_c[1]:.2f}, {z_c[2]:.2f})")
c3.metric("Components", "Acetone / Methanol / Water")

st.divider()

# ---------------- RESULTS ----------------
st.subheader("02 | Results")

st.markdown(f"**(a) BUBL T** at P = {P:.2f} kPa, x = ({x_a[0]:.3f}, {x_a[1]:.3f}, {x_a[2]:.3f})")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Bubble T", f"{T_a + 273.15:.3f} K ({T_a:.2f} °C)")
m2.metric("y1 (acetone)", f"{y_a[0]:.5f}")
m3.metric("y2 (methanol)", f"{y_a[1]:.5f}")
m4.metric("y3 (water)", f"{y_a[2]:.5f}")

st.markdown(f"**(b) DEW T** at P = {P:.2f} kPa, y = ({y_b[0]:.3f}, {y_b[1]:.3f}, {y_b[2]:.3f})")
n1, n2, n3, n4 = st.columns(4)
n1.metric("Dew T", f"{T_b + 273.15:.3f} K ({T_b:.2f} °C)")
n2.metric("x1 (acetone)", f"{x_b[0]:.5f}")
n3.metric("x2 (methanol)", f"{x_b[1]:.5f}")
n4.metric("x3 (water)", f"{x_b[2]:.5f}")

st.markdown(
    f"**(c) P,T-flash** at P = {P:.2f} kPa, z = ({z_c[0]:.3f}, {z_c[1]:.3f}, {z_c[2]:.3f}), "
    f"T = ½(T_bubble + T_dew)"
)
f1, f2, f3 = st.columns(3)
f1.metric("Flash T", f"{T_flash + 273.15:.3f} K ({T_flash:.2f} °C)")
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

# ---------------- WILSON LAMBDA ----------------
st.subheader("04 | Wilson Λ Matrix (at flash temperature)")

_, Lam_flash = wilson(x_f, T_flash + 273.15)
st.dataframe(pd.DataFrame(Lam_flash, index=names, columns=names).round(5),
             use_container_width=True)

# ---------------- SUMMARY TABLE ----------------
st.subheader("05 | Summary Table")

summary = pd.DataFrame({
    "Part": ["(a) BUBL T", "(b) DEW T", "(c) P,T-flash"],
    "T (K)": [T_a + 273.15, T_b + 273.15, T_flash + 273.15],
    "T (°C)": [T_a, T_b, T_flash],
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
    file_name="Q12_22_Ternary_Wilson.csv",
    mime="text/csv"
)

# ---------------- FORMULAS ----------------
with st.expander("View equations and calculation methodology"):
    st.markdown("""
    **Antoine equation:** ln(Psat/kPa) = A − B/(T°C + C)

    **Multicomponent Wilson equation**

    Λij = (Vj/Vi) exp(−aij/RT),  Λii = 1

    ln γi = 1 − ln(Σj xj Λij) − Σk [ xk Λki / Σj xj Λkj ]

    **BUBL T:** find T such that Σ xi γi Pisat(T) = P (bisection; Λ and γ
    recomputed at every trial T). Then yi = xi γi Pisat / P.

    **DEW T:** find T such that Σ yi P/(γi Pisat) = 1, with
    xi = yi P/(γi Pisat). γ depends on x and T, so x is updated and the
    T-search repeated until x converges.

    **P,T-flash:** T = ½(T_bubble + T_dew) at z. Ki = γi Pisat / P.
    The Rachford-Rice equation Σ zi(Ki−1)/(1+V(Ki−1)) = 0 is solved for V
    by bisection, then xi = zi/(1+V(Ki−1)) and yi = Ki xi, iterating until
    x converges.

    **Gas constant:** R = 1.987204 cal/(mol K)
    """)

st.divider()

st.markdown("""
<div style="text-align:center;color:#94a3b8;">
VLE Thermodynamics Laboratory | Assignment 3 | Question 12.22
<br>
Wilson Equation | Acetone – Methanol – Water
</div>
""", unsafe_allow_html=True)
