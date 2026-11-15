import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import math

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Q12.14 | VLE Dashboard",
    page_icon="⚗️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: #0b1220;
    color: #e5e7eb;
}
.block-container {
    padding-top: 2rem;
    max-width: 1400px;
}
h1, h2, h3 {
    color: #e2e8f0 !important;
}
div[data-testid="stMetric"] {
    background: #172338;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #293b56;
}
section[data-testid="stSidebar"] {
    background: #111c30;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("⚗️ VLE Thermodynamics Laboratory")
st.caption("ASSIGNMENT 3  /  QUESTION 12.14  /  NRTL MODEL")

st.markdown("""
Calculate and visualize the pressure-composition (P-x-y) diagram
for a binary liquid mixture using the NRTL activity coefficient model.
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

st.sidebar.subheader("Graph Resolution")
n_points = st.sidebar.slider(
    "Number of calculation points", min_value=21, max_value=201, value=101, step=10
)

# ---------------- CONSTANTS ----------------
R = 1.987204  # cal/(mol K)

# ---------------- ANTOINE EQUATION ----------------
def antoine(A, B, C, temp_C):
    return math.exp(A - B / (temp_C + C))

# ---------------- NRTL EQUATION ----------------
def nrtl(x1, T):
    x2 = 1.0 - x1

    tau12 = b12 / (R * T)
    tau21 = b21 / (R * T)
    G12 = math.exp(-alpha * tau12)
    G21 = math.exp(-alpha * tau21)

    ln_gamma1 = x2**2 * (
        tau21 * (G21 / (x1 + x2 * G21))**2
        + tau12 * G12 / (x2 + x1 * G12)**2
    )
    ln_gamma2 = x1**2 * (
        tau12 * (G12 / (x2 + x1 * G12))**2
        + tau21 * G21 / (x1 + x2 * G21)**2
    )

    return math.exp(ln_gamma1), math.exp(ln_gamma2)

# ---------------- VAPOR PRESSURES ----------------
P1_sat = antoine(A1, B1, C1, T_C)
P2_sat = antoine(A2, B2, C2, T_C)

# ---------------- CALCULATIONS ----------------
x_values = np.linspace(0.0001, 0.9999, n_points)

bubble_pressures = []
vapor_compositions = []
gamma1_values = []
gamma2_values = []

for x1 in x_values:
    x2 = 1 - x1
    gamma1, gamma2 = nrtl(x1, T)

    P_bubble = x1 * gamma1 * P1_sat + x2 * gamma2 * P2_sat
    y1 = x1 * gamma1 * P1_sat / P_bubble

    bubble_pressures.append(P_bubble)
    vapor_compositions.append(y1)
    gamma1_values.append(gamma1)
    gamma2_values.append(gamma2)

# ---------------- DATAFRAME ----------------
df = pd.DataFrame({
    "Liquid mole fraction x1": x_values,
    "Vapor mole fraction y1": vapor_compositions,
    "Bubble pressure (kPa)": bubble_pressures,
    "Gamma 1": gamma1_values,
    "Gamma 2": gamma2_values
})

# ---------------- METRICS ----------------
st.subheader("01 | System Overview")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Temperature", f"{T:.2f} K")
with c2:
    st.metric("Acetone Psat", f"{P1_sat:.3f} kPa")
with c3:
    st.metric("Water Psat", f"{P2_sat:.3f} kPa")
with c4:
    st.metric("Calculation points", n_points)

st.divider()

# ---------------- P-X-Y GRAPH ----------------
st.subheader("02 | Pressure – Composition Diagram")

plot_df = pd.concat([
    pd.DataFrame({
        "Mole fraction of acetone": x_values,
        "Pressure (kPa)": bubble_pressures,
        "Curve": "Bubble Point (P-x1)"
    }),
    pd.DataFrame({
        "Mole fraction of acetone": vapor_compositions,
        "Pressure (kPa)": bubble_pressures,
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

st.caption("Blue: Bubble pressure curve | Orange: Dew pressure curve")

st.divider()

# ---------------- NRTL PARAMETERS ----------------
st.subheader("03 | NRTL Parameters")

tau12 = b12 / (R * T)
tau21 = b21 / (R * T)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("τ12", f"{tau12:.5f}")
with col2:
    st.metric("τ21", f"{tau21:.5f}")
with col3:
    st.metric("G12", f"{math.exp(-alpha * tau12):.5f}")
with col4:
    st.metric("G21", f"{math.exp(-alpha * tau21):.5f}")

# ---------------- RESULTS TABLE ----------------
st.subheader("04 | Calculation Results")

st.dataframe(df.round(5), use_container_width=True, height=350)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Results CSV",
    data=csv,
    file_name="Q12_14_Acetone_Water_NRTL.csv",
    mime="text/csv"
)

# ---------------- FORMULAS ----------------
with st.expander("View equations and calculation methodology"):
    st.markdown("""
    **Antoine equation**

    ln(Psat/kPa) = A - B/(T°C + C)

    **Modified Raoult's law**

    P = x1 γ1 P1sat + x2 γ2 P2sat

    **Vapor composition**

    y1 = x1 γ1 P1sat / P

    **NRTL equation**

    τ12 = b12/(RT),  τ21 = b21/(RT)

    G12 = exp(-α τ12),  G21 = exp(-α τ21)

    ln γ1 = x2² [ τ21 (G21/(x1 + x2 G21))² + τ12 G12/(x2 + x1 G12)² ]

    ln γ2 = x1² [ τ12 (G12/(x2 + x1 G12))² + τ21 G21/(x1 + x2 G21)² ]

    **Gas constant**

    R = 1.987204 cal/(mol K)
    """)

st.divider()

st.markdown("""
<div style="text-align:center;color:#94a3b8;">
VLE Thermodynamics Laboratory | Assignment 3 | Question 12.14
<br>
NRTL Equation | Acetone – Water
</div>
""", unsafe_allow_html=True)
