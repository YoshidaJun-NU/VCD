import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from vcd_twist_sim import default_oscillators, delta_A, simulate

st.set_page_config(page_title="ねじれた2枚膜のCDシミュレーション", layout="wide")
st.title("ねじれた2枚のβシート膜：アミドI領域の円偏光透過差")
st.caption(
    "ローレンツ振動子 + Berreman 4×4伝達行列法（古典電磁気学、垂直入射）。"
    "ΔA = A_L − A_R = log10(T_R / T_L)。構造由来（見かけ）のCDであり、分子由来のVCDは含まない。"
)

with st.sidebar:
    st.header("パラメータ")
    thickness = st.slider("各膜の厚さ (μm)", 0.1, 5.0, 1.0, 0.1)
    gap = st.slider("膜間のすき間 (μm)", 0.0, 2.0, 0.0, 0.05)
    order = st.slider("配向度 s", 0.0, 1.0, 1.0, 0.05)
    nbg = st.slider("背景屈折率", 1.2, 2.0, 1.5, 0.05)
    st.subheader("アミドI バンド")
    nu0 = st.slider("主バンド中心 (cm⁻¹)", 1610, 1650, 1630, 1)
    gamma = st.slider("幅 γ (cm⁻¹)", 8, 40, 20, 1)
    fperp = st.slider("振動子強度 f（主バンド）", 0.001, 0.05, 0.012, 0.001, format="%.3f")
    ratio = st.slider("副バンド（+60 cm⁻¹）強度比", 0.0, 0.5, 0.1, 0.01)
    st.subheader("表示")
    probe = st.slider("ΔA vs φ を見る波数 (cm⁻¹)", 1580, 1720, 1640, 1)

nu = np.arange(1560.0, 1740.01, 1.0)
phi = np.arange(-90.0, 90.01, 2.0)


@st.cache_data
def run(thickness, gap, order, nbg, nu0, gamma, fperp, ratio):
    osc = default_oscillators(nu0, gamma, fperp, ratio)
    res = simulate(nu, phi, osc, nbg**2, thickness, gap, 1.0, order, 1.0, 1.0)
    return res["TL"], res["TR"], delta_A(res)


TL, TR, dA = run(thickness, gap, order, nbg, nu0, gamma, fperp, ratio)
j = int(np.argmin(np.abs(nu - probe)))

c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    m = max(np.max(np.abs(dA)), 1e-12) * 1e3
    pc = ax.pcolormesh(nu, phi, dA * 1e3, cmap="RdBu_r", vmin=-m, vmax=m, shading="auto")
    fig.colorbar(pc, ax=ax, label="ΔA = A_L − A_R (×10⁻³)")
    ax.axvline(nu[j], color="k", lw=0.8, ls="--")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Twist angle φ (deg)")
    ax.set_title("ΔA map")
    st.pyplot(fig)
with c2:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(phi, TL[:, j] / TR[:, j], lw=2)
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xticks(np.arange(-90, 91, 30))
    ax.set_xlabel("Twist angle φ (deg)")
    ax.set_ylabel("T_L / T_R")
    ax.set_title(f"Transmitted LCP/RCP ratio at {nu[j]:.0f} cm$^{{-1}}$")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

phi_sel = st.slider("スペクトルを見るφ (deg)", -90, 90, 30, 2)
i = int(np.argmin(np.abs(phi - phi_sel)))
fig, ax = plt.subplots(figsize=(12, 3.6))
ax.plot(nu, dA[i] * 1e3, lw=2, label=f"φ = {phi[i]:+.0f}°")
ax.plot(nu, dA[len(phi) - 1 - i] * 1e3, lw=2, ls="--", label=f"φ = {phi[len(phi)-1-i]:+.0f}°")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("ΔA (×10⁻³)")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
st.pyplot(fig)

st.caption("φ = 0°, ±90° でCD = 0、φ → −φ で符号反転、周期180°。")
