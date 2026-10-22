#!/usr/bin/env python3
"""
Step 1: classical (continuum) simulation of circular-polarization-dependent
transmission through TWO twisted anisotropic absorbing films.

Model
-----
* Each film is a uniform slab whose in-plane dielectric tensor is built from
  Lorentz oscillators (the amide I C=O stretch treated as a classical
  damped oscillator):

      eps(w) = eps_bg * I + sum_j  f_j * w_j^2 / (w_j^2 - w^2 - i*gamma_j*w) * P_j

  P_j = s * u_j u_j^T + (1 - s) * I/2   (u_j: transition-dipole direction,
  s: orientational order, s = 1 -> perfectly aligned).
  Default (antiparallel beta-sheet):
      1630 cm^-1, strong, dipole PERPENDICULAR to the strand axis
      1690 cm^-1, weak,   dipole PARALLEL      to the strand axis
  The strand ("polymer") axis of film 1 is along x.

* Film 2 is rotated about the beam axis (z) by the twist angle phi.
* Normal incidence, Berreman 4x4 transfer matrix, exp(-i w t) convention
  (Im eps > 0 = absorption), fully coherent (interfaces and interference
  included).

Sign conventions
----------------
* Light propagates along +z; film 1 is hit first.
* phi > 0: film 2 is rotated counterclockwise relative to film 1 when seen
  by an observer looking back toward the light source
  (= right-handed twist along the propagation direction).
* LCP = (x + i y)/sqrt(2), RCP = (x - i y)/sqrt(2)   [optics convention,
  exp(-i w t) time dependence].
* Delta A = A_L - A_R = log10(T_R / T_L)   (usual VCD sign convention).

Usage
-----
    python vcd_twist_sim.py                       # defaults, writes ./vcd_twist_out
    python vcd_twist_sim.py --thickness 2 --fperp 0.02 --probe 1620 1630 1640
    python vcd_twist_sim.py --help

Requires: numpy, scipy, matplotlib.
"""
import argparse
import csv
import os
from dataclasses import dataclass

import matplotlib
import numpy as np
from scipy.linalg import expm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402


# --------------------------------------------------------------------------
# Material model
# --------------------------------------------------------------------------
@dataclass
class Oscillator:
    nu0: float         # band center (cm^-1)
    gamma: float       # damping (cm^-1)
    f: float           # dimensionless oscillator strength
    angle_deg: float   # dipole direction in the film frame, measured from x (strand axis)


def default_oscillators(nu0=1630.0, gamma=20.0, f_perp=0.012, f_para_ratio=0.1):
    """Antiparallel beta-sheet amide I: strong perpendicular band + weak parallel band."""
    return [
        Oscillator(nu0, gamma, f_perp, 90.0),
        Oscillator(nu0 + 60.0, 0.75 * gamma, f_perp * f_para_ratio, 0.0),
    ]


def eps_inplane(nu, osc, eps_bg, phi_deg, order=1.0):
    """In-plane 2x2 dielectric tensor, shape nu.shape + (2, 2).

    nu: wavenumber(s) in cm^-1; phi_deg: rotation of the film about z.
    """
    nu = np.asarray(nu, float)
    eps = np.zeros(nu.shape + (2, 2), complex)
    eps[..., 0, 0] = eps_bg
    eps[..., 1, 1] = eps_bg
    for o in osc:
        chi = o.f * o.nu0**2 / (o.nu0**2 - nu**2 - 1j * o.gamma * nu)
        a = np.deg2rad(o.angle_deg + phi_deg)
        u = np.array([np.cos(a), np.sin(a)])
        proj = order * np.outer(u, u) + (1.0 - order) * 0.5 * np.eye(2)
        eps = eps + chi[..., None, None] * proj
    return eps


# --------------------------------------------------------------------------
# Berreman 4x4 transfer matrix at normal incidence
# psi = (Ex, Hy, Ey, -Hx)  (H in units of E),  d psi/dz = i k0 Delta psi
# --------------------------------------------------------------------------
def _expm_batch(a):
    flat = np.asarray(a, complex).reshape(-1, 4, 4)
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        out[i] = expm(flat[i])
    return out.reshape(np.shape(a))


def layer_matrix(eps2, nu, d_cm):
    """Propagation matrix of one layer. eps2: (..., 2, 2) with leading shape
    broadcast against nu (..., ); d_cm: thickness in cm."""
    k0 = 2.0 * np.pi * np.asarray(nu, float)  # 1/cm
    delta = np.zeros(eps2.shape[:-2] + (4, 4), complex)
    delta[..., 0, 1] = 1.0
    delta[..., 1, 0] = eps2[..., 0, 0]
    delta[..., 1, 2] = eps2[..., 0, 1]
    delta[..., 2, 3] = 1.0
    delta[..., 3, 0] = eps2[..., 1, 0]
    delta[..., 3, 2] = eps2[..., 1, 1]
    return _expm_batch(1j * (k0 * d_cm)[..., None, None] * delta)


def boundary_matrix(n):
    """psi = L @ (Ex_f, Ey_f, Ex_b, Ey_b) in an isotropic medium of index n."""
    return np.array(
        [[1, 0, 1, 0],
         [n, 0, -n, 0],
         [0, 1, 0, 1],
         [0, n, 0, -n]],
        dtype=complex,
    )


def stack_t_r(m_total, n_in, n_out):
    """Jones-type transmission t and reflection r matrices (..., 2, 2)."""
    a = np.linalg.inv(boundary_matrix(n_out)) @ m_total @ boundary_matrix(n_in)
    a11, a12 = a[..., :2, :2], a[..., :2, 2:]
    a21, a22 = a[..., 2:, :2], a[..., 2:, 2:]
    x = np.linalg.solve(a22, a21)
    return a11 - a12 @ x, -x


E_LCP = np.array([1.0, 1.0j]) / np.sqrt(2.0)
E_RCP = np.array([1.0, -1.0j]) / np.sqrt(2.0)


def circular_response(t, r, n_in, n_out):
    """Power transmittance / reflectance for LCP and RCP incidence."""
    fac = np.real(n_out) / np.real(n_in)
    out = {}
    for name, e in (("L", E_LCP), ("R", E_RCP)):
        out["T" + name] = fac * np.sum(np.abs(t @ e) ** 2, axis=-1)
        out["R" + name] = np.sum(np.abs(r @ e) ** 2, axis=-1)
    return out


# --------------------------------------------------------------------------
# Two-film stack
# --------------------------------------------------------------------------
def simulate(nu, phi_deg, osc, eps_bg=2.25, d_um=1.0, gap_um=0.0, n_gap=1.0,
             order=1.0, n_in=1.0, n_out=1.0):
    """Return dict of arrays with shape (len(phi), len(nu))."""
    nu = np.asarray(nu, float)
    phi = np.asarray(phi_deg, float)
    d_cm = d_um * 1e-4
    m1 = layer_matrix(eps_inplane(nu, osc, eps_bg, 0.0, order), nu, np.full(nu.shape, d_cm))
    eps2 = np.stack([eps_inplane(nu, osc, eps_bg, p, order) for p in phi])  # (nphi, nnu, 2, 2)
    m2 = layer_matrix(eps2, np.broadcast_to(nu, eps2.shape[:2]), np.full(eps2.shape[:2], d_cm))
    m = m2 @ m1[None]
    if gap_um > 0.0:
        eg = np.zeros(nu.shape + (2, 2), complex)
        eg[..., 0, 0] = eg[..., 1, 1] = n_gap**2
        mg = layer_matrix(eg, nu, np.full(nu.shape, gap_um * 1e-4))
        m = m2 @ mg[None] @ m1[None]
    t, r = stack_t_r(m, n_in, n_out)
    return circular_response(t, r, n_in, n_out)


def delta_A(res):
    """Delta A = A_L - A_R = log10(T_R / T_L)."""
    return np.log10(res["TR"] / res["TL"])


# --------------------------------------------------------------------------
# Self tests
# --------------------------------------------------------------------------
def selftest(args):
    print("Self tests")
    ok = True

    # 1) isotropic slab vs analytic Airy formula
    nu = np.linspace(1500, 1750, 11)
    osc = [Oscillator(1630.0, 20.0, 0.012, 0.0)]
    res = simulate(nu, [0.0], osc, args.nbg**2, args.thickness, order=0.0)
    eps = eps_inplane(nu, osc, args.nbg**2, 0.0, order=0.0)[:, 0, 0]
    n_c = np.sqrt(eps)
    delta = n_c * 2 * np.pi * nu * (2 * args.thickness * 1e-4)  # two films = one slab of 2d
    t12, t21, r21 = 2 / (1 + n_c), 2 * n_c / (1 + n_c), (n_c - 1) / (n_c + 1)
    t_an = np.abs(t12 * t21 * np.exp(1j * delta) / (1 - r21**2 * np.exp(2j * delta))) ** 2
    err = np.max(np.abs(res["TL"][0] - t_an))
    ok &= err < 1e-9
    print(f"  isotropic slab vs analytic Airy formula : max |dT| = {err:.1e}")

    # 2) lossless case: R + T = 1
    osc0 = [Oscillator(1630.0, 0.0, 0.012, 90.0), Oscillator(1690.0, 0.0, 0.002, 0.0)]
    res = simulate(np.array([1400.0, 1500.0]), [0.0, 30.0, 70.0], osc0, args.nbg**2, args.thickness)
    err = np.max(np.abs(res["TL"] + res["RL"] - 1)) + np.max(np.abs(res["TR"] + res["RR"] - 1))
    ok &= err < 1e-9
    print(f"  lossless stack: |R + T - 1|              = {err:.1e}")

    # 3) phi = 0: aligned films cannot show CD
    nu = np.linspace(1560, 1740, 61)
    osc = default_oscillators(args.nu0, args.gamma, args.fperp, args.fpara_ratio)
    kw = dict(eps_bg=args.nbg**2, d_um=args.thickness, gap_um=args.gap, order=args.order,
              n_in=args.nin, n_out=args.nout)
    res = simulate(nu, [0.0, 90.0, 180.0], osc, **kw)
    err = np.max(np.abs(res["TL"] - res["TR"]))
    ok &= err < 1e-12
    print(f"  phi = 0, 90, 180 deg: max |T_L - T_R|    = {err:.1e}")

    # 4) mirror symmetry: T_L(phi) = T_R(-phi) ; 180 deg periodicity
    p = np.array([17.0, 40.0, 63.0])
    a = simulate(nu, p, osc, **kw)
    b = simulate(nu, -p, osc, **kw)
    c = simulate(nu, p + 180.0, osc, **kw)
    err1 = np.max(np.abs(a["TL"] - b["TR"]))
    err2 = np.max(np.abs(a["TL"] - c["TL"]))
    ok &= err1 < 1e-10 and err2 < 1e-10
    print(f"  T_L(phi) = T_R(-phi)                     : {err1:.1e}")
    print(f"  180 deg periodicity                       : {err2:.1e}")
    print("  ->", "ALL PASSED" if ok else "SOMETHING FAILED")
    return ok


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated all-pairs)
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def make_plots(nu, phi, res, dA, probes, osc, outdir):
    plt.rcParams.update({"font.size": 10, "figure.facecolor": SURFACE})
    scale = 1e3

    # ---- Figure 1: map + spectra -------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    m = np.max(np.abs(dA)) * scale
    pc = a1.pcolormesh(nu, phi, dA * scale, cmap=DIVERGING, vmin=-m, vmax=m, shading="auto")
    cb = fig.colorbar(pc, ax=a1)
    cb.set_label(r"$\Delta A = A_L - A_R$  ($\times10^{-3}$)", color=INK)
    cb.ax.tick_params(colors=INK2, labelsize=9)
    a1.set_xlabel("Wavenumber (cm$^{-1}$)")
    a1.set_ylabel("Twist angle $\\varphi$ (deg)")
    a1.set_yticks(np.arange(-90, 91, 30))
    a1.set_title("(a) Apparent CD map of two twisted films", loc="left", fontsize=11)
    style(a1)
    a1.grid(False)

    absorb = -np.log10(0.5 * (res["TL"] + res["TR"]))
    a2.set_title("(b) CD spectra and absorbance", loc="left", fontsize=11)
    shown = [(30.0, SERIES[0], "-", r"$\varphi=+30^\circ$"),
             (-30.0, SERIES[1], "--", r"$\varphi=-30^\circ$"),
             (60.0, SERIES[2], "-.", r"$\varphi=+60^\circ$")]
    for ph, col, ls, lab in shown:
        i = int(np.argmin(np.abs(phi - ph)))
        a2.plot(nu, dA[i] * scale, color=col, ls=ls, lw=2, label=lab)
    a2.axhline(0, color=INK2, lw=0.8)
    a2.set_xlabel("Wavenumber (cm$^{-1}$)")
    a2.set_ylabel(r"$\Delta A$  ($\times10^{-3}$)")
    style(a2)
    a2b = a2.twiny()  # no second y-scale: absorbance drawn as scaled grey backdrop
    a2b.set_visible(False)
    i0 = int(np.argmin(np.abs(phi - 30.0)))
    ymax = np.max(np.abs(dA[[int(np.argmin(np.abs(phi - p))) for p in (30, -30, 60)]])) * scale
    prof = absorb[i0] / np.max(absorb[i0]) * ymax * 0.9
    a2.fill_between(nu, 0, prof, color="#c3c2b7", alpha=0.35, lw=0, zorder=0)
    a2.text(nu[np.argmax(absorb[i0])] + 6, ymax * 0.82, "absorbance\n(shape only)",
            color=INK2, fontsize=8.5, va="top")
    a2.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    fig.savefig(os.path.join(outdir, "vcd_twist_map.png"), dpi=170)
    plt.close(fig)

    # ---- Figure 2: ratio vs twist angle ---------------------------------
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    for k, nu_p in enumerate(probes):
        j = int(np.argmin(np.abs(nu - nu_p)))
        ratio = res["TL"][:, j] / res["TR"][:, j]
        col, ls = SERIES[k % 3], ["-", "--", "-."][k % 3]
        lab = f"{nu[j]:.0f} cm$^{{-1}}$"
        b1.plot(phi, ratio, color=col, ls=ls, lw=2, label=lab)
        b2.plot(phi, dA[:, j] * scale, color=col, ls=ls, lw=2, label=lab)
    b1.axhline(1.0, color=INK2, lw=0.8)
    b2.axhline(0.0, color=INK2, lw=0.8)
    b1.set_ylabel(r"$T_L\,/\,T_R$")
    b2.set_ylabel(r"$\Delta A = A_L - A_R$  ($\times10^{-3}$)")
    b1.set_title("(a) Intensity ratio of transmitted LCP / RCP", loc="left", fontsize=11)
    b2.set_title("(b) Same data as CD signal", loc="left", fontsize=11)
    for ax in (b1, b2):
        ax.set_xlabel("Twist angle between films $\\varphi$ (deg)")
        ax.set_xticks(np.arange(-90, 91, 30))
        style(ax)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper center",
                  bbox_to_anchor=(0.5, -0.2), ncol=len(probes))
    fig.savefig(os.path.join(outdir, "vcd_twist_phi.png"), dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thickness", type=float, default=1.0, help="thickness of EACH film (um)")
    ap.add_argument("--gap", type=float, default=0.0, help="isotropic spacer between films (um)")
    ap.add_argument("--nbg", type=float, default=1.5, help="background refractive index of the films")
    ap.add_argument("--nin", type=float, default=1.0, help="index of incident medium")
    ap.add_argument("--nout", type=float, default=1.0, help="index of exit medium")
    ap.add_argument("--nu0", type=float, default=1630.0, help="main amide I band (cm^-1)")
    ap.add_argument("--gamma", type=float, default=20.0, help="main band damping (cm^-1)")
    ap.add_argument("--fperp", type=float, default=0.012, help="oscillator strength, main band")
    ap.add_argument("--fpara-ratio", type=float, default=0.1, help="weak 1690 band strength / main band")
    ap.add_argument("--order", type=float, default=1.0, help="orientational order 0..1")
    ap.add_argument("--numin", type=float, default=1560.0)
    ap.add_argument("--numax", type=float, default=1740.0)
    ap.add_argument("--probe", type=float, nargs="+", default=[1620.0, 1630.0, 1640.0])
    ap.add_argument("--outdir", default="vcd_twist_out")
    args = ap.parse_args()

    selftest(args)
    os.makedirs(args.outdir, exist_ok=True)

    nu = np.arange(args.numin, args.numax + 0.01, 0.5)
    phi = np.arange(-90.0, 90.01, 2.0)
    osc = default_oscillators(args.nu0, args.gamma, args.fperp, args.fpara_ratio)
    res = simulate(nu, phi, osc, args.nbg**2, args.thickness, args.gap, 1.0,
                   args.order, args.nin, args.nout)
    dA = delta_A(res)

    # CSV of the phi scan at the probe wavenumbers
    with open(os.path.join(args.outdir, "phi_scan.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["phi_deg", "nu_cm-1", "T_L", "T_R", "T_L/T_R", "deltaA(A_L-A_R)"])
        for p in args.probe:
            j = int(np.argmin(np.abs(nu - p)))
            for i, ph in enumerate(phi):
                w.writerow([ph, nu[j], res["TL"][i, j], res["TR"][i, j],
                            res["TL"][i, j] / res["TR"][i, j], dA[i, j]])

    make_plots(nu, phi, res, dA, args.probe, osc, args.outdir)

    # short numerical summary
    i30 = int(np.argmin(np.abs(phi - 30)))
    absorb = -np.log10(0.5 * (res["TL"] + res["TR"]))
    jpk = int(np.argmax(absorb[i30]))
    print(f"\nPeak absorbance (mean of L/R, phi=30 deg): {absorb[i30, jpk]:.3f} at {nu[jpk]:.1f} cm^-1")
    print(f"max |Delta A| over the map              : {np.max(np.abs(dA)):.2e}")
    jm = np.unravel_index(np.argmax(np.abs(dA)), dA.shape)
    print(f"  reached at phi = {phi[jm[0]]:.0f} deg, nu = {nu[jm[1]]:.1f} cm^-1")
    print(f"Outputs written to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
