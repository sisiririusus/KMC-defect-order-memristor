# -*- coding: utf-8 -*-
"""
Phase-1 notebook module derived from KMC_2D_5.1.ipynb.
目标：固定总缺陷数，扫描初始 m0（由 A/B 位点占据定义），构造
m -> sigma_E, l_c -> barrier map -> KMC -> morphology/statistics
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit
from joblib import Parallel, delayed
from matplotlib.ticker import ScalarFormatter

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

# ==================================================
# 0) Phase-1 核心可调参数（先改这里）
# ==================================================
FAST_MODE = True
SAVE_DIR = Path("phase1_outputs")
SAVE_DIR.mkdir(exist_ok=True)

# geometry
WIDTH_CELLS = 24
THICKNESS_CELLS = 36
TOX_NM = 100.0
XSPAN_NM = 70.0

Z_BOT = 0
Z_TOP = THICKNESS_CELLS - 1
Z_OX_START = 1
Z_OX_END = THICKNESS_CELLS - 2
CELL_Z_NM = TOX_NM / (THICKNESS_CELLS - 2)
CELL_X_NM = XSPAN_NM / (WIDTH_CELLS - 1)

# constants
KB = 8.617e-5
NU0 = 1e12

# states
S_EMPTY = np.int8(0)
S_ION   = np.int8(1)
S_FIL   = np.int8(2)
S_EBOT  = np.int8(3)
S_ETOP  = np.int8(4)

# --- KMC kinetic parameters (保留 v5.1 主体) ---
BASE_BARRIER_EV = 0.85
MIN_BARRIER_EV  = 0.08
PHIB_BOTTOM = 0.70
PHIB_TOP    = 0.90
E_INJECT0_EV = 0.35
ALPHA_FIELD = 0.22
FIELD_ENHANCE_MIG  = 10.0
FIELD_ENHANCE_GROW = 120.0
ENABLE_TIP_FOCUS  = True
TIP_FOCUS_POWER   = 1.8
TIP_FOCUS_MAX_VNM = 10.0
E_GROW0_EV   = 1.12
GAMMA_GROW   = 0.020
DELTA_ATTACH = 0.24
HOP_SLOW_NEAR_TIP = 0.05
TIP_BEHIND = 0
TIP_AHEAD  = 5
ENABLE_RESET = False
E_DISS0_EV   = 1.35
GAMMA_DISS   = 0.015
FOCUS_KAPPA = 0.05
E_LATERAL_NOISE_VNM = 0.01
GROW_PREF_FORWARD = 2.0
GROW_PENALTY_LATERAL = 0.35
NUCLEATION_MODE = "centered_random_multi"
NUC_SEEDS = 3
ENABLE_POST_THICKEN = True
POST_CONNECT_STEPS = 3000 if FAST_MODE else 4000
ENABLE_CENTER_BIAS = True
CENTER_BIAS_STRENGTH = 0.30
CENTER_BIAS_SIGMA_CELLS = 4.0

# KMC controls
N_INIT_IONS = 140
INIT_ION_Z_MAX_FRAC = 0.35
MAX_STEPS_SET = 40000 if FAST_MODE else 120000
MAX_STEPS_RESET = 60000 if FAST_MODE else 100000
STOP_IF_CONNECTED_SET = True
STOP_IF_DISCONNECTED_RESET = True
RATE_SAMPLE_MAX = 1200
RATE_SAMPLE_STRIDE = 7
MASK_UPDATE_STRIDE = 5
DRIFT_THRESH_NM = 8.0

# --- Phase-1 scan controls ---
PHASE1_E0 = 0.080
PHASE1_T0 = 700.0
M0_TARGET_LIST = [0.0, 0.25, 0.50, 0.75, 0.90]
N_STRUCTURE_SEEDS = 3 if FAST_MODE else 8
N_MC_PER_STRUCTURE = 2 if FAST_MODE else 4
N_JOBS = 1 if FAST_MODE else 8

# --- A/B mask & initial occupancy ---
AB_MODE = "stripe_z"   # "stripe_z" | "checkerboard" | "stripe_x"
ACTIVE_Z_MAX_FRAC = INIT_ION_Z_MAX_FRAC  # m 的统计区域，默认跟初始离子分布窗口一致
FIX_TOTAL_ION_COUNT = True

# --- m -> disorder mapping ---
SIGMA_RES_EV = 0.00
SIGMA_DIS_EV = 0.18
P_SIGMA = 1.0
LC0_CELLS = 0.0
LC_BETA_CELLS = 6.0
Q_LC = 1.0

# --- sparse-DFT depth profile (你后面最需要替换的部分) ---
# 方法1：直接给稀疏点 (z_nm, mu_ev)，程序自动插值到每一层
Z_DFT_NM = np.array([0.0, 12.0, 30.0, 55.0, 80.0, 100.0])
MU_DFT_Z_EV = np.array([0.74, 0.70, 0.77, 0.83, 0.88, 0.90])
# x 方向默认比 z 方向高一点点，表示 vertical 更易迁移；没有数据时可先这么用
MU_X_EXTRA_EV = 0.05

# ==================================================
# 0.1) depth-dependent mean barrier profile helpers
# ==================================================
def build_mu_layer_from_sparse_points(z_nm_points, mu_ev_points, thickness_cells=THICKNESS_CELLS, tox_nm=TOX_NM):
    z_nm_points = np.asarray(z_nm_points, dtype=float)
    mu_ev_points = np.asarray(mu_ev_points, dtype=float)
    assert z_nm_points.ndim == 1 and mu_ev_points.ndim == 1
    assert len(z_nm_points) == len(mu_ev_points) and len(z_nm_points) >= 2
    assert np.all(np.diff(z_nm_points) >= 0), "z_nm_points 必须递增"

    z_layer = np.zeros(thickness_cells, dtype=float)
    z_layer[0] = 0.0
    z_layer[-1] = tox_nm
    if thickness_cells > 2:
        z_layer[1:-1] = np.linspace(0.0, tox_nm, thickness_cells - 2)

    mu_layer = np.interp(z_layer, z_nm_points, mu_ev_points)
    mu_layer[0] = mu_layer[1]
    mu_layer[-1] = mu_layer[-2]
    return z_layer, mu_layer

Z_LAYER_NM, MU_Z_LAYER = build_mu_layer_from_sparse_points(Z_DFT_NM, MU_DFT_Z_EV)
MU_X_LAYER = MU_Z_LAYER + MU_X_EXTRA_EV


def plot_mu_profile(mu_x_layer=MU_X_LAYER, mu_z_layer=MU_Z_LAYER, z_layer_nm=Z_LAYER_NM):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(z_layer_nm, mu_z_layer, marker='o', label='mu_z(z)')
    ax.plot(z_layer_nm, mu_x_layer, marker='s', label='mu_x(z)')
    ax.set_xlabel('Depth z (nm)')
    ax.set_ylabel('Mean barrier (eV)')
    ax.set_title('Depth-dependent mean barrier profile')
    ax.legend()
    _no_offset(ax)
    plt.tight_layout()
    plt.show()

# =========================
# 1) utilities / plotting
# =========================
def _no_offset(ax):
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

@njit(cache=True)
def _clamp(Eb, Eb_min):
    return Eb_min if Eb < Eb_min else Eb

@njit(cache=True)
def _tip_field(E_base, gap_cells, power, emax, enable):
    if not enable:
        return E_base
    g = gap_cells
    if g < 1:
        g = 1
    val = E_base * (1.0 + 1.0 / g) ** power
    return emax if val > emax else val

@njit(cache=True)
def _center_bias_factor(x, w, enable, strength, sigma_cells):
    if not enable:
        return 1.0
    xc = (w - 1) / 2.0
    dx = x - xc
    s2 = sigma_cells * sigma_cells
    if s2 < 1e-9:
        return 1.0
    return 1.0 + strength * np.exp(-0.5 * dx * dx / s2)


# =========================
# 2) topology (connected mask / tip)
# =========================
@njit(cache=True)
def _bfs_bottom_connected_mask(occ):
    w, h = occ.shape
    mask = np.zeros((w, h), dtype=np.uint8)
    qx = np.empty(w*h, dtype=np.int32)
    qz = np.empty(w*h, dtype=np.int32)
    head = 0
    tail = 0
    # bottom electrode line as sources
    for x in range(w):
        mask[x, 0] = 1
        qx[tail] = x; qz[tail] = 0
        tail += 1
    # BFS
    while head < tail:
        x = qx[head]; z = qz[head]; head += 1
        # 4-neighbor expansion only into FIL or electrodes
        if x > 0 and mask[x-1, z] == 0:
            s = occ[x-1, z]
            if s == S_FIL or s == S_EBOT or s == S_ETOP:
                mask[x-1, z] = 1; qx[tail] = x-1; qz[tail] = z; tail += 1
        if x < w-1 and mask[x+1, z] == 0:
            s = occ[x+1, z]
            if s == S_FIL or s == S_EBOT or s == S_ETOP:
                mask[x+1, z] = 1; qx[tail] = x+1; qz[tail] = z; tail += 1
        if z > 0 and mask[x, z-1] == 0:
            s = occ[x, z-1]
            if s == S_FIL or s == S_EBOT or s == S_ETOP:
                mask[x, z-1] = 1; qx[tail] = x; qz[tail] = z-1; tail += 1
        if z < h-1 and mask[x, z+1] == 0:
            s = occ[x, z+1]
            if s == S_FIL or s == S_EBOT or s == S_ETOP:
                mask[x, z+1] = 1; qx[tail] = x; qz[tail] = z+1; tail += 1
    return mask

@njit(cache=True)
def _connected_to_top(occ, mask):
    w, h = occ.shape
    z = h - 2
    for x in range(w):
        if mask[x, z] == 1 and occ[x, z+1] == S_ETOP:
            return True
    return False

@njit(cache=True)
def _tip_stats_from_filament(occ, mask):
    w, h = occ.shape
    tip_z = 1
    found_any = False
    for z in range(h-2, 0, -1):
        found = False
        for x in range(w):
            if mask[x, z] == 1 and occ[x, z] == S_FIL:
                tip_z = z
                found = True
                found_any = True
                break
        if found:
            break
    if not found_any:
        return 1, (w-1)/2.0
    sx = 0.0
    cnt = 0.0
    for x in range(w):
        if mask[x, tip_z] == 1 and occ[x, tip_z] == S_FIL:
            sx += x; cnt += 1.0
    tip_x = sx / cnt if cnt > 0 else (w-1)/2.0
    return tip_z, tip_x


# =========================
# 3) local neighborhood helpers
# =========================
@njit(cache=True)
def _count_attach_neighbors_including_electrodes(occ, x, z):
    cnt = 0
    w, h = occ.shape
    if x > 0:
        s = occ[x-1, z]
        if s == S_FIL or s == S_EBOT or s == S_ETOP: cnt += 1
    if x < w-1:
        s = occ[x+1, z]
        if s == S_FIL or s == S_EBOT or s == S_ETOP: cnt += 1
    if z > 0:
        s = occ[x, z-1]
        if s == S_FIL or s == S_EBOT or s == S_ETOP: cnt += 1
    if z < h-1:
        s = occ[x, z+1]
        if s == S_FIL or s == S_EBOT or s == S_ETOP: cnt += 1
    return cnt

@njit(cache=True)
def _count_fil_neighbors_only(occ, x, z):
    cnt = 0
    w, h = occ.shape
    if x > 0   and occ[x-1, z] == S_FIL: cnt += 1
    if x < w-1 and occ[x+1, z] == S_FIL: cnt += 1
    if z > 0   and occ[x, z-1] == S_FIL: cnt += 1
    if z < h-1 and occ[x, z+1] == S_FIL: cnt += 1
    return cnt

@njit(cache=True)
def _neighbor_flags_fil(occ, x, z):
    w, h = occ.shape
    left = (x > 0 and occ[x-1, z] == S_FIL)
    right = (x < w-1 and occ[x+1, z] == S_FIL)
    down = (z > 0 and occ[x, z-1] == S_FIL)
    up = (z < h-1 and occ[x, z+1] == S_FIL)
    return left, right, down, up


# =========================
# 4) morphology metrics (meander/neck/branches/drift/tortuosity)
# =========================
@njit(cache=True)
def _morphology_metrics_filament(occ, mask, cell_x_nm, drift_thresh_nm):
    w, h = occ.shape
    # neck
    neck_cells = 10**9
    for z in range(1, h-1):
        c = 0
        for x in range(w):
            if mask[x, z] == 1 and occ[x, z] == S_FIL:
                c += 1
        if c > 0 and c < neck_cells:
            neck_cells = c
    if neck_cells == 10**9:
        neck_cells = 0
    neck_nm = neck_cells * cell_x_nm

    # branches (deg>=3)
    branches = 0
    for z in range(1, h-1):
        for x in range(w):
            if mask[x, z] == 1 and occ[x, z] == S_FIL:
                deg = 0
                if x > 0   and mask[x-1, z] == 1 and occ[x-1, z] == S_FIL: deg += 1
                if x < w-1 and mask[x+1, z] == 1 and occ[x+1, z] == S_FIL: deg += 1
                if z > 1   and mask[x, z-1] == 1 and occ[x, z-1] == S_FIL: deg += 1
                if z < h-2 and mask[x, z+1] == 1 and occ[x, z+1] == S_FIL: deg += 1
                if deg >= 3:
                    branches += 1

    # COMx drift
    sx = 0.0; ctot = 0.0
    for z in range(1, h-1):
        for x in range(w):
            if mask[x, z] == 1 and occ[x, z] == S_FIL:
                sx += x; ctot += 1.0
    comx_nm = (sx / ctot) * cell_x_nm if ctot > 0 else -1.0
    center_nm = ((w-1)/2.0) * cell_x_nm
    drift_flag = 0
    if comx_nm >= 0.0 and abs(comx_nm - center_nm) > drift_thresh_nm:
        drift_flag = 1

    # meander RMS + tortuosity (sum |dx| between adjacent occupied layers)
    sum_x = 0.0; sum_x2 = 0.0; nz = 0.0
    tort = 0.0; last_x = -1.0
    for z in range(1, h-1):
        szz = 0.0; cntz = 0.0
        for x in range(w):
            if mask[x, z] == 1 and occ[x, z] == S_FIL:
                szz += x; cntz += 1.0
        if cntz > 0:
            xz = szz / cntz
            sum_x += xz; sum_x2 += xz * xz; nz += 1.0
            if last_x >= 0.0:
                tort += abs(xz - last_x) * cell_x_nm
            last_x = xz
    if nz > 1.0:
        mean_x = sum_x / nz
        var_x = sum_x2 / nz - mean_x * mean_x
        if var_x < 0.0: var_x = 0.0
        meander_rms_nm = np.sqrt(var_x) * cell_x_nm
    else:
        meander_rms_nm = 0.0
    return meander_rms_nm, neck_nm, branches, comx_nm, drift_flag, tort




# ==================================================
# 5) Phase-1 barrier generators + m-based initialization
# ==================================================
def build_ab_masks(mode=AB_MODE, active_zmax_frac=ACTIVE_Z_MAX_FRAC):
    """A/B 掩膜只定义在初始离子活跃窗口，用来计算 m。"""
    zmax = int(max(2, Z_OX_END * active_zmax_frac))
    ii, jj = np.meshgrid(np.arange(WIDTH_CELLS), np.arange(THICKNESS_CELLS), indexing='ij')
    active = (jj >= Z_OX_START) & (jj <= zmax)
    if mode == 'stripe_z':
        A = active & (((jj - Z_OX_START) % 2) == 0)
    elif mode == 'stripe_x':
        A = active & ((ii % 2) == 0)
    elif mode == 'checkerboard':
        A = active & (((ii + jj) % 2) == 0)
    else:
        raise ValueError(f'unknown AB mode: {mode}')
    B = active & (~A)
    return A, B, active, zmax

A_MASK, B_MASK, ACTIVE_MASK, Z_INIT_MAX = build_ab_masks()


def plot_ab_masks(A_mask=A_MASK, B_mask=B_MASK):
    arr = np.zeros_like(A_mask, dtype=int)
    arr[A_mask] = 1
    arr[B_mask] = 2
    plt.figure(figsize=(5.4, 4.4))
    plt.imshow(arr.T, origin='lower', aspect='auto')
    plt.title(f"A/B masks (mode={AB_MODE}, z<= {Z_INIT_MAX})")
    plt.xlabel('x cell')
    plt.ylabel('z cell')
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def counts_from_m_target(total_ions, nA_sites, nB_sites, m_target):
    """固定总缺陷数，反推出 A/B 上的目标缺陷数。"""
    m_target = float(np.clip(m_target, -0.999, 0.999))
    denom = (1.0 + m_target) * nA_sites + (1.0 - m_target) * nB_sites
    c_sum = 2.0 * total_ions / max(denom, 1e-12)
    cA = 0.5 * c_sum * (1.0 + m_target)
    cB = 0.5 * c_sum * (1.0 - m_target)
    nA = int(round(cA * nA_sites))
    nA = max(0, min(nA, int(nA_sites)))
    nB = int(total_ions - nA)
    if nB < 0:
        nB = 0
        nA = min(total_ions, int(nA_sites))
    if nB > nB_sites:
        nB = int(nB_sites)
        nA = min(total_ions - nB, int(nA_sites))
    return nA, nB


def init_occ_from_m(m_target, seed=0, A_mask=A_MASK, B_mask=B_MASK):
    rng = np.random.default_rng(seed)
    occ = np.zeros((WIDTH_CELLS, THICKNESS_CELLS), dtype=np.int8)
    occ[:, Z_BOT] = S_EBOT
    occ[:, Z_TOP] = S_ETOP

    A_idx = np.argwhere(A_mask)
    B_idx = np.argwhere(B_mask)
    nA, nB = counts_from_m_target(N_INIT_IONS, len(A_idx), len(B_idx), m_target)

    if nA > 0:
        pickA = rng.choice(len(A_idx), size=nA, replace=False)
        for idx in pickA:
            x, z = A_idx[int(idx)]
            occ[x, z] = S_ION
    if nB > 0:
        pickB = rng.choice(len(B_idx), size=nB, replace=False)
        for idx in pickB:
            x, z = B_idx[int(idx)]
            occ[x, z] = S_ION

    # nucleation seed 沿用 v5.1 逻辑
    if NUCLEATION_MODE == "center":
        occ[WIDTH_CELLS//2, 1] = S_FIL
    elif NUCLEATION_MODE == "random":
        occ[int(rng.integers(0, WIDTH_CELLS)), 1] = S_FIL
    elif NUCLEATION_MODE == "random_multi":
        xs = rng.choice(np.arange(WIDTH_CELLS), size=min(NUC_SEEDS, WIDTH_CELLS), replace=False)
        for x in xs:
            occ[int(x), 1] = S_FIL
    else:
        xc = (WIDTH_CELLS - 1) / 2.0
        xs = rng.choice(np.arange(WIDTH_CELLS), size=min(NUC_SEEDS, WIDTH_CELLS), replace=False)
        xs = sorted(xs, key=lambda x: abs(x - xc))
        for x in xs:
            occ[int(x), 1] = S_FIL
    return occ


def compute_m_from_occ(occ, A_mask=A_MASK, B_mask=B_MASK):
    ionA = np.count_nonzero((occ == S_ION) & A_mask)
    ionB = np.count_nonzero((occ == S_ION) & B_mask)
    nA = np.count_nonzero(A_mask)
    nB = np.count_nonzero(B_mask)
    cA = ionA / max(nA, 1)
    cB = ionB / max(nB, 1)
    m = (cA - cB) / max(cA + cB, 1e-12)
    return cA, cB, float(m), float(abs(m)), int(ionA), int(ionB), int(nA), int(nB)


def sigma_from_m(m_abs, sigma_res=SIGMA_RES_EV, sigma_dis=SIGMA_DIS_EV, p=P_SIGMA):
    return sigma_res + sigma_dis * (1.0 - float(np.clip(m_abs, 0.0, 1.0))**p)


def lc_from_m(m_abs, l0=LC0_CELLS, beta=LC_BETA_CELLS, q=Q_LC):
    return l0 + beta * (float(np.clip(m_abs, 0.0, 1.0))**q)


def delta_from_sigmaT(sigma_e, T):
    return sigma_e / (KB * T)


def correlated_gaussian_field(nx, nz, lc_cells, rng):
    """FFT-based correlated Gaussian field. lc_cells 是格点单位。"""
    z = rng.normal(size=(nx, nz))
    if lc_cells <= 1e-12:
        eta = z
    else:
        kx = 2.0 * np.pi * np.fft.fftfreq(nx)
        kz = 2.0 * np.pi * np.fft.fftfreq(nz)
        KX, KZ = np.meshgrid(kx, kz, indexing='ij')
        filt = np.exp(-0.5 * (lc_cells**2) * (KX**2 + KZ**2))
        eta = np.fft.ifft2(np.fft.fft2(z) * filt).real
    eta = eta - eta.mean()
    eta = eta / (eta.std() + 1e-12)
    return eta


def build_barrier_maps_from_m(m_abs, barrier_seed, mu_x_layer=MU_X_LAYER, mu_z_layer=MU_Z_LAYER):
    rng = np.random.default_rng(barrier_seed)
    sigma_E = sigma_from_m(m_abs)
    lc = lc_from_m(m_abs)
    eta = correlated_gaussian_field(WIDTH_CELLS, THICKNESS_CELLS, lc, rng)
    eta[:, Z_BOT] = 0.0
    eta[:, Z_TOP] = 0.0

    mu_x_site = np.tile(np.asarray(mu_x_layer, dtype=float), (WIDTH_CELLS, 1))
    mu_z_site = np.tile(np.asarray(mu_z_layer, dtype=float), (WIDTH_CELLS, 1))
    em_x = mu_x_site + sigma_E * eta
    em_z = mu_z_site + sigma_E * eta
    em_x[:, Z_BOT] = BASE_BARRIER_EV; em_x[:, Z_TOP] = BASE_BARRIER_EV
    em_z[:, Z_BOT] = BASE_BARRIER_EV; em_z[:, Z_TOP] = BASE_BARRIER_EV
    em_x = np.maximum(em_x, MIN_BARRIER_EV).astype(np.float64)
    em_z = np.maximum(em_z, MIN_BARRIER_EV).astype(np.float64)
    meta = dict(sigma_E=float(sigma_E), lc=float(lc), eta=eta, mu_x_site=mu_x_site, mu_z_site=mu_z_site)
    return em_x, em_z, meta


def plot_barrier_bundle(meta, em_x, em_z, occ0=None, title_prefix=""):
    fig, axes = plt.subplots(1, 4 if occ0 is not None else 3, figsize=(16 if occ0 is not None else 12, 4))
    ax0 = axes[0]
    im0 = ax0.imshow(meta['eta'].T, origin='lower', aspect='auto')
    ax0.set_title('correlated eta(x,z)')
    plt.colorbar(im0, ax=ax0, fraction=0.046)

    ax1 = axes[1]
    im1 = ax1.imshow(em_z.T, origin='lower', aspect='auto')
    ax1.set_title('em_z (vertical)')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = axes[2]
    im2 = ax2.imshow(em_x.T, origin='lower', aspect='auto')
    ax2.set_title('em_x (lateral)')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    if occ0 is not None:
        ax3 = axes[3]
        im3 = ax3.imshow(occ0.T, origin='lower', aspect='auto')
        ax3.set_title('initial occupancy occ0')
        plt.colorbar(im3, ax=ax3, fraction=0.046)

    for ax in axes:
        ax.set_xlabel('x cell')
        ax.set_ylabel('z cell')
    fig.suptitle(title_prefix)
    plt.tight_layout()
    plt.show()

# =========================
# 7) KMC core (anisotropic barriers em_x/em_z)
# =========================
@njit(cache=True)
def _run_kmc(
    occ0, em_x, em_z,
    E_avg_vnm, T,
    max_steps,
    do_growth, do_dissolve,
    stop_if_connected,
    stop_if_disconnected,
    seed,
    rate_sample_max,
    rate_sample_stride,
    mask_update_stride,
    alpha_field,
    min_barrier,
    field_enh_mig,
    field_enh_grow,
    tip_focus_power,
    tip_focus_max,
    enable_tip_focus,
    e_grow0,
    gamma_grow,
    delta_attach,
    hop_slow_near_tip,
    tip_behind,
    tip_ahead,
    e_diss0,
    gamma_diss,
    phib_bottom,
    phib_top,
    e_inject0,
    focus_kappa,
    e_lat_noise_vnm,
    grow_pref_forward,
    grow_penalty_lateral,
    cell_z_nm,
    enable_center_bias,
    center_bias_strength,
    center_bias_sigma_cells
):
    np.random.seed(seed)
    occ = occ0.copy()
    w, h = occ.shape

    t = 0.0
    t_connect = np.nan
    last_diss_x = -1
    last_diss_z = -1

    n_hopx = 0
    n_hopz = 0
    n_grow = 0
    n_diss = 0
    n_inj  = 0

    rates_s = np.zeros(rate_sample_max, dtype=np.float64)
    ns = 0

    max_events = w * (h-2) * 7 + w * 2
    rates = np.empty(max_events, dtype=np.float64)
    etype = np.empty(max_events, dtype=np.int8)     # 0 hop, 1 grow, 2 diss, 3 injB, 4 injT
    ex = np.empty(max_events, dtype=np.int16)
    ez = np.empty(max_events, dtype=np.int16)
    enx = np.empty(max_events, dtype=np.int16)
    enz = np.empty(max_events, dtype=np.int16)

    mask = _bfs_bottom_connected_mask(occ)
    connected = _connected_to_top(occ, mask)
    tip_z, tip_x = _tip_stats_from_filament(occ, mask)

    for step in range(max_steps):

        if step % mask_update_stride == 0:
            mask = _bfs_bottom_connected_mask(occ)
            connected = _connected_to_top(occ, mask)
            tip_z, tip_x = _tip_stats_from_filament(occ, mask)

            if connected and np.isnan(t_connect):
                t_connect = t
                if stop_if_connected:
                    energy_proxy_ev = abs(E_avg_vnm) * cell_z_nm * n_hopz
                    return (occ, t_connect, True, rates_s, ns, last_diss_x, last_diss_z,
                            n_hopx, n_hopz, n_grow, n_diss, n_inj, energy_proxy_ev)

            if (not connected) and (not np.isnan(t_connect)) and stop_if_disconnected:
                energy_proxy_ev = abs(E_avg_vnm) * cell_z_nm * n_hopz
                return (occ, t_connect, False, rates_s, ns, last_diss_x, last_diss_z,
                        n_hopx, n_hopz, n_grow, n_diss, n_inj, energy_proxy_ev)

        gap = (h - 1) - tip_z
        E_mig = abs(E_avg_vnm) * field_enh_mig
        E_grow_base = abs(E_avg_vnm) * field_enh_grow
        E_tip = _tip_field(E_grow_base, gap, tip_focus_power, tip_focus_max, enable_tip_focus)

        Ex_noise = e_lat_noise_vnm * (np.random.randn())

        n_evt = 0
        total_rate = 0.0

        # (1) injection: simplified
        if E_avg_vnm > 0.0:
            z = 1
            for x in range(w):
                if occ[x, z] == S_EMPTY:
                    Eb_inj = phib_bottom + e_inject0 - 0.15 * E_mig
                    Eb_inj = _clamp(Eb_inj, min_barrier)
                    r = NU0 * np.exp(-Eb_inj / (KB * T))
                    rates[n_evt] = r; etype[n_evt] = 3
                    ex[n_evt] = x; ez[n_evt] = z
                    enx[n_evt] = x; enz[n_evt] = z
                    total_rate += r; n_evt += 1
        elif E_avg_vnm < 0.0:
            z = h - 2
            for x in range(w):
                if occ[x, z] == S_EMPTY:
                    Eb_inj = phib_top + e_inject0 - 0.15 * E_mig
                    Eb_inj = _clamp(Eb_inj, min_barrier)
                    r = NU0 * np.exp(-Eb_inj / (KB * T))
                    rates[n_evt] = r; etype[n_evt] = 4
                    ex[n_evt] = x; ez[n_evt] = z
                    enx[n_evt] = x; enz[n_evt] = z
                    total_rate += r; n_evt += 1

        # (2) iterate oxide sites
        for z in range(1, h-1):
            for x in range(w):
                s = occ[x, z]

                if s == S_ION:
                    # z hop (use em_z)
                    if z+1 <= h-2 and occ[x, z+1] == S_EMPTY:
                        Eb = em_z[x, z] - alpha_field * E_mig * 1.0
                        Eb = _clamp(Eb, min_barrier)
                        r = NU0 * np.exp(-Eb / (KB * T))
                        if z >= tip_z and z <= tip_z + tip_ahead:
                            na = _count_attach_neighbors_including_electrodes(occ, x, z)
                            if na > 0: r *= hop_slow_near_tip
                        rates[n_evt] = r; etype[n_evt] = 0
                        ex[n_evt] = x; ez[n_evt] = z
                        enx[n_evt] = x; enz[n_evt] = z+1
                        total_rate += r; n_evt += 1

                    if z-1 >= 1 and occ[x, z-1] == S_EMPTY:
                        Eb = em_z[x, z] - alpha_field * E_mig * (-1.0)
                        Eb = _clamp(Eb, min_barrier)
                        r = NU0 * np.exp(-Eb / (KB * T))
                        if z >= tip_z and z <= tip_z + tip_ahead:
                            na = _count_attach_neighbors_including_electrodes(occ, x, z)
                            if na > 0: r *= hop_slow_near_tip
                        rates[n_evt] = r; etype[n_evt] = 0
                        ex[n_evt] = x; ez[n_evt] = z
                        enx[n_evt] = x; enz[n_evt] = z-1
                        total_rate += r; n_evt += 1

                    # x hop (use em_x)
                    if x-1 >= 0 and occ[x-1, z] == S_EMPTY:
                        Eb = em_x[x, z] - alpha_field * (Ex_noise) * (-1.0)
                        Eb = _clamp(Eb, min_barrier)
                        r = NU0 * np.exp(-Eb / (KB * T))
                        if z >= tip_z and z <= tip_z + tip_ahead:
                            na = _count_attach_neighbors_including_electrodes(occ, x, z)
                            if na > 0: r *= hop_slow_near_tip
                        rates[n_evt] = r; etype[n_evt] = 0
                        ex[n_evt] = x; ez[n_evt] = z
                        enx[n_evt] = x-1; enz[n_evt] = z
                        total_rate += r; n_evt += 1

                    if x+1 <= w-1 and occ[x+1, z] == S_EMPTY:
                        Eb = em_x[x, z] - alpha_field * (Ex_noise) * (1.0)
                        Eb = _clamp(Eb, min_barrier)
                        r = NU0 * np.exp(-Eb / (KB * T))
                        if z >= tip_z and z <= tip_z + tip_ahead:
                            na = _count_attach_neighbors_including_electrodes(occ, x, z)
                            if na > 0: r *= hop_slow_near_tip
                        rates[n_evt] = r; etype[n_evt] = 0
                        ex[n_evt] = x; ez[n_evt] = z
                        enx[n_evt] = x+1; enz[n_evt] = z
                        total_rate += r; n_evt += 1

                    # growth: ION -> FIL
                    if do_growth:
                        if z >= tip_z - tip_behind and z <= tip_z + tip_ahead:
                            na_fil = _count_fil_neighbors_only(occ, x, z)
                            if na_fil > 0:
                                Eb_g = e_grow0 - gamma_grow * E_tip - delta_attach * na_fil
                                Eb_g = _clamp(Eb_g, min_barrier)
                                r = NU0 * np.exp(-Eb_g / (KB * T))

                                l, rr, d, u = _neighbor_flags_fil(occ, x, z)
                                orient = 1.0
                                if d and (not l) and (not rr):
                                    orient = grow_pref_forward
                                elif (l or rr) and (not d):
                                    orient = grow_penalty_lateral
                                r *= orient

                                if focus_kappa > 0.0:
                                    dx = abs(x - tip_x)
                                    r *= 1.0 / (1.0 + focus_kappa * dx * dx)

                                r *= _center_bias_factor(x, w, enable_center_bias,
                                                        center_bias_strength, center_bias_sigma_cells)

                                rates[n_evt] = r; etype[n_evt] = 1
                                ex[n_evt] = x; ez[n_evt] = z
                                enx[n_evt] = x; enz[n_evt] = z
                                total_rate += r; n_evt += 1

                elif s == S_FIL and do_dissolve:
                    # FIL -> ION if has empty neighbor
                    na = 0
                    if x > 0   and occ[x-1, z] == S_EMPTY: na += 1
                    if x < w-1 and occ[x+1, z] == S_EMPTY: na += 1
                    if z > 1   and occ[x, z-1] == S_EMPTY: na += 1
                    if z < h-2 and occ[x, z+1] == S_EMPTY: na += 1
                    if na > 0:
                        Eb_d = e_diss0 - gamma_diss * E_tip
                        Eb_d = _clamp(Eb_d, min_barrier)
                        r = NU0 * np.exp(-Eb_d / (KB * T))
                        rates[n_evt] = r; etype[n_evt] = 2
                        ex[n_evt] = x; ez[n_evt] = z
                        enx[n_evt] = x; enz[n_evt] = z
                        total_rate += r; n_evt += 1

        if total_rate < 1e-40 or n_evt <= 0:
            energy_proxy_ev = abs(E_avg_vnm) * cell_z_nm * n_hopz
            return (occ, t_connect, connected, rates_s, ns, last_diss_x, last_diss_z,
                    n_hopx, n_hopz, n_grow, n_diss, n_inj, energy_proxy_ev)

        # KMC time increment
        u = np.random.random()
        dt = -np.log(u) / total_rate
        t += dt

        # roulette select
        rsel = np.random.random() * total_rate
        csum = 0.0
        sel = -1
        for i in range(n_evt):
            csum += rates[i]
            if csum >= rsel:
                sel = i
                break
        if sel < 0:
            continue

        if (step % rate_sample_stride == 0) and (ns < rate_sample_max):
            rates_s[ns] = rates[sel]
            ns += 1

        tp = etype[sel]
        x0 = ex[sel]; z0 = ez[sel]
        x1 = enx[sel]; z1 = enz[sel]

        if tp == 0:
            occ[x0, z0] = S_EMPTY
            occ[x1, z1] = S_ION
            if x1 != x0: n_hopx += 1
            if z1 != z0: n_hopz += 1
        elif tp == 1:
            occ[x0, z0] = S_FIL
            n_grow += 1
        elif tp == 2:
            occ[x0, z0] = S_ION
            last_diss_x = x0; last_diss_z = z0
            n_diss += 1
        elif tp == 3:
            occ[x0, z0] = S_ION
            n_inj += 1
        elif tp == 4:
            occ[x0, z0] = S_ION
            n_inj += 1

    mask = _bfs_bottom_connected_mask(occ)
    connected = _connected_to_top(occ, mask)
    energy_proxy_ev = abs(E_avg_vnm) * cell_z_nm * n_hopz
    return (occ, t_connect, connected, rates_s, ns, last_diss_x, last_diss_z,
            n_hopx, n_hopz, n_grow, n_diss, n_inj, energy_proxy_ev)


# =========================
# 8) wrappers: SET / LRS / RESET
# =========================
def run_set(occ0, em_x, em_z, E_vnm, T, mc_seed):
    return _run_kmc(
        occ0, em_x, em_z,
        E_vnm, T,
        MAX_STEPS_SET,
        True, False,
        STOP_IF_CONNECTED_SET, False,
        mc_seed,
        RATE_SAMPLE_MAX, RATE_SAMPLE_STRIDE,
        MASK_UPDATE_STRIDE,
        ALPHA_FIELD, MIN_BARRIER_EV,
        FIELD_ENHANCE_MIG, FIELD_ENHANCE_GROW,
        TIP_FOCUS_POWER, TIP_FOCUS_MAX_VNM, ENABLE_TIP_FOCUS,
        E_GROW0_EV, GAMMA_GROW, DELTA_ATTACH,
        HOP_SLOW_NEAR_TIP, TIP_BEHIND, TIP_AHEAD,
        E_DISS0_EV, GAMMA_DISS,
        PHIB_BOTTOM, PHIB_TOP, E_INJECT0_EV,
        FOCUS_KAPPA,
        E_LATERAL_NOISE_VNM,
        GROW_PREF_FORWARD,
        GROW_PENALTY_LATERAL,
        CELL_Z_NM,
        ENABLE_CENTER_BIAS,
        CENTER_BIAS_STRENGTH,
        CENTER_BIAS_SIGMA_CELLS
    )

def run_post_thicken(occ_crit, em_x, em_z, E_vnm, T, mc_seed):
    return _run_kmc(
        occ_crit, em_x, em_z,
        E_vnm, T,
        POST_CONNECT_STEPS,
        True, False,
        False, False,
        mc_seed,
        RATE_SAMPLE_MAX, RATE_SAMPLE_STRIDE,
        MASK_UPDATE_STRIDE,
        ALPHA_FIELD, MIN_BARRIER_EV,
        FIELD_ENHANCE_MIG, FIELD_ENHANCE_GROW,
        TIP_FOCUS_POWER, TIP_FOCUS_MAX_VNM, ENABLE_TIP_FOCUS,
        E_GROW0_EV, GAMMA_GROW, DELTA_ATTACH,
        HOP_SLOW_NEAR_TIP, TIP_BEHIND, TIP_AHEAD,
        E_DISS0_EV, GAMMA_DISS,
        PHIB_BOTTOM, PHIB_TOP, E_INJECT0_EV,
        FOCUS_KAPPA,
        E_LATERAL_NOISE_VNM,
        GROW_PREF_FORWARD,
        GROW_PENALTY_LATERAL,
        CELL_Z_NM,
        ENABLE_CENTER_BIAS,
        CENTER_BIAS_STRENGTH,
        CENTER_BIAS_SIGMA_CELLS
    )

def run_reset(occ_lrs, em_x, em_z, E_vnm, T, mc_seed):
    return _run_kmc(
        occ_lrs, em_x, em_z,
        -abs(E_vnm), T,
        MAX_STEPS_RESET,
        False, True,
        False, STOP_IF_DISCONNECTED_RESET,
        mc_seed,
        RATE_SAMPLE_MAX, RATE_SAMPLE_STRIDE,
        MASK_UPDATE_STRIDE,
        ALPHA_FIELD, MIN_BARRIER_EV,
        FIELD_ENHANCE_MIG, FIELD_ENHANCE_GROW,
        TIP_FOCUS_POWER, TIP_FOCUS_MAX_VNM, ENABLE_TIP_FOCUS,
        E_GROW0_EV, GAMMA_GROW, DELTA_ATTACH,
        HOP_SLOW_NEAR_TIP, TIP_BEHIND, TIP_AHEAD,
        E_DISS0_EV, GAMMA_DISS,
        PHIB_BOTTOM, PHIB_TOP, E_INJECT0_EV,
        FOCUS_KAPPA,
        E_LATERAL_NOISE_VNM,
        GROW_PREF_FORWARD,
        GROW_PENALTY_LATERAL,
        CELL_Z_NM,
        ENABLE_CENTER_BIAS,
        CENTER_BIAS_STRENGTH,
        CENTER_BIAS_SIGMA_CELLS
    )




# ==================================================
# 9) phase-1 simulation / scan / plotting
# ==================================================
def morphology_from_occ(occ):
    mask = _bfs_bottom_connected_mask(occ)
    return _morphology_metrics_filament(occ, mask, CELL_X_NM, DRIFT_THRESH_NM)


def simulate_one_phase1(m0_target, E_vnm=PHASE1_E0, T=PHASE1_T0,
                        structure_seed=2025, mc_seed=0,
                        return_snapshots=False, store_rate_samples=False):
    # 1) occupancy from m0
    occ0 = init_occ_from_m(m0_target, seed=structure_seed)
    cA, cB, m_actual, m_abs, ionA, ionB, nA, nB = compute_m_from_occ(occ0)

    # 2) build barrier map from actual m
    em_x, em_z, meta = build_barrier_maps_from_m(m_abs, barrier_seed=structure_seed)

    # 3) KMC SET -> critical
    snaps = {'HRS0': occ0.copy()} if return_snapshots else None
    occ_crit, t_set, connected, rates_s, ns, _, _, n_hopx, n_hopz, n_grow, n_diss, n_inj, energy_set = run_set(
        occ0, em_x, em_z, E_vnm, T, mc_seed
    )
    formed = bool(connected)
    if return_snapshots:
        snaps['Critical'] = occ_crit.copy()

    # 4) optional post-thicken to define LRS morphology
    occ_lrs = occ_crit
    if formed and ENABLE_POST_THICKEN:
        occ_lrs, *_ = run_post_thicken(occ_crit, em_x, em_z, E_vnm, T, mc_seed + 77777)
    if return_snapshots:
        snaps['LRS'] = occ_lrs.copy()

    meander_rms_nm, neck_nm, branches, comx_nm, drift_flag, tort_nm = morphology_from_occ(occ_lrs)
    barrier_std_x = float(np.std(em_x[:, Z_OX_START:Z_OX_END+1]))
    barrier_std_z = float(np.std(em_z[:, Z_OX_START:Z_OX_END+1]))

    out = dict(
        m_target=float(m0_target),
        m_actual=float(m_actual),
        m_abs=float(m_abs),
        cA=float(cA), cB=float(cB), ionA=int(ionA), ionB=int(ionB),
        nA_sites=int(nA), nB_sites=int(nB),
        sigma_E=float(meta['sigma_E']),
        lc=float(meta['lc']),
        Delta=float(delta_from_sigmaT(meta['sigma_E'], T)),
        barrier_std_x=barrier_std_x,
        barrier_std_z=barrier_std_z,
        structure_seed=int(structure_seed),
        mc_seed=int(mc_seed),
        E=float(E_vnm), T=float(T),
        formed=int(formed),
        t_set=float(t_set),
        energy_set_ev=float(energy_set),
        n_hopx=int(n_hopx), n_hopz=int(n_hopz), n_grow=int(n_grow), n_inj=int(n_inj),
        meander_rms_nm=float(meander_rms_nm),
        tortuosity_nm=float(tort_nm),
        neck_nm=float(neck_nm),
        branches=int(branches),
        comx_nm=float(comx_nm),
        drift=int(drift_flag),
        rate_var_log=float(np.var(np.log10(rates_s[:ns] + 1e-40))) if ns > 5 else np.nan,
    )

    if store_rate_samples:
        out['rate_samples'] = rates_s[:max(1, ns)].copy()
    if return_snapshots:
        out['snapshots'] = snaps
        out['occ0'] = occ0.copy()
        out['em_x'] = em_x.copy()
        out['em_z'] = em_z.copy()
        out['eta'] = meta['eta'].copy()
    return out


def run_phase1_scan(m0_list=M0_TARGET_LIST, E0=PHASE1_E0, T0=PHASE1_T0,
                    n_structure_seeds=N_STRUCTURE_SEEDS,
                    n_mc_per_structure=N_MC_PER_STRUCTURE,
                    n_jobs=N_JOBS):
    tasks = []
    for m0 in m0_list:
        for sdev in range(n_structure_seeds):
            structure_seed = 2025 + 1000 * sdev + int(round(m0 * 100))
            for smc in range(n_mc_per_structure):
                mc_seed = 10_000 * structure_seed + smc
                tasks.append((float(m0), int(structure_seed), int(mc_seed)))

    def _task(m0, structure_seed, mc_seed):
        return simulate_one_phase1(m0_target=m0, E_vnm=E0, T=T0,
                                   structure_seed=structure_seed, mc_seed=mc_seed,
                                   return_snapshots=False, store_rate_samples=False)

    t0 = time.time()
    if n_jobs == 1:
        rows = [_task(m0, ss, ms) for (m0, ss, ms) in tasks]
    else:
        rows = Parallel(n_jobs=n_jobs, prefer='threads', batch_size=1)(
            delayed(_task)(m0, ss, ms) for (m0, ss, ms) in tasks
        )
    df = pd.DataFrame(rows)
    df['log10_t_set'] = np.log10(df['t_set'].clip(lower=1e-30))
    df['m_target_label'] = df['m_target'].map(lambda v: f"{v:.2f}")
    print(f"Phase-1 scan done: {len(df)} runs in {time.time()-t0:.1f} s")
    return df


def summarize_phase1(df):
    g = df.groupby('m_target', as_index=False).agg(
        m_actual_mean=('m_actual', 'mean'),
        m_actual_std=('m_actual', 'std'),
        sigma_E_mean=('sigma_E', 'mean'),
        sigma_E_std=('sigma_E', 'std'),
        lc_mean=('lc', 'mean'),
        lc_std=('lc', 'std'),
        Delta_mean=('Delta', 'mean'),
        formed_prob=('formed', 'mean'),
        log10_t_set_mean=('log10_t_set', 'mean'),
        log10_t_set_std=('log10_t_set', 'std'),
        neck_nm_mean=('neck_nm', 'mean'),
        tortuosity_nm_mean=('tortuosity_nm', 'mean'),
        branches_mean=('branches', 'mean'),
        barrier_std_z_mean=('barrier_std_z', 'mean'),
        rate_var_log_mean=('rate_var_log', 'mean'),
    )
    return g


def plot_phase1_summary(df):
    summary = summarize_phase1(df)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0,0]
    ax.errorbar(summary['m_target'], summary['m_actual_mean'], yerr=summary['m_actual_std'], marker='o')
    ax.plot(summary['m_target'], summary['m_target'], '--', alpha=0.6)
    ax.set_title('m_actual vs m_target')
    ax.set_xlabel('m_target'); ax.set_ylabel('m_actual')

    ax = axes[0,1]
    sns.boxplot(data=df, x='m_target_label', y='log10_t_set', ax=ax)
    ax.set_title('log10(t_set) distribution')
    ax.set_xlabel('m_target'); ax.set_ylabel('log10 t_set (s)')

    ax = axes[0,2]
    ax.plot(summary['m_target'], summary['formed_prob'], marker='o')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Formation probability')
    ax.set_xlabel('m_target'); ax.set_ylabel('P(form)')

    ax = axes[1,0]
    ax.plot(summary['m_target'], summary['barrier_std_z_mean'], marker='o', label='std em_z')
    ax.plot(summary['m_target'], summary['sigma_E_mean'], marker='s', label='sigma_E input')
    ax.set_title('Barrier-width check')
    ax.set_xlabel('m_target'); ax.set_ylabel('eV')
    ax.legend()

    ax = axes[1,1]
    ax.plot(summary['m_target'], summary['tortuosity_nm_mean'], marker='o', label='tortuosity')
    ax.plot(summary['m_target'], summary['neck_nm_mean'], marker='s', label='neck width')
    ax.set_title('Morphology vs m')
    ax.set_xlabel('m_target'); ax.set_ylabel('nm')
    ax.legend()

    ax = axes[1,2]
    ax.plot(summary['m_target'], summary['branches_mean'], marker='o', label='branches')
    ax.plot(summary['m_target'], summary['rate_var_log_mean'], marker='s', label='var(log10 rate)')
    ax.set_title('Topology / kinetic heterogeneity')
    ax.set_xlabel('m_target')
    ax.legend()

    for a in axes.ravel():
        _no_offset(a)
    plt.tight_layout()
    plt.show()
    return summary


def export_phase1_results(df, prefix='phase1_mscan'):
    summary = summarize_phase1(df)
    csv1 = SAVE_DIR / f'{prefix}_raw.csv'
    csv2 = SAVE_DIR / f'{prefix}_summary.csv'
    df.to_csv(csv1, index=False)
    summary.to_csv(csv2, index=False)
    print('saved:', csv1)
    print('saved:', csv2)
    return csv1, csv2


def run_one_demo_case(m0_target=0.75, structure_seed=2025, mc_seed=1,
                      E0=PHASE1_E0, T0=PHASE1_T0):
    demo = simulate_one_phase1(m0_target=m0_target, E_vnm=E0, T=T0,
                               structure_seed=structure_seed, mc_seed=mc_seed,
                               return_snapshots=True)
    print({k: demo[k] for k in ['m_target', 'm_actual', 'sigma_E', 'lc', 'Delta', 'formed', 't_set', 'neck_nm', 'tortuosity_nm', 'branches']})
    plot_barrier_bundle(
        meta={'eta': demo['eta']},
        em_x=demo['em_x'], em_z=demo['em_z'], occ0=demo['occ0'],
        title_prefix=f"demo m_target={m0_target:.2f}, m_actual={demo['m_actual']:.3f}"
    )
    snaps = demo['snapshots']
    keys = list(snaps.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4))
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        ax.imshow(snaps[key].T, origin='lower', aspect='auto')
        ax.set_title(key)
        ax.set_xlabel('x cell'); ax.set_ylabel('z cell')
    plt.tight_layout()
    plt.show()
    return demo


# ==================================================
# Phase-2 controls: m–E / m–T coupling scans
# ==================================================
PHASE2_MODE = "mE"   # "mE" or "mT"
PHASE2_M_LIST = [0.00, 0.25, 0.50, 0.75, 0.90]
PHASE2_E_LIST = [0.045, 0.060, 0.075, 0.090, 0.105, 0.120]
PHASE2_T_LIST = [500.0, 600.0, 700.0, 800.0, 900.0]
PHASE2_E_FIXED = PHASE1_E0
PHASE2_T_FIXED = PHASE1_T0
PHASE2_N_STRUCTURE_SEEDS = 3 if FAST_MODE else 8
PHASE2_N_MC_PER_STRUCTURE = 2 if FAST_MODE else 4
PHASE2_N_JOBS = N_JOBS
PHASE2_SAVE_TAG = "phase2"


def simulate_one_phase2(m0_target, E_vnm, T,
                        structure_seed=2025, mc_seed=0,
                        return_snapshots=False, store_rate_samples=False):
    return simulate_one_phase1(
        m0_target=m0_target,
        E_vnm=E_vnm,
        T=T,
        structure_seed=structure_seed,
        mc_seed=mc_seed,
        return_snapshots=return_snapshots,
        store_rate_samples=store_rate_samples,
    )


def run_phase2_grid(mode=PHASE2_MODE,
                    m_list=PHASE2_M_LIST,
                    E_list=PHASE2_E_LIST,
                    T_list=PHASE2_T_LIST,
                    E_fixed=PHASE2_E_FIXED,
                    T_fixed=PHASE2_T_FIXED,
                    n_structure_seeds=PHASE2_N_STRUCTURE_SEEDS,
                    n_mc_per_structure=PHASE2_N_MC_PER_STRUCTURE,
                    n_jobs=PHASE2_N_JOBS):
    """
    mode="mE": 扫 m 和 E，温度固定为 T_fixed
    mode="mT": 扫 m 和 T，电场固定为 E_fixed
    """
    assert mode in ["mE", "mT"]
    tasks = []
    if mode == "mE":
        for m0 in m_list:
            for E0 in E_list:
                for sdev in range(n_structure_seeds):
                    structure_seed = 2025 + 1000 * sdev + int(round(m0 * 100)) + int(round(E0 * 1000))
                    for smc in range(n_mc_per_structure):
                        mc_seed = 10_000 * structure_seed + smc
                        tasks.append((float(m0), float(E0), float(T_fixed), int(structure_seed), int(mc_seed)))
    else:
        for m0 in m_list:
            for T0 in T_list:
                for sdev in range(n_structure_seeds):
                    structure_seed = 2025 + 1000 * sdev + int(round(m0 * 100)) + int(round(T0))
                    for smc in range(n_mc_per_structure):
                        mc_seed = 10_000 * structure_seed + smc
                        tasks.append((float(m0), float(E_fixed), float(T0), int(structure_seed), int(mc_seed)))

    def _task(m0, E0, T0, structure_seed, mc_seed):
        return simulate_one_phase2(
            m0_target=m0, E_vnm=E0, T=T0,
            structure_seed=structure_seed, mc_seed=mc_seed,
            return_snapshots=False, store_rate_samples=False
        )

    t0 = time.time()
    if n_jobs == 1:
        rows = [_task(m0, E0, T0, ss, ms) for (m0, E0, T0, ss, ms) in tasks]
    else:
        rows = Parallel(n_jobs=n_jobs, prefer='threads', batch_size=1)(
            delayed(_task)(m0, E0, T0, ss, ms) for (m0, E0, T0, ss, ms) in tasks
        )
    df = pd.DataFrame(rows)
    df['log10_t_set'] = np.log10(df['t_set'].clip(lower=1e-30))
    df['m_target_label'] = df['m_target'].map(lambda v: f"{v:.2f}")
    df['E_label'] = df['E'].map(lambda v: f"{v:.3f}")
    df['T_label'] = df['T'].map(lambda v: f"{v:.0f}")
    print(f"Phase-2 {mode} scan done: {len(df)} runs in {time.time()-t0:.1f} s")
    return df


def summarize_phase2(df, mode="mE"):
    key2 = 'E' if mode == 'mE' else 'T'
    g = df.groupby(['m_target', key2], as_index=False).agg(
        runs=('formed', 'size'),
        m_actual_mean=('m_actual', 'mean'),
        sigma_E_mean=('sigma_E', 'mean'),
        lc_mean=('lc', 'mean'),
        Delta_mean=('Delta', 'mean'),
        formed_prob=('formed', 'mean'),
        log10_t_set_mean=('log10_t_set', 'mean'),
        log10_t_set_std=('log10_t_set', 'std'),
        neck_nm_mean=('neck_nm', 'mean'),
        tortuosity_nm_mean=('tortuosity_nm', 'mean'),
        branches_mean=('branches', 'mean'),
        barrier_std_z_mean=('barrier_std_z', 'mean'),
        rate_var_log_mean=('rate_var_log', 'mean'),
        energy_set_mean=('energy_set_ev', 'mean'),
    )

    # 额外给一个简单的 regime 标签，便于后面画边界/挑点
    def classify_row(row):
        p = row['formed_prob']
        br = row['branches_mean']
        tt = row['tortuosity_nm_mean']
        if p < 0.2:
            return 'HRS-like'
        if p >= 0.2 and p < 0.8:
            return 'critical'
        if br <= 8 and tt <= 28:
            return 'single-path LRS'
        return 'multi-path LRS'

    g['regime'] = g.apply(classify_row, axis=1)
    return g


def _pivot_metric(summary, mode, metric):
    key2 = 'E' if mode == 'mE' else 'T'
    pv = summary.pivot(index='m_target', columns=key2, values=metric).sort_index().sort_index(axis=1)
    return pv


def plot_phase2_maps(df, mode="mE"):
    summary = summarize_phase2(df, mode=mode)
    key2 = 'E' if mode == 'mE' else 'T'
    xlab = 'E (V/nm)' if mode == 'mE' else 'T (K)'

    metrics = [
        ('formed_prob', 'Formation probability'),
        ('log10_t_set_mean', 'mean log10(t_set / s)'),
        ('tortuosity_nm_mean', 'mean tortuosity (nm)'),
        ('branches_mean', 'mean branches'),
        ('Delta_mean', 'mean Δ = σE / kBT'),
        ('rate_var_log_mean', 'mean var(log10 rate)'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        pv = _pivot_metric(summary, mode, metric)
        sns.heatmap(pv, ax=ax, annot=True, fmt='.2f', cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel('m_target')
    plt.tight_layout()
    plt.show()
    return summary


def plot_phase2_slices(df, mode="mE"):
    summary = summarize_phase2(df, mode=mode)
    key2 = 'E' if mode == 'mE' else 'T'
    xlab = 'E (V/nm)' if mode == 'mE' else 'T (K)'

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    metrics = [
        ('formed_prob', 'Formation probability'),
        ('log10_t_set_mean', 'mean log10(t_set / s)'),
        ('tortuosity_nm_mean', 'mean tortuosity (nm)'),
        ('branches_mean', 'mean branches'),
        ('Delta_mean', 'mean Δ'),
        ('barrier_std_z_mean', 'mean std(em_z)'),
    ]
    mvals = sorted(summary['m_target'].unique())
    palette = sns.color_palette('tab10', n_colors=len(mvals))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for c, m0 in zip(palette, mvals):
            sub = summary[summary['m_target'] == m0].sort_values(key2)
            ax.plot(sub[key2], sub[metric], marker='o', label=f'm={m0:.2f}', color=c)
        ax.set_title(title)
        ax.set_xlabel(xlab)
        _no_offset(ax)
    axes[0,0].legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()
    return summary


def plot_phase2_regime_map(df, mode="mE"):
    summary = summarize_phase2(df, mode=mode)
    key2 = 'E' if mode == 'mE' else 'T'
    xlab = 'E (V/nm)' if mode == 'mE' else 'T (K)'
    regime_order = ['HRS-like', 'critical', 'single-path LRS', 'multi-path LRS']
    regime_code = {name: i for i, name in enumerate(regime_order)}
    temp = summary.copy()
    temp['regime_code'] = temp['regime'].map(regime_code)
    pv = temp.pivot(index='m_target', columns=key2, values='regime_code').sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(pv, ax=ax, annot=False, cmap='Set2', cbar=True,
                cbar_kws={'ticks': list(regime_code.values())})
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(regime_order)
    ax.set_title(f'Regime map ({mode})')
    ax.set_xlabel(xlab)
    ax.set_ylabel('m_target')
    plt.tight_layout()
    plt.show()
    return summary


def export_phase2_results(df, mode="mE", save_dir=SAVE_DIR, tag=PHASE2_SAVE_TAG):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    summary = summarize_phase2(df, mode=mode)
    raw_csv = save_dir / f'{tag}_{mode}_raw.csv'
    sum_csv = save_dir / f'{tag}_{mode}_summary.csv'
    df.to_csv(raw_csv, index=False)
    summary.to_csv(sum_csv, index=False)
    print('saved:', raw_csv)
    print('saved:', sum_csv)
    return raw_csv, sum_csv


def run_phase2_demo(mode="mE", m0_target=0.75, E_vnm=0.09, T=700.0, structure_seed=2025, mc_seed=0):
    out = simulate_one_phase2(m0_target=m0_target, E_vnm=E_vnm, T=T,
                              structure_seed=structure_seed, mc_seed=mc_seed,
                              return_snapshots=True, store_rate_samples=True)
    print({k: out[k] for k in ['m_target','m_actual','sigma_E','lc','Delta','E','T','formed','t_set','neck_nm','tortuosity_nm','branches']})
    plot_barrier_bundle({'eta': out['eta']}, out['em_x'], out['em_z'], out['occ0'], title_prefix=f"Phase-2 demo m={m0_target:.2f}, E={E_vnm:.3f}, T={T:.0f}")

    snaps = out['snapshots']
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, key in zip(axes, ['HRS0', 'Critical', 'LRS']):
        ax.imshow(snaps[key].T, origin='lower', aspect='auto', interpolation='nearest')
        ax.set_title(key)
        ax.set_xlabel('z cell'); ax.set_ylabel('x cell')
    plt.tight_layout()
    plt.show()
    return out




# ==================================================
# 11) Dynamic-m exploration branch (Phase-2.5)
# ==================================================
# 目标：在不破坏你前面的静态 m0 主线的前提下，增加一个“自动更新 m(t)”的探索版本。
# 推荐先用 sigma_only：只更新 sigma_E(t)，固定随机场骨架 eta0(r)。

DYN_ENABLE_DEFAULT = True
DYN_M_COUNT_MODE = "ion_plus_fil"   # "ion_only" | "ion_plus_fil"
DYN_UPDATE_EVENTS = 100 if FAST_MODE else 250
DYN_ALPHA_SIGMA = 0.10
DYN_ALPHA_LC = 0.10
DYN_UPDATE_MODE = "sigma_only"      # "sigma_only" | "sigma_lc" | "sigma_lc_eta_mix"
DYN_ETA_MIX_GAMMA = 0.15             # 仅在 sigma_lc_eta_mix 下使用
DYN_TRACE_MAX = 500

# quick compare controls
DYN_COMPARE_MODE = "mE"  # "m" | "mE" | "mT"
DYN_COMPARE_M_LIST = [0.00, 0.25, 0.50, 0.75, 0.90]
DYN_COMPARE_E_LIST = [0.060, 0.075, 0.090, 0.105]
DYN_COMPARE_T_LIST = [500.0, 600.0, 700.0, 800.0, 900.0]
DYN_COMPARE_E_FIXED = PHASE2_E_FIXED
DYN_COMPARE_T_FIXED = PHASE2_T_FIXED
DYN_COMPARE_M_FIXED = 0.75
DYN_COMPARE_N_STRUCTURE_SEEDS = 2 if FAST_MODE else 6
DYN_COMPARE_N_MC_PER_STRUCTURE = 2 if FAST_MODE else 4
DYN_COMPARE_N_JOBS = 1 if FAST_MODE else 8


def compute_m_from_occ_mode(occ, count_mode=DYN_M_COUNT_MODE, A_mask=A_MASK, B_mask=B_MASK):
    """
    动态版 m(t) 的统计方式：
    - ion_only: 只把 S_ION 当作“缺陷占据”
    - ion_plus_fil: 把 S_ION + S_FIL 都当作“缺陷/占据”
      更适合先看动态反馈，因为成丝会显著重排占据图样。
    """
    if count_mode == "ion_only":
        occupied = (occ == S_ION)
    elif count_mode == "ion_plus_fil":
        occupied = (occ == S_ION) | (occ == S_FIL)
    else:
        raise ValueError(f"unknown count_mode: {count_mode}")

    occA = np.count_nonzero(occupied & A_mask)
    occB = np.count_nonzero(occupied & B_mask)
    nA = np.count_nonzero(A_mask)
    nB = np.count_nonzero(B_mask)
    cA = occA / max(nA, 1)
    cB = occB / max(nB, 1)
    m = (cA - cB) / max(cA + cB, 1e-12)
    return cA, cB, float(m), float(abs(m)), int(occA), int(occB), int(nA), int(nB)


def _normalize_eta(eta):
    eta = eta - np.mean(eta)
    std = np.std(eta)
    if std < 1e-12:
        return np.zeros_like(eta)
    return eta / std


def build_barrier_maps_from_sigma_eta(sigma_E, eta, mu_x_site, mu_z_site):
    em_x = mu_x_site + sigma_E * eta
    em_z = mu_z_site + sigma_E * eta
    em_x[:, Z_BOT] = BASE_BARRIER_EV; em_x[:, Z_TOP] = BASE_BARRIER_EV
    em_z[:, Z_BOT] = BASE_BARRIER_EV; em_z[:, Z_TOP] = BASE_BARRIER_EV
    em_x = np.maximum(em_x, MIN_BARRIER_EV).astype(np.float64)
    em_z = np.maximum(em_z, MIN_BARRIER_EV).astype(np.float64)
    return em_x, em_z


def _run_kmc_dynamic_m(
    occ0, mu_x_site, mu_z_site, eta0,
    sigma_init, lc_init,
    E_avg_vnm, T,
    max_steps,
    do_growth, do_dissolve,
    stop_if_connected,
    stop_if_disconnected,
    seed,
    rate_sample_max,
    rate_sample_stride,
    mask_update_stride,
    alpha_field,
    min_barrier,
    field_enh_mig,
    field_enh_grow,
    tip_focus_power,
    tip_focus_max,
    enable_tip_focus,
    e_grow0,
    gamma_grow,
    delta_attach,
    hop_slow_near_tip,
    tip_behind,
    tip_ahead,
    e_diss0,
    gamma_diss,
    phib_bottom,
    phib_top,
    e_inject0,
    focus_kappa,
    e_lat_noise_vnm,
    grow_pref_forward,
    grow_penalty_lateral,
    cell_z_nm,
    enable_center_bias,
    center_bias_strength,
    center_bias_sigma_cells,
    # dynamic controls
    dynamic_enable=True,
    dynamic_count_mode=DYN_M_COUNT_MODE,
    dynamic_update_events=DYN_UPDATE_EVENTS,
    dynamic_alpha_sigma=DYN_ALPHA_SIGMA,
    dynamic_alpha_lc=DYN_ALPHA_LC,
    dynamic_update_mode=DYN_UPDATE_MODE,
    dynamic_eta_mix_gamma=DYN_ETA_MIX_GAMMA,
    dynamic_trace_max=DYN_TRACE_MAX,
):
    np.random.seed(seed)
    rng_eta = np.random.default_rng(seed + 1357911)

    occ = occ0.copy()
    w, h = occ.shape

    sigma_current = float(sigma_init)
    lc_current = float(lc_init)
    eta_current = eta0.copy().astype(float)
    em_x, em_z = build_barrier_maps_from_sigma_eta(sigma_current, eta_current, mu_x_site, mu_z_site)

    t = 0.0
    t_connect = np.nan
    last_diss_x = -1
    last_diss_z = -1

    n_hopx = 0
    n_hopz = 0
    n_grow = 0
    n_diss = 0
    n_inj  = 0
    n_dyn_updates = 0

    rates_s = np.zeros(rate_sample_max, dtype=np.float64)
    ns = 0

    max_events = w * (h-2) * 7 + w * 2
    rates = np.empty(max_events, dtype=np.float64)
    etype = np.empty(max_events, dtype=np.int8)
    ex = np.empty(max_events, dtype=np.int16)
    ez = np.empty(max_events, dtype=np.int16)
    enx = np.empty(max_events, dtype=np.int16)
    enz = np.empty(max_events, dtype=np.int16)

    mask = _bfs_bottom_connected_mask(occ)
    connected = _connected_to_top(occ, mask)
    tip_z, tip_x = _tip_stats_from_filament(occ, mask)

    trace_step = np.full(dynamic_trace_max, -1, dtype=np.int32)
    trace_time = np.full(dynamic_trace_max, np.nan, dtype=np.float64)
    trace_m = np.full(dynamic_trace_max, np.nan, dtype=np.float64)
    trace_sigma = np.full(dynamic_trace_max, np.nan, dtype=np.float64)
    trace_lc = np.full(dynamic_trace_max, np.nan, dtype=np.float64)
    trace_cA = np.full(dynamic_trace_max, np.nan, dtype=np.float64)
    trace_cB = np.full(dynamic_trace_max, np.nan, dtype=np.float64)
    trace_ns = 0

    def _record_trace(step_now, t_now, occ_now, sigma_now, lc_now):
        nonlocal trace_ns
        if trace_ns >= dynamic_trace_max:
            return
        cA, cB, m_now, m_abs_now, *_ = compute_m_from_occ_mode(occ_now, count_mode=dynamic_count_mode)
        trace_step[trace_ns] = int(step_now)
        trace_time[trace_ns] = float(t_now)
        trace_m[trace_ns] = float(m_now)
        trace_sigma[trace_ns] = float(sigma_now)
        trace_lc[trace_ns] = float(lc_now)
        trace_cA[trace_ns] = float(cA)
        trace_cB[trace_ns] = float(cB)
        trace_ns += 1

    _record_trace(0, 0.0, occ, sigma_current, lc_current)

    for step in range(max_steps):

        if step % mask_update_stride == 0:
            mask = _bfs_bottom_connected_mask(occ)
            connected = _connected_to_top(occ, mask)
            tip_z, tip_x = _tip_stats_from_filament(occ, mask)

            if connected and np.isnan(t_connect):
                t_connect = t
                if stop_if_connected:
                    energy_proxy_ev = abs(E_avg_vnm) * cell_z_nm * n_hopz
                    return dict(
                        occ=occ, t_connect=t_connect, connected=True,
                        rates_s=rates_s, ns=ns,
                        last_diss_x=last_diss_x, last_diss_z=last_diss_z,
                        n_hopx=n_hopx, n_hopz=n_hopz, n_grow=n_grow, n_diss=n_diss, n_inj=n_inj,
                        energy_proxy_ev=energy_proxy_ev,
                        sigma_final=float(sigma_current), lc_final=float(lc_current),
                        eta_final=eta_current.copy(), em_x_final=em_x.copy(), em_z_final=em_z.copy(),
                        trace_step=trace_step[:trace_ns].copy(), trace_time=trace_time[:trace_ns].copy(),
                        trace_m=trace_m[:trace_ns].copy(), trace_sigma=trace_sigma[:trace_ns].copy(),
                        trace_lc=trace_lc[:trace_ns].copy(), trace_cA=trace_cA[:trace_ns].copy(), trace_cB=trace_cB[:trace_ns].copy(),
                        n_dyn_updates=int(n_dyn_updates)
                    )

            if (not connected) and (not np.isnan(t_connect)) and stop_if_disconnected:
                energy_proxy_ev = abs(E_avg_vnm) * cell_z_nm * n_hopz
                return dict(
                    occ=occ, t_connect=t_connect, connected=False,
                    rates_s=rates_s, ns=ns,
                    last_diss_x=last_diss_x, last_diss_z=last_diss_z,
                    n_hopx=n_hopx, n_hopz=n_hopz, n_grow=n_grow, n_diss=n_diss, n_inj=n_inj,
                    energy_proxy_ev=energy_proxy_ev,
                    sigma_final=float(sigma_current), lc_final=float(lc_current),
                    eta_final=eta_current.copy(), em_x_final=em_x.copy(), em_z_final=em_z.copy(),
                    trace_step=trace_step[:trace_ns].copy(), trace_time=trace_time[:trace_ns].copy(),
                    trace_m=trace_m[:trace_ns].copy(), trace_sigma=trace_sigma[:trace_ns].copy(),
                    trace_lc=trace_lc[:trace_ns].copy(), trace_cA=trace_cA[:trace_ns].copy(), trace_cB=trace_cB[:trace_ns].copy(),
                    n_dyn_updates=int(n_dyn_updates)
                )

        gap = (h - 1) - tip_z
        E_mig = abs(E_avg_vnm) * field_enh_mig
        E_grow_base = abs(E_avg_vnm) * field_enh_grow
        E_tip = _tip_field(E_grow_base, gap, tip_focus_power, tip_focus_max, enable_tip_focus)

        Ex_noise = e_lat_noise_vnm * (np.random.randn())

        n_evt = 0
        total_rate = 0.0

        if E_avg_vnm > 0.0:
            z = 1
            for x in range(w):
                if occ[x, z] == S_EMPTY:
                    Eb_inj = phib_bottom + e_inject0 - 0.15 * E_mig
                    Eb_inj = _clamp(Eb_inj, min_barrier)
                    r = NU0 * np.exp(-Eb_inj / (KB * T))
                    rates[n_evt] = r; etype[n_evt] = 3
                    ex[n_evt] = x; ez[n_evt] = z
                    enx[n_evt] = x; enz[n_evt] = z
                    total_rate += r; n_evt += 1
        elif E_avg_vnm < 0.0:
            z = h - 2
            for x in range(w):
                if occ[x, z] == S_EMPTY:
                    Eb_inj = phib_top + e_inject0 - 0.15 * E_mig
                    Eb_inj = _clamp(Eb_inj, min_barrier)
                    r = NU0 * np.exp(-Eb_inj / (KB * T))
                    rates[n_evt] = r; etype[n_evt] = 4
                    ex[n_evt] = x; ez[n_evt] = z
                    enx[n_evt] = x; enz[n_evt] = z
                    total_rate += r; n_evt += 1

        for z in range(1, h-1):
            for x in range(w):
                s = occ[x, z]

                if s == S_ION:
                    if z+1 <= h-2 and occ[x, z+1] == S_EMPTY:
                        Eb = em_z[x, z] - alpha_field * E_mig * 1.0
                        Eb = _clamp(Eb, min_barrier)
                        r = NU0 * np.exp(-Eb / (KB * T))
                        if z >= tip_z and z <= tip_z + tip_ahead:
                            na = _count_attach_neighbors_including_electrodes(occ, x, z)
                            if na > 0: r *= hop_slow_near_tip
                        rates[n_evt] = r; etype[n_evt] = 0
                        ex[n_evt] = x; ez[n_evt] = z
                        enx[n_evt] = x; enz[n_evt] = z+1
                        total_rate += r; n_evt += 1

                    if z-1 >= 1 and occ[x, z-1] == S_EMPTY:
                        Eb = em_z[x, z] - alpha_field * E_mig * (-1.0)
                        Eb = _clamp(Eb, min_barrier)
                        r = NU0 * np.exp(-Eb / (KB * T))
                        if z >= tip_z and z <= tip_z + tip_ahead:
                            na = _count_attach_neighbors_including_electrodes(occ, x, z)
                            if na > 0: r *= hop_slow_near_tip
                        rates[n_evt] = r; etype[n_evt] = 0
                        ex[n_evt] = x; ez[n_evt] = z
                        enx[n_evt] = x; enz[n_evt] = z-1
                        total_rate += r; n_evt += 1

                    if x-1 >= 0 and occ[x-1, z] == S_EMPTY:
                        Eb = em_x[x, z] - alpha_field * (Ex_noise) * (-1.0)
                        Eb = _clamp(Eb, min_barrier)
                        r = NU0 * np.exp(-Eb / (KB * T))
                        if z >= tip_z and z <= tip_z + tip_ahead:
                            na = _count_attach_neighbors_including_electrodes(occ, x, z)
                            if na > 0: r *= hop_slow_near_tip
                        rates[n_evt] = r; etype[n_evt] = 0
                        ex[n_evt] = x; ez[n_evt] = z
                        enx[n_evt] = x-1; enz[n_evt] = z
                        total_rate += r; n_evt += 1

                    if x+1 <= w-1 and occ[x+1, z] == S_EMPTY:
                        Eb = em_x[x, z] - alpha_field * (Ex_noise) * (1.0)
                        Eb = _clamp(Eb, min_barrier)
                        r = NU0 * np.exp(-Eb / (KB * T))
                        if z >= tip_z and z <= tip_z + tip_ahead:
                            na = _count_attach_neighbors_including_electrodes(occ, x, z)
                            if na > 0: r *= hop_slow_near_tip
                        rates[n_evt] = r; etype[n_evt] = 0
                        ex[n_evt] = x; ez[n_evt] = z
                        enx[n_evt] = x+1; enz[n_evt] = z
                        total_rate += r; n_evt += 1

                    if do_growth:
                        if z >= tip_z - tip_behind and z <= tip_z + tip_ahead:
                            na_fil = _count_fil_neighbors_only(occ, x, z)
                            if na_fil > 0:
                                Eb_g = e_grow0 - gamma_grow * E_tip - delta_attach * na_fil
                                Eb_g = _clamp(Eb_g, min_barrier)
                                r = NU0 * np.exp(-Eb_g / (KB * T))

                                l, rr, d, u = _neighbor_flags_fil(occ, x, z)
                                orient = 1.0
                                if d and (not l) and (not rr):
                                    orient = grow_pref_forward
                                elif (l or rr) and (not d):
                                    orient = grow_penalty_lateral
                                r *= orient

                                if focus_kappa > 0.0:
                                    dx = abs(x - tip_x)
                                    r *= 1.0 / (1.0 + focus_kappa * dx * dx)

                                r *= _center_bias_factor(x, w, enable_center_bias,
                                                        center_bias_strength, center_bias_sigma_cells)

                                rates[n_evt] = r; etype[n_evt] = 1
                                ex[n_evt] = x; ez[n_evt] = z
                                enx[n_evt] = x; enz[n_evt] = z
                                total_rate += r; n_evt += 1

                elif s == S_FIL and do_dissolve:
                    na = 0
                    if x > 0   and occ[x-1, z] == S_EMPTY: na += 1
                    if x < w-1 and occ[x+1, z] == S_EMPTY: na += 1
                    if z > 1   and occ[x, z-1] == S_EMPTY: na += 1
                    if z < h-2 and occ[x, z+1] == S_EMPTY: na += 1
                    if na > 0:
                        Eb_d = e_diss0 - gamma_diss * E_tip
                        Eb_d = _clamp(Eb_d, min_barrier)
                        r = NU0 * np.exp(-Eb_d / (KB * T))
                        rates[n_evt] = r; etype[n_evt] = 2
                        ex[n_evt] = x; ez[n_evt] = z
                        enx[n_evt] = x; enz[n_evt] = z
                        total_rate += r; n_evt += 1

        if total_rate < 1e-40 or n_evt <= 0:
            energy_proxy_ev = abs(E_avg_vnm) * cell_z_nm * n_hopz
            return dict(
                occ=occ, t_connect=t_connect, connected=connected,
                rates_s=rates_s, ns=ns,
                last_diss_x=last_diss_x, last_diss_z=last_diss_z,
                n_hopx=n_hopx, n_hopz=n_hopz, n_grow=n_grow, n_diss=n_diss, n_inj=n_inj,
                energy_proxy_ev=energy_proxy_ev,
                sigma_final=float(sigma_current), lc_final=float(lc_current),
                eta_final=eta_current.copy(), em_x_final=em_x.copy(), em_z_final=em_z.copy(),
                trace_step=trace_step[:trace_ns].copy(), trace_time=trace_time[:trace_ns].copy(),
                trace_m=trace_m[:trace_ns].copy(), trace_sigma=trace_sigma[:trace_ns].copy(),
                trace_lc=trace_lc[:trace_ns].copy(), trace_cA=trace_cA[:trace_ns].copy(), trace_cB=trace_cB[:trace_ns].copy(),
                n_dyn_updates=int(n_dyn_updates)
            )

        u = np.random.random()
        dt = -np.log(u) / total_rate
        t += dt

        rsel = np.random.random() * total_rate
        csum = 0.0
        sel = -1
        for i in range(n_evt):
            csum += rates[i]
            if csum >= rsel:
                sel = i
                break
        if sel < 0:
            continue

        if (step % rate_sample_stride == 0) and (ns < rate_sample_max):
            rates_s[ns] = rates[sel]
            ns += 1

        tp = etype[sel]
        x0 = ex[sel]; z0 = ez[sel]
        x1 = enx[sel]; z1 = enz[sel]

        if tp == 0:
            occ[x0, z0] = S_EMPTY
            occ[x1, z1] = S_ION
            if x1 != x0: n_hopx += 1
            if z1 != z0: n_hopz += 1
        elif tp == 1:
            occ[x0, z0] = S_FIL
            n_grow += 1
        elif tp == 2:
            occ[x0, z0] = S_ION
            last_diss_x = x0; last_diss_z = z0
            n_diss += 1
        elif tp == 3:
            occ[x0, z0] = S_ION
            n_inj += 1
        elif tp == 4:
            occ[x0, z0] = S_ION
            n_inj += 1

        if dynamic_enable and dynamic_update_events > 0 and ((step + 1) % dynamic_update_events == 0):
            cA_now, cB_now, m_now, m_abs_now, *_ = compute_m_from_occ_mode(occ, count_mode=dynamic_count_mode)
            sigma_target = sigma_from_m(m_abs_now)
            lc_target = lc_from_m(m_abs_now)
            sigma_current = (1.0 - dynamic_alpha_sigma) * sigma_current + dynamic_alpha_sigma * sigma_target

            if dynamic_update_mode in ["sigma_lc", "sigma_lc_eta_mix"]:
                lc_current = (1.0 - dynamic_alpha_lc) * lc_current + dynamic_alpha_lc * lc_target
            if dynamic_update_mode == "sigma_lc_eta_mix":
                eta_fresh = correlated_gaussian_field(WIDTH_CELLS, THICKNESS_CELLS, lc_current, rng_eta)
                eta_fresh[:, Z_BOT] = 0.0
                eta_fresh[:, Z_TOP] = 0.0
                eta_current = _normalize_eta((1.0 - dynamic_eta_mix_gamma) * eta_current + dynamic_eta_mix_gamma * eta_fresh)

            em_x, em_z = build_barrier_maps_from_sigma_eta(sigma_current, eta_current, mu_x_site, mu_z_site)
            _record_trace(step + 1, t, occ, sigma_current, lc_current)
            n_dyn_updates += 1

    mask = _bfs_bottom_connected_mask(occ)
    connected = _connected_to_top(occ, mask)
    energy_proxy_ev = abs(E_avg_vnm) * cell_z_nm * n_hopz
    return dict(
        occ=occ, t_connect=t_connect, connected=connected,
        rates_s=rates_s, ns=ns,
        last_diss_x=last_diss_x, last_diss_z=last_diss_z,
        n_hopx=n_hopx, n_hopz=n_hopz, n_grow=n_grow, n_diss=n_diss, n_inj=n_inj,
        energy_proxy_ev=energy_proxy_ev,
        sigma_final=float(sigma_current), lc_final=float(lc_current),
        eta_final=eta_current.copy(), em_x_final=em_x.copy(), em_z_final=em_z.copy(),
        trace_step=trace_step[:trace_ns].copy(), trace_time=trace_time[:trace_ns].copy(),
        trace_m=trace_m[:trace_ns].copy(), trace_sigma=trace_sigma[:trace_ns].copy(),
        trace_lc=trace_lc[:trace_ns].copy(), trace_cA=trace_cA[:trace_ns].copy(), trace_cB=trace_cB[:trace_ns].copy(),
        n_dyn_updates=int(n_dyn_updates)
    )


def run_set_dynamic(occ0, mu_x_site, mu_z_site, eta0, sigma_init, lc_init,
                    E_vnm, T, mc_seed,
                    dynamic_enable=DYN_ENABLE_DEFAULT,
                    dynamic_count_mode=DYN_M_COUNT_MODE,
                    dynamic_update_events=DYN_UPDATE_EVENTS,
                    dynamic_alpha_sigma=DYN_ALPHA_SIGMA,
                    dynamic_alpha_lc=DYN_ALPHA_LC,
                    dynamic_update_mode=DYN_UPDATE_MODE,
                    dynamic_eta_mix_gamma=DYN_ETA_MIX_GAMMA,
                    dynamic_trace_max=DYN_TRACE_MAX):
    return _run_kmc_dynamic_m(
        occ0, mu_x_site, mu_z_site, eta0,
        sigma_init, lc_init,
        E_vnm, T,
        MAX_STEPS_SET,
        True, False,
        STOP_IF_CONNECTED_SET, False,
        mc_seed,
        RATE_SAMPLE_MAX, RATE_SAMPLE_STRIDE,
        MASK_UPDATE_STRIDE,
        ALPHA_FIELD, MIN_BARRIER_EV,
        FIELD_ENHANCE_MIG, FIELD_ENHANCE_GROW,
        TIP_FOCUS_POWER, TIP_FOCUS_MAX_VNM, ENABLE_TIP_FOCUS,
        E_GROW0_EV, GAMMA_GROW, DELTA_ATTACH,
        HOP_SLOW_NEAR_TIP, TIP_BEHIND, TIP_AHEAD,
        E_DISS0_EV, GAMMA_DISS,
        PHIB_BOTTOM, PHIB_TOP, E_INJECT0_EV,
        FOCUS_KAPPA,
        E_LATERAL_NOISE_VNM,
        GROW_PREF_FORWARD,
        GROW_PENALTY_LATERAL,
        CELL_Z_NM,
        ENABLE_CENTER_BIAS,
        CENTER_BIAS_STRENGTH,
        CENTER_BIAS_SIGMA_CELLS,
        dynamic_enable=dynamic_enable,
        dynamic_count_mode=dynamic_count_mode,
        dynamic_update_events=dynamic_update_events,
        dynamic_alpha_sigma=dynamic_alpha_sigma,
        dynamic_alpha_lc=dynamic_alpha_lc,
        dynamic_update_mode=dynamic_update_mode,
        dynamic_eta_mix_gamma=dynamic_eta_mix_gamma,
        dynamic_trace_max=dynamic_trace_max,
    )


def simulate_one_dynamic(m0_target, E_vnm, T,
                         structure_seed=2025, mc_seed=0,
                         dynamic_enable=True,
                         dynamic_count_mode=DYN_M_COUNT_MODE,
                         dynamic_update_events=DYN_UPDATE_EVENTS,
                         dynamic_alpha_sigma=DYN_ALPHA_SIGMA,
                         dynamic_alpha_lc=DYN_ALPHA_LC,
                         dynamic_update_mode=DYN_UPDATE_MODE,
                         dynamic_eta_mix_gamma=DYN_ETA_MIX_GAMMA,
                         return_snapshots=False, store_rate_samples=False):
    occ0 = init_occ_from_m(m0_target, seed=structure_seed)
    cA0, cB0, m_actual0, m_abs0, occA0, occB0, nA, nB = compute_m_from_occ_mode(occ0, count_mode=dynamic_count_mode)

    em_x0, em_z0, meta0 = build_barrier_maps_from_m(m_abs0, barrier_seed=structure_seed)
    sigma_init = float(meta0['sigma_E'])
    lc_init = float(meta0['lc'])
    eta0 = meta0['eta'].copy()
    mu_x_site = meta0['mu_x_site'].copy()
    mu_z_site = meta0['mu_z_site'].copy()

    snaps = {'HRS0': occ0.copy()} if return_snapshots else None

    dyn = run_set_dynamic(
        occ0, mu_x_site, mu_z_site, eta0, sigma_init, lc_init,
        E_vnm, T, mc_seed,
        dynamic_enable=dynamic_enable,
        dynamic_count_mode=dynamic_count_mode,
        dynamic_update_events=dynamic_update_events,
        dynamic_alpha_sigma=dynamic_alpha_sigma,
        dynamic_alpha_lc=dynamic_alpha_lc,
        dynamic_update_mode=dynamic_update_mode,
        dynamic_eta_mix_gamma=dynamic_eta_mix_gamma,
        dynamic_trace_max=DYN_TRACE_MAX,
    )

    occ_crit = dyn['occ']
    t_set = dyn['t_connect']
    formed = bool(dyn['connected'])
    if return_snapshots:
        snaps['Critical'] = occ_crit.copy()

    # post-thicken: 用动态 SET 结束时的 barrier field 继续做静态厚化，减少额外数值干扰
    occ_lrs = occ_crit
    if formed and ENABLE_POST_THICKEN:
        occ_lrs, *_ = run_post_thicken(occ_crit, dyn['em_x_final'], dyn['em_z_final'], E_vnm, T, mc_seed + 77777)
    if return_snapshots:
        snaps['LRS'] = occ_lrs.copy()

    cA1, cB1, m_actual1, m_abs1, occA1, occB1, *_ = compute_m_from_occ_mode(occ_lrs, count_mode=dynamic_count_mode)
    meander_rms_nm, neck_nm, branches, comx_nm, drift_flag, tort_nm = morphology_from_occ(occ_lrs)
    barrier_std_x = float(np.std(dyn['em_x_final'][:, Z_OX_START:Z_OX_END+1]))
    barrier_std_z = float(np.std(dyn['em_z_final'][:, Z_OX_START:Z_OX_END+1]))

    out = dict(
        dynamic=int(dynamic_enable),
        dynamic_count_mode=str(dynamic_count_mode),
        dynamic_update_events=int(dynamic_update_events),
        dynamic_alpha_sigma=float(dynamic_alpha_sigma),
        dynamic_alpha_lc=float(dynamic_alpha_lc),
        dynamic_update_mode=str(dynamic_update_mode),
        dynamic_eta_mix_gamma=float(dynamic_eta_mix_gamma),
        n_dyn_updates=int(dyn['n_dyn_updates']),
        m_target=float(m0_target),
        m_actual=float(m_actual0),
        m_abs=float(m_abs0),
        m_final=float(m_actual1),
        m_final_abs=float(m_abs1),
        cA=float(cA0), cB=float(cB0),
        cA_final=float(cA1), cB_final=float(cB1),
        occA_init=int(occA0), occB_init=int(occB0),
        occA_final=int(occA1), occB_final=int(occB1),
        nA_sites=int(nA), nB_sites=int(nB),
        sigma_E=float(sigma_init),
        sigma_final=float(dyn['sigma_final']),
        lc=float(lc_init),
        lc_final=float(dyn['lc_final']),
        Delta=float(delta_from_sigmaT(sigma_init, T)),
        Delta_final=float(delta_from_sigmaT(dyn['sigma_final'], T)),
        barrier_std_x=barrier_std_x,
        barrier_std_z=barrier_std_z,
        structure_seed=int(structure_seed),
        mc_seed=int(mc_seed),
        E=float(E_vnm), T=float(T),
        formed=int(formed),
        t_set=float(t_set),
        energy_set_ev=float(dyn['energy_proxy_ev']),
        n_hopx=int(dyn['n_hopx']), n_hopz=int(dyn['n_hopz']),
        n_grow=int(dyn['n_grow']), n_inj=int(dyn['n_inj']),
        meander_rms_nm=float(meander_rms_nm),
        tortuosity_nm=float(tort_nm),
        neck_nm=float(neck_nm),
        branches=int(branches),
        comx_nm=float(comx_nm),
        drift=int(drift_flag),
        rate_var_log=float(np.var(np.log10(dyn['rates_s'][:dyn['ns']] + 1e-40))) if dyn['ns'] > 5 else np.nan,
    )

    if store_rate_samples:
        out['rate_samples'] = dyn['rates_s'][:max(1, dyn['ns'])].copy()
    if return_snapshots:
        out['snapshots'] = snaps
        out['occ0'] = occ0.copy()
        out['em_x0'] = em_x0.copy()
        out['em_z0'] = em_z0.copy()
        out['eta0'] = eta0.copy()
        out['em_x_final'] = dyn['em_x_final'].copy()
        out['em_z_final'] = dyn['em_z_final'].copy()
        out['eta_final'] = dyn['eta_final'].copy()
        out['trace_step'] = dyn['trace_step'].copy()
        out['trace_time'] = dyn['trace_time'].copy()
        out['trace_m'] = dyn['trace_m'].copy()
        out['trace_sigma'] = dyn['trace_sigma'].copy()
        out['trace_lc'] = dyn['trace_lc'].copy()
        out['trace_cA'] = dyn['trace_cA'].copy()
        out['trace_cB'] = dyn['trace_cB'].copy()
    return out


def simulate_one_static_or_dynamic(m0_target, E_vnm, T,
                                   structure_seed=2025, mc_seed=0,
                                   dynamic=False,
                                   **dyn_kwargs):
    if dynamic:
        return simulate_one_dynamic(
            m0_target=m0_target, E_vnm=E_vnm, T=T,
            structure_seed=structure_seed, mc_seed=mc_seed,
            dynamic_enable=True,
            **dyn_kwargs,
        )
    out = simulate_one_phase1(
        m0_target=m0_target, E_vnm=E_vnm, T=T,
        structure_seed=structure_seed, mc_seed=mc_seed,
        return_snapshots=dyn_kwargs.get('return_snapshots', False),
        store_rate_samples=dyn_kwargs.get('store_rate_samples', False),
    )
    out['dynamic'] = 0
    out['dynamic_count_mode'] = dyn_kwargs.get('dynamic_count_mode', DYN_M_COUNT_MODE)
    out['dynamic_update_events'] = dyn_kwargs.get('dynamic_update_events', DYN_UPDATE_EVENTS)
    out['dynamic_alpha_sigma'] = dyn_kwargs.get('dynamic_alpha_sigma', DYN_ALPHA_SIGMA)
    out['dynamic_alpha_lc'] = dyn_kwargs.get('dynamic_alpha_lc', DYN_ALPHA_LC)
    out['dynamic_update_mode'] = dyn_kwargs.get('dynamic_update_mode', DYN_UPDATE_MODE)
    out['dynamic_eta_mix_gamma'] = dyn_kwargs.get('dynamic_eta_mix_gamma', DYN_ETA_MIX_GAMMA)
    out['n_dyn_updates'] = 0
    out['m_final'] = out['m_actual']
    out['m_final_abs'] = out['m_abs']
    out['sigma_final'] = out['sigma_E']
    out['lc_final'] = out['lc']
    out['Delta_final'] = out['Delta']
    out['cA_final'] = out['cA']
    out['cB_final'] = out['cB']
    return out


def run_dynamic_pair_demo(m0_target=0.75, E_vnm=0.09, T=700.0,
                          structure_seed=2025, mc_seed=0,
                          dynamic_count_mode=DYN_M_COUNT_MODE,
                          dynamic_update_events=DYN_UPDATE_EVENTS,
                          dynamic_alpha_sigma=DYN_ALPHA_SIGMA,
                          dynamic_alpha_lc=DYN_ALPHA_LC,
                          dynamic_update_mode=DYN_UPDATE_MODE,
                          dynamic_eta_mix_gamma=DYN_ETA_MIX_GAMMA):
    out_static = simulate_one_static_or_dynamic(
        m0_target=m0_target, E_vnm=E_vnm, T=T,
        structure_seed=structure_seed, mc_seed=mc_seed,
        dynamic=False, return_snapshots=True, store_rate_samples=True,
        dynamic_count_mode=dynamic_count_mode,
        dynamic_update_events=dynamic_update_events,
        dynamic_alpha_sigma=dynamic_alpha_sigma,
        dynamic_alpha_lc=dynamic_alpha_lc,
        dynamic_update_mode=dynamic_update_mode,
        dynamic_eta_mix_gamma=dynamic_eta_mix_gamma,
    )
    out_dynamic = simulate_one_static_or_dynamic(
        m0_target=m0_target, E_vnm=E_vnm, T=T,
        structure_seed=structure_seed, mc_seed=mc_seed,
        dynamic=True, return_snapshots=True, store_rate_samples=True,
        dynamic_count_mode=dynamic_count_mode,
        dynamic_update_events=dynamic_update_events,
        dynamic_alpha_sigma=dynamic_alpha_sigma,
        dynamic_alpha_lc=dynamic_alpha_lc,
        dynamic_update_mode=dynamic_update_mode,
        dynamic_eta_mix_gamma=dynamic_eta_mix_gamma,
    )
    return out_static, out_dynamic


def plot_dynamic_pair_demo(out_static, out_dynamic):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    im0 = axes[0,0].imshow(out_static['eta'] if 'eta' in out_static else out_static['eta0'], origin='lower', aspect='auto')
    axes[0,0].set_title('Static: initial eta')
    plt.colorbar(im0, ax=axes[0,0], fraction=0.046)

    im1 = axes[1,0].imshow(out_dynamic['eta0'], origin='lower', aspect='auto')
    axes[1,0].set_title('Dynamic: initial eta0')
    plt.colorbar(im1, ax=axes[1,0], fraction=0.046)

    im2 = axes[0,1].imshow(out_static['em_z'], origin='lower', aspect='auto')
    axes[0,1].set_title(f"Static em_z, sigma={out_static['sigma_E']:.3f} eV")
    plt.colorbar(im2, ax=axes[0,1], fraction=0.046)

    im3 = axes[1,1].imshow(out_dynamic['em_z_final'], origin='lower', aspect='auto')
    axes[1,1].set_title(f"Dynamic em_z final, sigma={out_dynamic['sigma_final']:.3f} eV")
    plt.colorbar(im3, ax=axes[1,1], fraction=0.046)

    axes[0,2].plot(out_static['snapshots']['HRS0'].sum(axis=0), label='HRS0')
    axes[0,2].plot(out_static['snapshots']['Critical'].sum(axis=0), label='Critical')
    axes[0,2].plot(out_static['snapshots']['LRS'].sum(axis=0), label='LRS')
    axes[0,2].set_title('Static column occupancy sum')
    axes[0,2].legend(fontsize=8)

    axes[1,2].plot(out_dynamic['trace_step'], out_dynamic['trace_m'], marker='o', ms=3, label='m(t)')
    ax2b = axes[1,2].twinx()
    ax2b.plot(out_dynamic['trace_step'], out_dynamic['trace_sigma'], color='tab:red', marker='s', ms=3, label='sigma(t)')
    axes[1,2].set_title('Dynamic traces')
    axes[1,2].set_xlabel('KMC step')
    axes[1,2].set_ylabel('m(t)')
    ax2b.set_ylabel('sigma_E(t) [eV]')

    axes[0,3].imshow(out_static['snapshots']['LRS'].T, origin='lower', aspect='auto')
    axes[0,3].set_title(f"Static LRS\nformed={out_static['formed']} t_set={out_static['t_set']:.2e}s")

    axes[1,3].imshow(out_dynamic['snapshots']['LRS'].T, origin='lower', aspect='auto')
    axes[1,3].set_title(f"Dynamic LRS\nformed={out_dynamic['formed']} t_set={out_dynamic['t_set']:.2e}s")

    plt.tight_layout()
    plt.show()

    comp = pd.DataFrame([
        {"branch": "static", "formed": out_static['formed'], "t_set": out_static['t_set'],
         "m_init": out_static['m_actual'], "m_final": out_static['m_final'],
         "sigma_init": out_static['sigma_E'], "sigma_final": out_static['sigma_final'],
         "branches": out_static['branches'], "tortuosity_nm": out_static['tortuosity_nm'],
         "neck_nm": out_static['neck_nm']},
        {"branch": "dynamic", "formed": out_dynamic['formed'], "t_set": out_dynamic['t_set'],
         "m_init": out_dynamic['m_actual'], "m_final": out_dynamic['m_final'],
         "sigma_init": out_dynamic['sigma_E'], "sigma_final": out_dynamic['sigma_final'],
         "branches": out_dynamic['branches'], "tortuosity_nm": out_dynamic['tortuosity_nm'],
         "neck_nm": out_dynamic['neck_nm']},
    ])
    return comp


def run_dynamic_compare_grid(mode=DYN_COMPARE_MODE,
                             m_list=DYN_COMPARE_M_LIST,
                             E_list=DYN_COMPARE_E_LIST,
                             T_list=DYN_COMPARE_T_LIST,
                             m_fixed=DYN_COMPARE_M_FIXED,
                             E_fixed=DYN_COMPARE_E_FIXED,
                             T_fixed=DYN_COMPARE_T_FIXED,
                             n_structure_seeds=DYN_COMPARE_N_STRUCTURE_SEEDS,
                             n_mc_per_structure=DYN_COMPARE_N_MC_PER_STRUCTURE,
                             n_jobs=DYN_COMPARE_N_JOBS,
                             dynamic_count_mode=DYN_M_COUNT_MODE,
                             dynamic_update_events=DYN_UPDATE_EVENTS,
                             dynamic_alpha_sigma=DYN_ALPHA_SIGMA,
                             dynamic_alpha_lc=DYN_ALPHA_LC,
                             dynamic_update_mode=DYN_UPDATE_MODE,
                             dynamic_eta_mix_gamma=DYN_ETA_MIX_GAMMA):
    assert mode in ["m", "mE", "mT"]
    tasks = []
    dyn_flags = [0, 1]
    if mode == "m":
        for dyn in dyn_flags:
            for m0 in m_list:
                for sdev in range(n_structure_seeds):
                    structure_seed = 3301 + 1000*sdev + int(round(m0*100)) + 19*dyn
                    for smc in range(n_mc_per_structure):
                        mc_seed = 10_000*structure_seed + smc
                        tasks.append((dyn, float(m0), float(E_fixed), float(T_fixed), int(structure_seed), int(mc_seed)))
    elif mode == "mE":
        for dyn in dyn_flags:
            for m0 in m_list:
                for E0 in E_list:
                    for sdev in range(n_structure_seeds):
                        structure_seed = 4401 + 1000*sdev + int(round(m0*100)) + int(round(E0*1000)) + 23*dyn
                        for smc in range(n_mc_per_structure):
                            mc_seed = 10_000*structure_seed + smc
                            tasks.append((dyn, float(m0), float(E0), float(T_fixed), int(structure_seed), int(mc_seed)))
    else:
        for dyn in dyn_flags:
            for m0 in [m_fixed]:
                for T0 in T_list:
                    for sdev in range(n_structure_seeds):
                        structure_seed = 5501 + 1000*sdev + int(round(m0*100)) + int(round(T0)) + 29*dyn
                        for smc in range(n_mc_per_structure):
                            mc_seed = 10_000*structure_seed + smc
                            tasks.append((dyn, float(m0), float(E_fixed), float(T0), int(structure_seed), int(mc_seed)))

    def _task(dyn, m0, E0, T0, structure_seed, mc_seed):
        return simulate_one_static_or_dynamic(
            m0_target=m0, E_vnm=E0, T=T0,
            structure_seed=structure_seed, mc_seed=mc_seed,
            dynamic=bool(dyn),
            dynamic_count_mode=dynamic_count_mode,
            dynamic_update_events=dynamic_update_events,
            dynamic_alpha_sigma=dynamic_alpha_sigma,
            dynamic_alpha_lc=dynamic_alpha_lc,
            dynamic_update_mode=dynamic_update_mode,
            dynamic_eta_mix_gamma=dynamic_eta_mix_gamma,
            return_snapshots=False, store_rate_samples=False,
        )

    t0 = time.time()
    if n_jobs == 1:
        rows = [_task(*t) for t in tasks]
    else:
        rows = Parallel(n_jobs=n_jobs, prefer='threads', batch_size=1)(delayed(_task)(*t) for t in tasks)
    df = pd.DataFrame(rows)
    df['branch'] = df['dynamic'].map({0: 'static', 1: 'dynamic'})
    df['log10_t_set'] = np.log10(df['t_set'].clip(lower=1e-30))
    df['m_target_label'] = df['m_target'].map(lambda v: f"{v:.2f}")
    df['E_label'] = df['E'].map(lambda v: f"{v:.3f}")
    df['T_label'] = df['T'].map(lambda v: f"{v:.0f}")
    print(f"Dynamic-compare {mode} done: {len(df)} runs in {time.time()-t0:.1f} s")
    return df


def summarize_dynamic_compare(df, mode='m'):
    grp_keys = ['branch']
    if mode == 'm':
        grp_keys += ['m_target']
    elif mode == 'mE':
        grp_keys += ['m_target', 'E']
    else:
        grp_keys += ['T']
    return df.groupby(grp_keys, dropna=False).agg(
        formed_prob=('formed', 'mean'),
        log10_t_set_mean=('log10_t_set', lambda s: np.nanmean(s[df.loc[s.index, 'formed'] == 1]) if np.any(df.loc[s.index, 'formed'] == 1) else np.nan),
        log10_t_set_std=('log10_t_set', lambda s: np.nanstd(s[df.loc[s.index, 'formed'] == 1]) if np.any(df.loc[s.index, 'formed'] == 1) else np.nan),
        branches_mean=('branches', 'mean'),
        tortuosity_nm_mean=('tortuosity_nm', 'mean'),
        neck_nm_mean=('neck_nm', 'mean'),
        m_final_mean=('m_final', 'mean'),
        sigma_final_mean=('sigma_final', 'mean'),
        Delta_final_mean=('Delta_final', 'mean'),
        n_dyn_updates_mean=('n_dyn_updates', 'mean'),
    ).reset_index()


def plot_dynamic_compare(df, mode='m'):
    summ = summarize_dynamic_compare(df, mode=mode)

    if mode == 'm':
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        metrics = [
            ('formed_prob', 'Formation probability'),
            ('log10_t_set_mean', 'mean log10(t_set)'),
            ('branches_mean', 'branches'),
            ('tortuosity_nm_mean', 'tortuosity [nm]'),
            ('m_final_mean', 'm_final'),
            ('sigma_final_mean', 'sigma_final [eV]'),
        ]
        for ax, (col, title) in zip(axes.flat, metrics):
            sns.lineplot(data=summ, x='m_target', y=col, hue='branch', marker='o', ax=ax)
            ax.set_title(title)
        plt.tight_layout()
        plt.show()
        display(summ)
        return summ

    # mE or mT: side-by-side static/dynamic heatmaps + delta
    if mode == 'mE':
        xcol, ycol = 'E', 'm_target'
        xlabel, ylabel = 'E (V/nm)', 'm_target'
    else:
        xcol, ycol = 'T', 'branch'
        xlabel, ylabel = 'T (K)', 'branch'

    metric_pairs = [
        ('formed_prob', 'Formation probability'),
        ('log10_t_set_mean', 'mean log10(t_set)'),
        ('branches_mean', 'branches_mean'),
        ('tortuosity_nm_mean', 'tortuosity_nm_mean'),
    ]

    for metric, title in metric_pairs:
        if mode == 'mE':
            p_static = summ[summ['branch'] == 'static'].pivot(index='m_target', columns='E', values=metric)
            p_dynamic = summ[summ['branch'] == 'dynamic'].pivot(index='m_target', columns='E', values=metric)
            p_delta = p_dynamic - p_static
            fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
            sns.heatmap(p_static, annot=True, fmt='.2f', cmap='viridis', ax=axes[0])
            axes[0].set_title(f'Static: {title}')
            sns.heatmap(p_dynamic, annot=True, fmt='.2f', cmap='viridis', ax=axes[1])
            axes[1].set_title(f'Dynamic: {title}')
            sns.heatmap(p_delta, annot=True, fmt='.2f', cmap='coolwarm', center=0.0, ax=axes[2])
            axes[2].set_title(f'Dynamic - Static: {title}')
            for ax in axes:
                ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            plt.tight_layout(); plt.show()
        else:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.lineplot(data=summ, x='T', y=metric, hue='branch', marker='o', ax=ax)
            ax.set_title(title)
            ax.set_xlabel('T (K)')
            plt.tight_layout(); plt.show()
    display(summ)
    return summ


def export_dynamic_compare_results(df, mode='m', tag='dynamic_compare'):
    out_dir = SAVE_DIR / 'dynamic_compare'
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / f'{tag}_{mode}_raw.csv'
    summ = out_dir / f'{tag}_{mode}_summary.csv'
    df.to_csv(raw, index=False)
    summarize_dynamic_compare(df, mode=mode).to_csv(summ, index=False)
    print(f'Saved raw -> {raw}')
    print(f'Saved summary -> {summ}')


# ===== dynamic validation around static descriptor Psi ≈ (E/E50)/Delta =====
DYN_VALIDATION_POINTS = [
    {"label": "m025_sub", "m0": 0.25, "E": 0.055, "T": 700.0, "E_over_E50_target": 0.90},
    {"label": "m025_thr", "m0": 0.25, "E": 0.060, "T": 700.0, "E_over_E50_target": 0.98},
    {"label": "m025_sup", "m0": 0.25, "E": 0.070, "T": 700.0, "E_over_E50_target": 1.15},
    {"label": "m090_sub", "m0": 0.90, "E": 0.085, "T": 700.0, "E_over_E50_target": 0.89},
    {"label": "m090_thr", "m0": 0.90, "E": 0.095, "T": 700.0, "E_over_E50_target": 1.00},
    {"label": "m090_sup", "m0": 0.90, "E": 0.105, "T": 700.0, "E_over_E50_target": 1.10},
]

DYN_VALIDATION_N_STRUCTURE_SEEDS = 2 if FAST_MODE else 6
DYN_VALIDATION_N_MC_PER_STRUCTURE = 2 if FAST_MODE else 4
DYN_VALIDATION_N_JOBS = 1 if FAST_MODE else 6

def _attach_stage_metrics(out):
    """
    从 snapshots 中补充 first-percolation(Critical) 和 LRS 的形貌指标。
    这个函数只在小规模 validation 里用，因此允许 return_snapshots=True。
    """
    if "snapshots" not in out:
        return out

    if "Critical" in out["snapshots"]:
        occ_crit = out["snapshots"]["Critical"]
        meander_rms_nm_crit, neck_nm_crit, branches_crit, comx_nm_crit, drift_crit, tortuosity_nm_crit = morphology_from_occ(occ_crit)
        out["meander_rms_nm_crit"] = float(meander_rms_nm_crit)
        out["neck_nm_crit"] = float(neck_nm_crit)
        out["branches_crit"] = int(branches_crit)
        out["comx_nm_crit"] = float(comx_nm_crit)
        out["drift_crit"] = int(drift_crit)
        out["tortuosity_nm_crit"] = float(tortuosity_nm_crit)
    else:
        out["meander_rms_nm_crit"] = np.nan
        out["neck_nm_crit"] = np.nan
        out["branches_crit"] = np.nan
        out["comx_nm_crit"] = np.nan
        out["drift_crit"] = np.nan
        out["tortuosity_nm_crit"] = np.nan

    if "LRS" in out["snapshots"]:
        occ_lrs = out["snapshots"]["LRS"]
        meander_rms_nm_lrs, neck_nm_lrs, branches_lrs, comx_nm_lrs, drift_lrs, tortuosity_nm_lrs = morphology_from_occ(occ_lrs)
        out["meander_rms_nm_lrs"] = float(meander_rms_nm_lrs)
        out["neck_nm_lrs"] = float(neck_nm_lrs)
        out["branches_lrs"] = int(branches_lrs)
        out["comx_nm_lrs"] = float(comx_nm_lrs)
        out["drift_lrs"] = int(drift_lrs)
        out["tortuosity_nm_lrs"] = float(tortuosity_nm_lrs)
    else:
        out["meander_rms_nm_lrs"] = np.nan
        out["neck_nm_lrs"] = np.nan
        out["branches_lrs"] = np.nan
        out["comx_nm_lrs"] = np.nan
        out["drift_lrs"] = np.nan
        out["tortuosity_nm_lrs"] = np.nan
    return out

def run_dynamic_validation_points(
    point_list=DYN_VALIDATION_POINTS,
    n_structure_seeds=DYN_VALIDATION_N_STRUCTURE_SEEDS,
    n_mc_per_structure=DYN_VALIDATION_N_MC_PER_STRUCTURE,
    n_jobs=DYN_VALIDATION_N_JOBS,
    dynamic_count_mode=DYN_M_COUNT_MODE,
    dynamic_update_events=DYN_UPDATE_EVENTS,
    dynamic_alpha_sigma=DYN_ALPHA_SIGMA,
    dynamic_alpha_lc=DYN_ALPHA_LC,
    dynamic_update_mode=DYN_UPDATE_MODE,
    dynamic_eta_mix_gamma=DYN_ETA_MIX_GAMMA,
):
    """
    只跑少量代表点：
    - 低 m / 高 m
    - E/E50 < 1, ≈1, >1
    对每个点比较 static vs dynamic。
    """
    tasks = []
    for ip, pt in enumerate(point_list):
        for dyn in [0, 1]:
            for sdev in range(n_structure_seeds):
                structure_seed = 8101 + 5000*ip + 1000*sdev + 37*dyn
                for smc in range(n_mc_per_structure):
                    mc_seed = 10000*structure_seed + smc
                    tasks.append((ip, pt, dyn, structure_seed, mc_seed))

    def _task(ip, pt, dyn, structure_seed, mc_seed):
        out = simulate_one_static_or_dynamic(
            m0_target=float(pt["m0"]),
            E_vnm=float(pt["E"]),
            T=float(pt["T"]),
            structure_seed=int(structure_seed),
            mc_seed=int(mc_seed),
            dynamic=bool(dyn),
            return_snapshots=True,
            store_rate_samples=False,
            dynamic_count_mode=dynamic_count_mode,
            dynamic_update_events=dynamic_update_events,
            dynamic_alpha_sigma=dynamic_alpha_sigma,
            dynamic_alpha_lc=dynamic_alpha_lc,
            dynamic_update_mode=dynamic_update_mode,
            dynamic_eta_mix_gamma=dynamic_eta_mix_gamma,
        )
        out = _attach_stage_metrics(out)
        out["point_id"] = int(ip)
        out["label"] = str(pt["label"])
        out["E_over_E50_target"] = float(pt.get("E_over_E50_target", np.nan))
        out["branch"] = "dynamic" if int(dyn) == 1 else "static"
        return out

    t0 = time.time()
    if n_jobs == 1:
        rows = [_task(*t) for t in tasks]
    else:
        rows = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=1)(
            delayed(_task)(*t) for t in tasks
        )
    df = pd.DataFrame(rows)
    print(f"Dynamic validation done: {len(df)} runs in {time.time()-t0:.1f} s")
    return df

def summarize_dynamic_validation(df):
    grp = ['label', 'point_id', 'm_target', 'E', 'T', 'E_over_E50_target', 'branch']
    def _mean_if_formed(s, base):
        formed_mask = df.loc[s.index, 'formed'] == 1
        vals = df.loc[s.index, base][formed_mask]
        return float(np.nanmean(vals)) if np.any(formed_mask) else np.nan
    def _median_if_formed(s, base):
        formed_mask = df.loc[s.index, 'formed'] == 1
        vals = df.loc[s.index, base][formed_mask]
        return float(np.nanmedian(vals)) if np.any(formed_mask) else np.nan
    out = df.groupby(grp, dropna=False).agg(
        runs=('formed', 'size'),
        formed_prob=('formed', 'mean'),
        log10_t_set_mean_formed=('log10_t_set', lambda s: _mean_if_formed(s, 'log10_t_set')),
        log10_t_set_median_formed=('log10_t_set', lambda s: _median_if_formed(s, 'log10_t_set')),
        branches_crit_mean_formed=('branches_crit', lambda s: _mean_if_formed(s, 'branches_crit')),
        branches_crit_median_formed=('branches_crit', lambda s: _median_if_formed(s, 'branches_crit')),
        tortuosity_nm_crit_mean_formed=('tortuosity_nm_crit', lambda s: _mean_if_formed(s, 'tortuosity_nm_crit')),
        tortuosity_nm_crit_median_formed=('tortuosity_nm_crit', lambda s: _median_if_formed(s, 'tortuosity_nm_crit')),
        branches_lrs_mean_formed=('branches_lrs', lambda s: _mean_if_formed(s, 'branches_lrs')),
        branches_lrs_median_formed=('branches_lrs', lambda s: _median_if_formed(s, 'branches_lrs')),
        m_final_mean=('m_final', 'mean'),
        sigma_final_mean=('sigma_final', 'mean'),
        Delta_final_mean=('Delta_final', 'mean'),
        n_dyn_updates_mean=('n_dyn_updates', 'mean'),
    ).reset_index()
    return out

def plot_dynamic_validation(df):
    summ = summarize_dynamic_validation(df)
    display(summ)

    # 1) formation / time
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    sns.pointplot(data=summ, x='label', y='formed_prob', hue='branch', ax=axes[0,0], dodge=0.35)
    axes[0,0].set_title('Formation probability')
    axes[0,0].tick_params(axis='x', rotation=30)

    sns.pointplot(data=summ, x='label', y='log10_t_set_median_formed', hue='branch', ax=axes[0,1], dodge=0.35)
    axes[0,1].set_title('median log10(t_set) among formed runs')
    axes[0,1].tick_params(axis='x', rotation=30)

    # 2) crit topology
    sns.pointplot(data=summ, x='label', y='branches_crit_median_formed', hue='branch', ax=axes[1,0], dodge=0.35)
    axes[1,0].set_title('Critical branches (median, formed-only)')
    axes[1,0].tick_params(axis='x', rotation=30)

    sns.pointplot(data=summ, x='label', y='tortuosity_nm_crit_median_formed', hue='branch', ax=axes[1,1], dodge=0.35)
    axes[1,1].set_title('Critical tortuosity (median, formed-only)')
    axes[1,1].tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.show()

    # 3) dynamic trace endpoints
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.scatterplot(data=summ, x='m_target', y='m_final_mean', hue='branch', style='label', s=80, ax=axes[0])
    axes[0].set_title('m_final vs m_target')

    sns.scatterplot(data=summ, x='E_over_E50_target', y='Delta_final_mean', hue='branch', style='label', s=80, ax=axes[1])
    axes[1].set_title('Delta_final vs E/E50(target)')
    plt.tight_layout()
    plt.show()
    return summ

def export_dynamic_validation_results(df, tag='dynamic_validation_6pt'):
    out_dir = SAVE_DIR / 'dynamic_validation'
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / f'{tag}_raw.csv'
    summ = out_dir / f'{tag}_summary.csv'
    df.to_csv(raw, index=False)
    summarize_dynamic_validation(df).to_csv(summ, index=False)
    print(f'Saved raw -> {raw}')
    print(f'Saved summary -> {summ}')
