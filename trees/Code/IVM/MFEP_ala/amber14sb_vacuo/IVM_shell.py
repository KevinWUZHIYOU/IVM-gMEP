import os
import glob
import subprocess
import numpy as np
import torch
import torch.nn as nn
import time
import sys

# ============================================================
# 绘图设置
#
# DISPLAY_FIGS=False:
#   使用非交互式 Agg 后端，不弹出图形窗口。
#
# SAVE_FIGS=True:
#   继续执行 fig_save 并保存图片。
# ============================================================
DISPLAY_FIGS = False
SAVE_FIGS = True

import matplotlib

if not DISPLAY_FIGS:
    matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ============================================================
# GROMACS / PLUMED 设置
# ============================================================
GMX_EXECUTABLE = "/usr/local/gromacs/bin/gmx"
PLUMED_KERNEL_PATH = "/Users/wuzhiyou/plumed/lib/libplumedKernel.dylib"
PLUMED_BIN = "/Users/wuzhiyou/plumed/bin/plumed"

# True：不在 Terminal 打印 GROMACS/grompp 的大量日志
# False：恢复 GROMACS 输出
GROMACS_QUIET = True

# 将 gmx 所在目录加入 PATH
GMX_BIN_DIR = os.path.dirname(GMX_EXECUTABLE)
os.environ["PATH"] = GMX_BIN_DIR + os.pathsep + os.environ.get("PATH", "")

# 外部 PLUMED kernel
os.environ["PLUMED_KERNEL"] = PLUMED_KERNEL_PATH

step = 10
deg_to_rad = np.pi / 180.0


def run_shell_command(command, quiet=None):
    """
    执行 shell 命令。

    quiet=True:
        stdout 和 stderr 均不输出到 Terminal。

    check=True:
        命令失败时仍然抛出异常，避免静默忽略模拟错误。
    """
    if quiet is None:
        quiet = GROMACS_QUIET

    kwargs = {
        "shell": True,
        "check": True,
    }

    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.STDOUT

    return subprocess.run(command, **kwargs)


# ============================================================
# 0) Helper: discard fraction -> frame count
# ============================================================
def compute_discard_n(n_total, discard_frac):
    """
    Convert discard fraction to number of frames to discard.
    """
    if discard_frac is None:
        return 0

    if not (0.0 <= discard_frac < 1.0):
        raise ValueError(
            f"discard_frac must satisfy 0 <= discard_frac < 1, "
            f"got {discard_frac}"
        )

    discard_n = int(np.floor(n_total * discard_frac))

    if discard_n >= n_total and n_total > 0:
        raise ValueError(
            f"discard_frac={discard_frac} removes all frames: "
            f"n_total={n_total}, discard_n={discard_n}"
        )

    return discard_n


# ============================================================
# 1) Run restrained simulation
# ============================================================
def restrained_simulation(grid_point):
    phi_deg, psi_deg, phi, psi = grid_point

    dir_name = f"phi_{phi_deg}_psi_{psi_deg}"
    os.makedirs(dir_name, exist_ok=True)

    # --------------------------------------------------------
    # Pre-equilibration: soft restraint
    # --------------------------------------------------------
    plumed_content_initial = f"""
MOLINFO STRUCTURE=./ala2.pdb

phi: TORSION ATOMS=@phi-2
psi: TORSION ATOMS=@psi-2
restraint: RESTRAINT ARG=phi,psi AT={phi},{psi} KAPPA=100.0,100.0

PRINT ARG=phi,psi,restraint.bias FILE={dir_name}/COLVAR_initial STRIDE=100
"""

    # --------------------------------------------------------
    # Production: stiff restraint + metric projection
    # --------------------------------------------------------
    plumed_content = f"""
MOLINFO STRUCTURE=./ala2.pdb

phi: TORSION ATOMS=@phi-2
psi: TORSION ATOMS=@psi-2
restraint: RESTRAINT ARG=phi,psi AT={phi},{psi} KAPPA=1000.0,1000.0

DUMPPROJECTIONS ARG=phi,psi FILE={dir_name}/PROJ STRIDE=20
PRINT ARG=phi,psi,restraint.bias FILE={dir_name}/COLVAR_restrained STRIDE=20
"""

    with open(
        os.path.join(dir_name, "plumed.dat"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(plumed_content)

    with open(
        os.path.join(dir_name, "plumed_initial.dat"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(plumed_content_initial)

    # quiet 时不使用 -v
    verbose_flag = "" if GROMACS_QUIET else " -v"

    # --------------------------------------------------------
    # Soft restrained stage
    # --------------------------------------------------------
    run_shell_command(
        f'"{GMX_EXECUTABLE}" mdrun '
        f"-s nvt.tpr "
        f"-plumed {dir_name}/plumed_initial.dat "
        f"-nb cpu -pme cpu -ntmpi 1 "
        f"-deffnm {dir_name}/initial"
        f"{verbose_flag}"
    )

    # --------------------------------------------------------
    # Generate TPR for stiff restrained stage
    # --------------------------------------------------------
    run_shell_command(
        f'"{GMX_EXECUTABLE}" grompp '
        f"-f nvt.mdp "
        f"-c {dir_name}/initial.gro "
        f"-r {dir_name}/initial.gro "
        f"-p topol.top "
        f"-o {dir_name}/restrained.tpr"
    )

    # --------------------------------------------------------
    # Stiff restrained production stage
    # --------------------------------------------------------
    run_shell_command(
        f'"{GMX_EXECUTABLE}" mdrun '
        f"-s {dir_name}/restrained.tpr "
        f"-plumed {dir_name}/plumed.dat "
        f"-nb cpu -pme cpu -ntmpi 1 "
        f"-deffnm {dir_name}/restrained"
        f"{verbose_flag}"
    )


# ============================================================
# 2) COLVAR plot helper
# ============================================================
def plot_colvar(filename, figname):
    colvar = np.loadtxt(filename, comments="#")

    if colvar.ndim == 1:
        colvar = colvar.reshape(1, -1)

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        colvar[:, 1] / deg_to_rad,
        colvar[:, 2] / deg_to_rad,
        c=colvar[:, 3],
        alpha=0.5,
    )

    fig.colorbar(
        scatter,
        ax=ax,
        label="Restraint Bias",
    )

    ax.set_xlabel("phi (deg)")
    ax.set_ylabel("psi (deg)")
    ax.set_title("COLVAR Analysis")

    fig.savefig(
        figname,
        bbox_inches="tight",
    )

    if DISPLAY_FIGS:
        plt.show()

    plt.close(fig)


# ============================================================
# 3) Parse DUMPPROJECTIONS output
#
# Expected columns:
# time, g11, g12, g21, g22
# ============================================================
def metric_from_proj(
    proj_path,
    tmin=None,
    tmax=None,
    discard_frac=0.0,
):
    arr = np.loadtxt(
        proj_path,
        comments="#",
    )

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    t = arr[:, 0]
    g11, g12, g21, g22 = arr[:, 1:5].T

    # Optional time window
    mask = np.ones_like(
        t,
        dtype=bool,
    )

    if tmin is not None:
        mask &= t >= tmin

    if tmax is not None:
        mask &= t <= tmax

    t = t[mask]
    g11 = g11[mask]
    g12 = g12[mask]
    g21 = g21[mask]
    g22 = g22[mask]

    n_total = len(t)

    if n_total == 0:
        raise ValueError(
            f"No frames left in {proj_path} "
            f"after applying tmin/tmax filter."
        )

    discard_n = compute_discard_n(
        n_total,
        discard_frac,
    )

    print(
        f"{proj_path}: discarding "
        f"{discard_n}/{n_total} frames "
        f"({discard_frac:.1%})"
    )

    if discard_n > 0:
        t = t[discard_n:]
        g11 = g11[discard_n:]
        g12 = g12[discard_n:]
        g21 = g21[discard_n:]
        g22 = g22[discard_n:]

    n = len(t)

    if n == 0:
        raise ValueError(
            f"No frames left in {proj_path} after discarding "
            f"discard_frac={discard_frac}."
        )

    M11 = g11.mean()
    M22 = g22.mean()
    M12 = 0.5 * (
        g12.mean()
        + g21.mean()
    )

    M = np.array(
        [
            [M11, M12],
            [M12, M22],
        ],
        dtype=float,
    )

    def se(x):
        if n > 1:
            return x.std(ddof=1) / np.sqrt(n)
        return np.nan

    offdiag = 0.5 * (
        g12 + g21
    )

    M_se = np.array(
        [
            [se(g11), se(offdiag)],
            [se(offdiag), se(g22)],
        ],
        dtype=float,
    )

    return M, M_se, n


# ============================================================
# 4) Free-energy gradient
# ============================================================
def FES_gradient(
    grid_point,
    discard_frac=0.0,
    tmin=None,
    tmax=None,
    kappa=1000.0,
):
    phi_deg, psi_deg, phi, psi = grid_point

    dir_name = f"phi_{phi_deg}_psi_{psi_deg}"
    colvar_file = f"{dir_name}/COLVAR_restrained"

    colvar = np.loadtxt(
        colvar_file,
        comments="#",
    )

    if colvar.ndim == 1:
        colvar = colvar.reshape(1, -1)

    t = colvar[:, 0]
    phi_restrained = colvar[:, 1]
    psi_restrained = colvar[:, 2]

    # Optional time window
    mask = np.ones_like(
        t,
        dtype=bool,
    )

    if tmin is not None:
        mask &= t >= tmin

    if tmax is not None:
        mask &= t <= tmax

    t = t[mask]
    phi_restrained = phi_restrained[mask]
    psi_restrained = psi_restrained[mask]

    n_total = len(t)

    if n_total == 0:
        raise ValueError(
            f"No frames left in {colvar_file} "
            f"after applying tmin/tmax filter."
        )

    discard_n = compute_discard_n(
        n_total,
        discard_frac,
    )

    print(
        f"{colvar_file}: discarding "
        f"{discard_n}/{n_total} frames "
        f"({discard_frac:.1%})"
    )

    if discard_n > 0:
        t = t[discard_n:]
        phi_restrained = phi_restrained[discard_n:]
        psi_restrained = psi_restrained[discard_n:]

    if len(t) == 0:
        raise ValueError(
            f"No frames left in {colvar_file} after discarding "
            f"discard_frac={discard_frac}."
        )

    dphi = phi_restrained - phi
    dpsi = psi_restrained - psi

    # Wrap to [-pi, pi)
    dphi = (
        (dphi + np.pi)
        % (2.0 * np.pi)
        - np.pi
    )

    dpsi = (
        (dpsi + np.pi)
        % (2.0 * np.pi)
        - np.pi
    )

    gradient_phi = -kappa * np.mean(dphi)
    gradient_psi = -kappa * np.mean(dpsi)

    return gradient_phi, gradient_psi


# ============================================================
# 5) Metric assembly
# ============================================================
def FES_metric(
    grid_point,
    discard_frac=0.0,
    tmin=None,
    tmax=None,
):
    phi_deg, psi_deg, phi, psi = grid_point

    dir_name = f"phi_{phi_deg}_psi_{psi_deg}"
    proj_file = os.path.join(
        dir_name,
        "PROJ",
    )

    if not os.path.exists(proj_file):
        raise FileNotFoundError(
            f"Cannot find {proj_file}; "
            f"make sure DUMPPROJECTIONS ran."
        )

    M, M_se, n = metric_from_proj(
        proj_file,
        tmin=tmin,
        tmax=tmax,
        discard_frac=discard_frac,
    )

    return M


# ============================================================
# 6) Housekeeping
# ============================================================
def delete_backupfile():
    directory = "."
    backup_pattern = os.path.join(
        directory,
        "bck.*.PLUMED.OUT",
    )

    for file_path in glob.glob(backup_pattern):
        try:
            os.remove(file_path)
        except OSError as exc:
            print(
                f"Error deleting {file_path}: {exc}"
            )


def delete_temporary_phi_directories():
    """
    删除本次 restrained MD 产生的 phi_* 临时目录。
    """
    subprocess.run(
        'find . -type d -name "phi_*" -exec rm -rf {} +',
        shell=True,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ============================================================
# 7) Main MD interface
# ============================================================
def M_gradient(
    q,
    discard_frac=0.2,
    tmin=None,
    tmax=None,
):
    """
    Parameters
    ----------
    q : torch.Tensor
        Shape (1, 2), containing (phi, psi) in radians.

    discard_frac : float
        Fraction of initial restrained-MD frames to discard.

    Returns
    -------
    M : np.ndarray
        Shape (2, 2).

    F_orig : np.ndarray
        Shape (2,), free-energy gradient in the original
        GROMACS/PLUMED energy unit.
    """
    q = q.cpu().detach().numpy()

    phi_deg_recovered = (
        q[:, 0] / deg_to_rad
    )

    psi_deg_recovered = (
        q[:, 1] / deg_to_rad
    )

    # g:
    # [phi_degree, psi_degree, phi_radian, psi_radian]
    g = np.column_stack(
        (
            phi_deg_recovered,
            psi_deg_recovered,
            q,
        )
    )[0]

    # Restrained simulation
    restrained_simulation(g)

    delete_backupfile()

    # Free-energy gradient
    grad_phi, grad_psi = FES_gradient(
        g,
        discard_frac=discard_frac,
        tmin=tmin,
        tmax=tmax,
        kappa=1000.0,
    )

    F_orig = np.array(
        [grad_phi, grad_psi],
        dtype=float,
    )

    # Metric
    M = FES_metric(
        g,
        discard_frac=discard_frac,
        tmin=tmin,
        tmax=tmax,
    )

    delete_temporary_phi_directories()

    return M, F_orig


# ============================================================
# FES plotting helpers
# ============================================================
def to_numpy(x):
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    return np.asarray(
        x,
        dtype=float,
    )


def crop_regular_grid(
    X,
    Y,
    Z,
    xlim=None,
    ylim=None,
):
    """
    Crop a regular 2D meshgrid X, Y, Z to xlim and ylim.
    """
    if xlim is None:
        xlim = (
            float(np.nanmin(X)),
            float(np.nanmax(X)),
        )
    else:
        xlim = tuple(
            sorted(xlim)
        )

    if ylim is None:
        ylim = (
            float(np.nanmin(Y)),
            float(np.nanmax(Y)),
        )
    else:
        ylim = tuple(
            sorted(ylim)
        )

    inside = (
        (X >= xlim[0])
        & (X <= xlim[1])
        & (Y >= ylim[0])
        & (Y <= ylim[1])
    )

    if not np.any(inside):
        raise ValueError(
            f"No grid points inside xlim={xlim}, ylim={ylim}. "
            f"Available X range: "
            f"[{np.nanmin(X):.6f}, {np.nanmax(X):.6f}], "
            f"Y range: "
            f"[{np.nanmin(Y):.6f}, {np.nanmax(Y):.6f}]"
        )

    rows = np.where(
        np.any(inside, axis=1)
    )[0]

    cols = np.where(
        np.any(inside, axis=0)
    )[0]

    r0 = rows[0]
    r1 = rows[-1] + 1
    c0 = cols[0]
    c1 = cols[-1] + 1

    return (
        X[r0:r1, c0:c1],
        Y[r0:r1, c0:c1],
        Z[r0:r1, c0:c1],
        xlim,
        ylim,
    )


def compute_figsize_from_window(
    xlim,
    ylim,
    scale_factor=1.0,
    base=6.0,
):
    dx = max(
        abs(xlim[1] - xlim[0]),
        1e-12,
    )

    dy = max(
        abs(ylim[1] - ylim[0]),
        1e-12,
    )

    base *= scale_factor
    aspect = dx / dy

    if aspect >= 1.0:
        return (
            base,
            max(4.0, base / aspect),
        )

    return (
        max(4.0, base * aspect),
        base,
    )


def visualize_fes_zoomin_from_npz(
    k,
    x_plot1=None,
    x_plot2=None,
    x_plot3=None,
    x_plot4=None,
    force1=None,
    force2=None,
    force3=None,
    npz_path=(
        "/Users/wuzhiyou/Code/IVM/"
        "MFEP_ala/amber14sb_vacuo/"
        "wtmeta_phi_psi/fes_XYZ.npz"
    ),
    title="Potential Surface",
    xlim=(-np.pi, np.pi),
    ylim=(-np.pi, np.pi),
    scale_factor=1.0,
):
    # --------------------------------------------------------
    # Load X, Y, Z
    # --------------------------------------------------------
    with np.load(npz_path) as loaded:
        grid_x = np.asarray(
            loaded["X"],
            dtype=float,
        )

        grid_y = np.asarray(
            loaded["Y"],
            dtype=float,
        )

        grid_z = np.ma.masked_invalid(
            np.asarray(
                loaded["Z"],
                dtype=float,
            )
        )

    if (
        grid_x.ndim == 1
        and grid_y.ndim == 1
    ):
        grid_x, grid_y = np.meshgrid(
            grid_x,
            grid_y,
            indexing="xy",
        )

    if (
        grid_x.shape != grid_y.shape
        or grid_x.shape != grid_z.shape
    ):
        raise ValueError(
            f"Shape mismatch: "
            f"X{grid_x.shape}, "
            f"Y{grid_y.shape}, "
            f"Z{grid_z.shape}"
        )

    # npz X/Y assumed in degrees
    grid_x = np.deg2rad(grid_x)
    grid_y = np.deg2rad(grid_y)

    # Optional paths and forces assumed in radians
    x_plot1 = to_numpy(x_plot1)
    x_plot2 = to_numpy(x_plot2)
    x_plot3 = to_numpy(x_plot3)
    x_plot4 = to_numpy(x_plot4)

    force1 = to_numpy(force1)
    force2 = to_numpy(force2)
    force3 = to_numpy(force3)

    X_plot, Y_plot, Z_plot, xlim_used, ylim_used = (
        crop_regular_grid(
            grid_x,
            grid_y,
            grid_z,
            xlim=xlim,
            ylim=ylim,
        )
    )

    figsize = compute_figsize_from_window(
        xlim_used,
        ylim_used,
        scale_factor=scale_factor,
    )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    z_min = float(
        np.ma.min(Z_plot)
    )

    z_max = float(
        np.ma.max(Z_plot)
    )

    if z_max > 40.0:
        norm = mpl.colors.TwoSlopeNorm(
            vmin=z_min,
            vcenter=40.0,
            vmax=z_max,
        )

        cp = ax.contourf(
            X_plot,
            Y_plot,
            Z_plot,
            levels=50,
            cmap=mpl.cm.jet,
            norm=norm,
        )
    else:
        cp = ax.contourf(
            X_plot,
            Y_plot,
            Z_plot,
            levels=50,
            cmap=mpl.cm.jet,
        )

    has_legend = False

    if x_plot1 is not None:
        ax.plot(
            x_plot1[:, 0],
            x_plot1[:, 1],
            color="green",
            linestyle="-",
            marker="o",
            markersize=4,
            label=r"x_hat_old",
        )
        has_legend = True

    if x_plot2 is not None:
        ax.plot(
            x_plot2[:, 0],
            x_plot2[:, 1],
            color="blue",
            linestyle="-",
            marker="o",
            markersize=4,
            label=r"x_*",
        )
        has_legend = True

    if x_plot3 is not None:
        ax.plot(
            x_plot3[:, 0],
            x_plot3[:, 1],
            color="purple",
            linestyle="-",
            marker="o",
            markersize=4,
            label=r"x_hat_old + P(x_* - x_hat_old)",
        )
        has_legend = True

    if x_plot4 is not None:
        ax.plot(
            x_plot4[:, 0],
            x_plot4[:, 1],
            color="red",
            linestyle="-",
            marker="o",
            markersize=4,
            label=r"x_reparam_new",
        )
        has_legend = True

    # --------------------------------------------------------
    # Force vectors
    # --------------------------------------------------------
    if x_plot1 is not None:
        traj_x = x_plot1[:, 0]
        traj_y = x_plot1[:, 1]

        base_quiver_kwargs = {
            "angles": "xy",
            "scale_units": "xy",
            "scale": 1,
        }

        if force1 is not None:
            if force1.shape != x_plot1.shape:
                raise ValueError(
                    f"force1 shape {force1.shape} "
                    f"!= x_plot1 shape {x_plot1.shape}"
                )

            ax.quiver(
                traj_x,
                traj_y,
                force1[:, 0],
                force1[:, 1],
                color="purple",
                width=0.006,
                headwidth=4,
                headlength=5,
                **base_quiver_kwargs,
            )

        if force2 is not None:
            if force2.shape != x_plot1.shape:
                raise ValueError(
                    f"force2 shape {force2.shape} "
                    f"!= x_plot1 shape {x_plot1.shape}"
                )

            ax.quiver(
                traj_x,
                traj_y,
                force2[:, 0],
                force2[:, 1],
                color="orange",
                width=0.004,
                headwidth=4,
                headlength=5,
                **base_quiver_kwargs,
            )

        if force3 is not None:
            if force3.shape != x_plot1.shape:
                raise ValueError(
                    f"force3 shape {force3.shape} "
                    f"!= x_plot1 shape {x_plot1.shape}"
                )

            ax.quiver(
                traj_x,
                traj_y,
                force3[:, 0],
                force3[:, 1],
                color="pink",
                width=0.003,
                headwidth=4,
                headlength=5,
                **base_quiver_kwargs,
            )

    special_points = [
        (-2.62593029, 2.77365591, r"$C5$"),
        (-1.33433281, 0.99268968, r"$C7_{eq}$"),
    ]

    for x, y, label in special_points:
        ax.plot(
            x,
            y,
            marker="x",
            color="white",
            markersize=10,
            markeredgewidth=2.5,
            linestyle="None",
            zorder=10,
        )

        ax.text(
            x - 0.5,
            y - 0.5,
            label,
            color="white",
            fontsize=30,
            weight="bold",
            zorder=10,
        )

    divider = make_axes_locatable(ax)

    cax = divider.append_axes(
        "right",
        size="4.5%",
        pad=0.12,
    )

    cbar = fig.colorbar(
        cp,
        cax=cax,
    )

    cbar.set_label("Potential")

    ax.set_xlabel(r"$\phi$ (rad)")
    ax.set_ylabel(r"$\psi$ (rad)")
    ax.set_xlim(xlim_used)
    ax.set_ylim(ylim_used)
    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    if has_legend:
        ax.legend()

    fig.tight_layout()

    if DISPLAY_FIGS:
        plt.show()

    return fig, X_plot, Y_plot, Z_plot


def fig_contour(
    k,
    x_plot1,
    x_plot2,
    x_plot3,
    x_plot4,
    xlim,
    ylim,
    force1=None,
    force2=None,
    force3=None,
    scale_factor=1.3,
):
    """
    构造 contour 图。

    DISPLAY_FIGS=False 时不会弹出窗口。
    该函数本身不保存；实际图片由 fig_save 保存。
    """
    fig, _, _, _ = visualize_fes_zoomin_from_npz(
        k,
        x_plot1=x_plot1,
        x_plot2=x_plot2,
        x_plot3=x_plot3,
        x_plot4=x_plot4,
        force1=force1,
        force2=force2,
        force3=force3,
        xlim=xlim,
        ylim=ylim,
        scale_factor=scale_factor,
    )

    if not DISPLAY_FIGS:
        plt.close(fig)


def fig_save(
    k,
    x_plot1,
    x_plot2,
    x_plot3,
    x_plot4,
    xlim,
    ylim,
    force1=None,
    force2=None,
    force3=None,
    scale_factor=1.3,
    figure_path=None,
):
    """
    保存 contour/path 图，但不要求显示窗口。
    """
    fig, _, _, _ = visualize_fes_zoomin_from_npz(
        k,
        x_plot1=x_plot1,
        x_plot2=x_plot2,
        x_plot3=x_plot3,
        x_plot4=x_plot4,
        force1=force1,
        force2=force2,
        force3=force3,
        xlim=xlim,
        ylim=ylim,
        scale_factor=scale_factor,
    )

    if figure_path is None:
        raise ValueError(
            "figure_path cannot be None in fig_save."
        )

    os.makedirs(
        figure_path,
        exist_ok=True,
    )

    fig.savefig(
        os.path.join(
            figure_path,
            f"images_k{k}.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MFEP / IVM settings
# ============================================================
device = torch.device("cpu")
dtype = torch.float64
N_dim = 2


# ============================================================
# FMEP gradient/metric wrapper
# ============================================================
def gradient_md_use(
    q,
    discard_frac=0.05,
    tmin=None,
    tmax=None,
):
    t0 = time.perf_counter()

    if not torch.is_tensor(q):
        raise TypeError(
            f"q must be a torch.Tensor, got {type(q)}"
        )

    if q.ndim == 1:
        q = q.unsqueeze(0)

    if q.ndim != 2 or q.shape[1] != 2:
        raise ValueError(
            f"q must have shape (N, 2), "
            f"got {tuple(q.shape)}"
        )

    out_device = q.device
    out_dtype = torch.float64
    q_cpu = q.detach().cpu()
    n_points = q_cpu.shape[0]

    def _to_numpy(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()

        return np.asarray(
            x,
            dtype=float,
        )

    Mg_list = []
    M_seq = []
    g_list = []

    for idx in range(n_points):
        q_i = q_cpu[idx:idx + 1]

        M_i, F_i = M_gradient(
            q_i,
            discard_frac=discard_frac,
            tmin=tmin,
            tmax=tmax,
        )

        M_i = _to_numpy(
            M_i
        ).reshape(2, 2)

        F_i = _to_numpy(
            F_i
        ).reshape(2,)

        # kJ/mol -> kcal/mol
        F_i = F_i / 4.184

        Mg_i = M_i @ F_i

        Mg_list.append(Mg_i)

        M_seq.append(
            torch.tensor(
                M_i,
                dtype=out_dtype,
                device=out_device,
            )
        )

        g_list.append(F_i)

    Mg = torch.tensor(
        np.stack(
            Mg_list,
            axis=0,
        ),
        dtype=out_dtype,
        device=out_device,
    )

    g = torch.tensor(
        np.stack(
            g_list,
            axis=0,
        ),
        dtype=out_dtype,
        device=out_device,
    )

    elapsed = (
        time.perf_counter()
        - t0
    )

    print(
        f"serial wall time = {elapsed:.3f} s  "
        f"({n_points} points)"
    )

    return Mg, M_seq, g


# ============================================================
# Image path representation
# ============================================================
class PathPoints(nn.Module):
    def __init__(
        self,
        x0,
        x1,
        N,
        custom_init=None,
    ):
        super().__init__()

        x0 = torch.as_tensor(
            x0,
            dtype=torch.float64,
        )

        x1 = torch.as_tensor(
            x1,
            dtype=torch.float64,
        )

        self.d = x0.numel()
        self.N = N

        self.register_buffer(
            "x0",
            x0,
        )

        self.register_buffer(
            "x1",
            x1,
        )

        if custom_init is not None:
            P_init = torch.as_tensor(
                custom_init,
                dtype=torch.float64,
            )
        else:
            t = torch.linspace(
                0.0,
                1.0,
                N + 2,
                dtype=torch.float64,
            )[1:-1]

            P_init = (
                (1.0 - t).unsqueeze(1) * x0
                + t.unsqueeze(1) * x1
            )

        self.P_list = nn.ParameterList(
            [
                nn.Parameter(
                    P_init[i].unsqueeze(0)
                )
                for i in range(N)
            ]
        )

    def get_full_path(self):
        P = torch.cat(
            list(self.P_list),
            dim=0,
        )

        return torch.vstack(
            [
                self.x0.unsqueeze(0),
                P,
                self.x1.unsqueeze(0),
            ]
        )

    def forward(self):
        return torch.cat(
            list(self.P_list),
            dim=0,
        )


@torch.no_grad()
def set_model_internal_positions(
    model,
    X_NxD,
):
    X_NxD = X_NxD.detach().to(
        device=model.P_list[0].device,
        dtype=model.P_list[0].dtype,
    )

    for j, p in enumerate(model.P_list):
        p.data = X_NxD[j:j + 1].clone()


def central_difference_tangent(X_full):
    return (
        X_full[2:]
        - X_full[:-2]
    )


# ============================================================
# G-orthogonal projectors
# ============================================================
def G_projectors(
    partial_s_x,
    M_seq,
):
    N_local, d = partial_s_x.shape

    assert d == 2

    dev = partial_s_x.device
    dt = partial_s_x.dtype

    G_list = []
    tG_list = []

    for i in range(N_local):
        M_i = M_seq[i]
        G_i = torch.linalg.inv(M_i)

        v_i = partial_s_x[i:i + 1, :]
        Gv = v_i @ G_i

        norm2 = (
            torch.sum(
                v_i * Gv,
                dim=1,
                keepdim=True,
            )
            + 1e-16
        )

        norm_G = torch.sqrt(norm2)
        tG_i = v_i / norm_G

        G_list.append(G_i)
        tG_list.append(tG_i)

    G_all = torch.stack(
        G_list,
        dim=0,
    )

    tG_all = torch.cat(
        tG_list,
        dim=0,
    )

    tG_all_3d = tG_all.unsqueeze(2)

    G_tG = torch.bmm(
        G_all,
        tG_all_3d,
    )

    P_parallel_mats = torch.bmm(
        tG_all_3d,
        G_tG.transpose(1, 2),
    )

    denom = torch.sum(
        tG_all
        * G_tG.squeeze(-1),
        dim=1,
        keepdim=True,
    )

    denom = (
        denom.view(-1, 1, 1)
        + 1e-16
    )

    P_parallel_mats = (
        P_parallel_mats
        / denom
    )

    identity = (
        torch.eye(
            2,
            device=dev,
            dtype=dt,
        )
        .unsqueeze(0)
        .expand(N_local, -1, -1)
    )

    P_perp_mats = (
        identity
        - P_parallel_mats
    )

    def P_parallel_G(x):
        return torch.bmm(
            P_parallel_mats,
            x.unsqueeze(-1),
        ).squeeze(-1)

    def P_perp_G(x):
        return torch.bmm(
            P_perp_mats,
            x.unsqueeze(-1),
        ).squeeze(-1)

    return (
        tG_all,
        P_parallel_G,
        P_perp_G,
        P_parallel_mats,
        P_perp_mats,
    )


def P_perp_euclid(
    x,
    partial_s_x,
    eps=1e-16,
):
    n = torch.sum(
        partial_s_x ** 2,
        dim=1,
        keepdim=True,
    ) ** 0.5

    t = (
        partial_s_x
        / (n + eps)
    )

    inner = torch.sum(
        t * x,
        dim=1,
        keepdim=True,
    )

    return (
        x
        - inner * t
    )


# ============================================================
# NEB tangential reparameterization
# ============================================================
def reparam_by_neb(
    X_full,
    k_s=1.0,
    dtau=0.5,
    max_iter=5000,
    tol=1e-3,
    verbose=False,
):
    X = X_full.clone().detach()

    eps = 1e-16
    resid = float("inf")

    for r in range(max_iter):
        seg = (
            X[1:]
            - X[:-1]
        )

        seg_len = torch.linalg.norm(
            seg,
            dim=1,
        )

        l_minus = seg_len[:-1]
        l_plus = seg_len[1:]

        tang = (
            X[2:]
            - X[:-2]
        )

        tang = tang / (
            torch.linalg.norm(
                tang,
                dim=1,
                keepdim=True,
            )
            + eps
        )

        F_sp = (
            k_s
            * (l_plus - l_minus).unsqueeze(1)
            * tang
        )

        resid = torch.max(
            torch.linalg.norm(
                F_sp,
                dim=1,
            )
        ).item()

        X[1:-1] = (
            X[1:-1]
            + dtau * F_sp
        )

        if resid < tol:
            if verbose:
                print(
                    f"  [reparam_by_neb] "
                    f"converged @ iter {r}, "
                    f"max|F_sp|={resid:.3e}"
                )
            break
    else:
        if verbose:
            print(
                f"  [reparam_by_neb] "
                f"hit max_iter={max_iter}, "
                f"max|F_sp|={resid:.3e}"
            )

    return X, resid


# ============================================================
# Recover free-energy profile along path
# ============================================================
def recover_V_along_curve(
    s,
    g,
    phi_prime,
    V_ref=0.0,
):
    dVds = (
        g * phi_prime
    ).sum(
        dim=1,
        keepdim=True,
    )

    ds = (
        s[1:]
        - s[:-1]
    )

    inc = (
        0.5
        * (dVds[:-1] + dVds[1:])
        * ds
    )

    V = torch.zeros_like(dVds)
    V[0] = V_ref

    V[1:] = (
        V_ref
        + torch.cumsum(
            inc,
            dim=0,
        )
    )

    return (
        V - (torch.min(V) - 1e-6),
        dVds,
    )


# ============================================================
# Optional diagnostic figures
# ============================================================
def fig_cos_V_force(
    i,
    s,
    arc,
    cos,
    g,
    v,
    x_pred,
    norm_F_perp,
    folder_path,
    mycase,
    casenum,
):
    force = np.sqrt(
        np.sum(
            g * g,
            axis=1,
        )
    )

    sin_square = (
        1.0
        - cos ** 2
    )

    fig, axs = plt.subplots(
        1,
        5,
        figsize=(20, 4),
    )

    axs[0].plot(arc, cos, "b.")
    axs[0].set_ylim(-1.1, 1.1)
    axs[0].set_title("cos")
    axs[0].set_box_aspect(1)

    axs[1].plot(arc, sin_square, "b.")
    axs[1].set_ylim(-0.1, 1.1)
    axs[1].set_title("1-cos^2")
    axs[1].set_box_aspect(1)

    axs[2].plot(
        arc.flatten(),
        v,
        "b.",
    )
    axs[2].set_title("Free energy")
    axs[2].set_box_aspect(1)

    axs[3].plot(arc, force, "b.")
    axs[3].set_title("Force")
    axs[3].set_box_aspect(1)

    axs[4].plot(
        arc,
        norm_F_perp,
        "b.",
    )
    axs[4].set_title("norm F_perp")
    axs[4].set_box_aspect(1)

    fig.tight_layout()

    if i % 50 == 0 or i in [1, 100, 200]:
        fig.savefig(
            os.path.join(
                folder_path,
                f"Dot_plot_Iter{i}_{mycase}_{casenum}.eps",
            ),
            bbox_inches="tight",
        )

    if DISPLAY_FIGS:
        plt.show()

    plt.close(fig)


def fig_loss_batch(
    loss,
    loss_reparam,
    loss_1,
    loss_IVP,
    loss_3,
    loss_EL,
):
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(25, 4),
    )

    axes[0].semilogy(
        loss,
        color="#B41830",
        label="I_perp",
    )

    axes[0].semilogy(
        loss_reparam,
        color="#4F18B4",
        label="reparam resid",
    )

    axes[0].set_title(
        "I_perp & reparam"
    )
    axes[0].legend()

    axes[1].semilogy(
        loss_1,
        color="#B41830",
    )
    axes[1].set_title("fmax")

    axes[2].semilogy(
        loss_IVP,
        color="#B41830",
    )
    axes[2].set_title("I_perp")

    axes[3].semilogy(
        loss_3,
        color="#B41830",
    )
    axes[3].set_title("1-cos^2")

    axes[4].semilogy(
        loss_EL,
        color="#B41830",
    )
    axes[4].set_title("fmax (EL)")

    fig.tight_layout()

    if DISPLAY_FIGS:
        plt.show()

    plt.close(fig)


# ============================================================
# Fixed settings
# ============================================================
C1 = [-2.60866606, 2.77661136]
C2 = [-1.3304716, 0.99118795]

N = 20
seed = 41

xlim = (-np.pi, 0)
ylim = (0, np.pi)

REEVALUATE_GRAD_INNER = True

REPARAM_KS = 1.0
REPARAM_DTAU = 1e-3
REPARAM_TOL = 1e-3
REPARAM_MAXITER = 5000
REPARAM_EVERY = 1

base_path = (
    "/Users/wuzhiyou/Code/IVM/"
    "MFEP_ala/amber14sb_vacuo"
)

# 保留原来的变量名。
# True 表示执行 fig_contour/fig_save。
# 是否弹窗由 DISPLAY_FIGS 单独控制。
SHOW_FIGS = SAVE_FIGS


# ============================================================
# Single experiment
# ============================================================
def run_experiment(
    N_sub,
    alpha_IVP,
    eta_in,
):
    batches = 200 // N_sub

    np.random.seed(seed)
    torch.manual_seed(seed)

    mycase = "alanine"

    casenum = (
        f"MFEP_image_Adam_N{N}"
        f"_Nsub{N_sub}"
        f"_alpha{alpha_IVP}"
        f"_lr{eta_in}"
        f"_NEBreparam"
    )

    folder_path = os.path.join(
        base_path,
        casenum,
    )

    os.makedirs(
        folder_path,
        exist_ok=True,
    )

    figure_path = os.path.join(
        folder_path,
        "figures",
    )

    os.makedirs(
        figure_path,
        exist_ok=True,
    )

    print("=" * 78)

    print(
        f"[GRID] "
        f"N_sub={N_sub}  "
        f"alpha_IVP={alpha_IVP}  "
        f"eta_in={eta_in}  "
        f"batches={batches}  "
        f"(N_sub*batches={N_sub * batches})"
    )

    print("=" * 78)

    model = PathPoints(
        C1,
        C2,
        N,
    ).to(device).double()

    # Correct paper I_perp
    loss_perp_seq = []

    # Original Euclidean-mean residual
    loss_perp_euclid_mean_seq = []

    loss_reparam_seq = []
    loss_fmax_seq = []
    loss_cos_seq = []
    index_seq = []

    data = {}

    for i in range(batches + 1):
        # ----------------------------------------------------
        # Frozen reference path
        # ----------------------------------------------------
        X_full = (
            model
            .get_full_path()
            .detach()
        )

        x_hat = (
            model()
            .detach()
        )

        tangent = central_difference_tangent(
            X_full
        )

        # M(phi_k) and grad F(phi_k)
        Mg_ref, M_seq, g_ref = gradient_md_use(
            x_hat
        )

        G_all = torch.stack(
            [
                torch.linalg.inv(M_i)
                for M_i in M_seq
            ],
            dim=0,
        )

        (
            tG_hat,
            P_par_G,
            P_perp_G,
            P_par_mats,
            P_perp_mats,
        ) = G_projectors(
            tangent,
            M_seq,
        )

        M_all = torch.stack(
            M_seq,
            dim=0,
        )

        RESCALE_MG = False

        def build_perpG_Mg(g_now):
            grad_G_F = torch.bmm(
                M_all,
                g_now.unsqueeze(-1),
            ).squeeze(-1)

            if RESCALE_MG:
                gn = torch.linalg.norm(
                    g_now,
                    dim=1,
                    keepdim=True,
                )

                mn = torch.linalg.norm(
                    grad_G_F,
                    dim=1,
                    keepdim=True,
                )

                grad_G_F = grad_G_F * (
                    gn / (mn + 1e-14)
                )

            return torch.bmm(
                P_perp_mats,
                grad_G_F.unsqueeze(-1),
            ).squeeze(-1)

        # ----------------------------------------------------
        # Diagnostics at reference path
        # ----------------------------------------------------
        perpG_Mg = P_perp_G(
            Mg_ref
        )

        # Original Euclidean norm
        norm_F_perp = torch.sum(
            perpG_Mg ** 2,
            dim=1,
            keepdim=True,
        ) ** 0.5

        fmax = float(
            np.max(
                norm_F_perp
                .detach()
                .cpu()
                .numpy()
            )
        )

        # Correct G-norm squared
        norm_F_perp_G_sq = torch.einsum(
            "ni,nij,nj->n",
            perpG_Mg,
            G_all,
            perpG_Mg,
        ).clamp_min(0.0)

        norm_F_perp_G = torch.sqrt(
            norm_F_perp_G_sq
        )

        # G-arclength quadrature weights
        seg_minus = (
            x_hat
            - X_full[:-2]
        )

        seg_plus = (
            X_full[2:]
            - x_hat
        )

        seg_minus_G_sq = torch.einsum(
            "ni,nij,nj->n",
            seg_minus,
            G_all,
            seg_minus,
        ).clamp_min(0.0)

        seg_plus_G_sq = torch.einsum(
            "ni,nij,nj->n",
            seg_plus,
            G_all,
            seg_plus,
        ).clamp_min(0.0)

        seg_minus_G = torch.sqrt(
            seg_minus_G_sq
        )

        seg_plus_G = torch.sqrt(
            seg_plus_G_sq
        )

        node_weights = 0.5 * (
            seg_minus_G
            + seg_plus_G
        )

        # Correct discrete I_perp
        I_perp = torch.sum(
            norm_F_perp_G_sq
            * node_weights
        ).item()

        # Original diagnostic
        I_perp_euclid_mean = torch.mean(
            norm_F_perp ** 2
        ).item()

        tang_e = tangent / (
            torch.sum(
                tangent ** 2,
                dim=1,
                keepdim=True,
            ) ** 0.5
            + 1e-16
        )

        Mg_perp = P_perp_euclid(
            Mg_ref,
            tang_e,
        )

        inner_G = torch.sum(
            tangent * g_ref,
            dim=1,
            keepdim=True,
        )

        tGt = torch.sum(
            tangent
            * torch.bmm(
                G_all,
                tangent.unsqueeze(-1),
            ).squeeze(-1),
            dim=1,
            keepdim=True,
        ) ** 0.5

        gMg = torch.sum(
            g_ref
            * torch.bmm(
                M_all,
                g_ref.unsqueeze(-1),
            ).squeeze(-1),
            dim=1,
            keepdim=True,
        ) ** 0.5

        cos = inner_G / (
            tGt * gMg
            + 1e-16
        )

        cos2 = cos ** 2

        # Free-energy profile
        s_col = torch.linspace(
            0.0,
            1.0,
            N,
            dtype=dtype,
        ).unsqueeze(1)

        ds = 1.0 / (N - 1)

        phi_prime = torch.zeros_like(
            x_hat
        )

        phi_prime[1:-1] = (
            x_hat[2:]
            - x_hat[:-2]
        ) / (2.0 * ds)

        phi_prime[0] = (
            x_hat[1]
            - x_hat[0]
        ) / ds

        phi_prime[-1] = (
            x_hat[-1]
            - x_hat[-2]
        ) / ds

        V_vals, _ = recover_V_along_curve(
            s_col,
            g_ref,
            phi_prime,
            V_ref=0.0,
        )

        v = (
            V_vals
            .cpu()
            .detach()
            .numpy()
            .flatten()
        )

        # ----------------------------------------------------
        # Inner Adam
        # ----------------------------------------------------
        delta = nn.Parameter(
            torch.zeros(
                N,
                2,
                dtype=dtype,
                device=device,
            )
        )

        opt_inner = torch.optim.Adam(
            [delta],
            lr=eta_in,
        )

        perpG_Mg_ref = None

        for i_sub in range(N_sub):
            with torch.no_grad():
                current_x = torch.bmm(
                    P_perp_mats,
                    delta.detach().unsqueeze(-1),
                ).squeeze(-1)

                x_eval = (
                    x_hat
                    + current_x
                )

            if (
                i_sub == 0
                or not REEVALUATE_GRAD_INNER
            ):
                g = g_ref
            else:
                _, _, g = gradient_md_use(
                    x_eval
                )

            with torch.no_grad():
                perpG_Mg_eval = build_perpG_Mg(
                    g
                )

                if i_sub == 0:
                    perpG_Mg_ref = (
                        perpG_Mg_eval.clone()
                    )

                dI_ddelta = (
                    perpG_Mg_eval
                    + (1.0 / alpha_IVP)
                    * current_x
                )

            opt_inner.zero_grad()

            delta.grad = (
                dI_ddelta
                .detach()
                .clone()
            )

            opt_inner.step()

        # ----------------------------------------------------
        # Commit normal update
        # ----------------------------------------------------
        with torch.no_grad():
            current_x = torch.bmm(
                P_perp_mats,
                delta.detach().unsqueeze(-1),
            ).squeeze(-1)

            x_new = (
                x_hat
                + current_x
            )

        max_step = float(
            torch.max(
                torch.linalg.norm(
                    current_x,
                    dim=1,
                )
            ).item()
        )

        print(
            f"[outer {i}] "
            f"I_perp={I_perp:.4e}  "
            f"I_perp_euclid_mean={I_perp_euclid_mean:.4e}  "
            f"fmax={fmax:.4e}  "
            f"max_normal_step={max_step:.4e} rad"
        )

        set_model_internal_positions(
            model,
            x_new,
        )

        # ----------------------------------------------------
        # NEB reparameterization
        # ----------------------------------------------------
        reparam_resid = 0.0

        if (
            REPARAM_EVERY > 0
            and i % REPARAM_EVERY == 0
        ):
            X_new_full = (
                model
                .get_full_path()
                .detach()
            )

            X_reparam, reparam_resid = (
                reparam_by_neb(
                    X_new_full,
                    k_s=REPARAM_KS,
                    dtau=REPARAM_DTAU,
                    max_iter=REPARAM_MAXITER,
                    tol=REPARAM_TOL,
                    verbose=(i % 10 == 0),
                )
            )

            set_model_internal_positions(
                model,
                X_reparam[1:-1],
            )

        x_reparam_new = (
            model()
            .detach()
        )

        # ----------------------------------------------------
        # Record sequences
        # ----------------------------------------------------
        index_seq.append(i)

        loss_perp_seq.append(
            I_perp
        )

        loss_perp_euclid_mean_seq.append(
            I_perp_euclid_mean
        )

        loss_reparam_seq.append(
            reparam_resid
            if reparam_resid > 0
            else 1e-16
        )

        loss_fmax_seq.append(
            fmax
        )

        loss_cos_seq.append(
            float(
                torch.mean(
                    1.0 - cos2
                ).item()
            )
        )

        # ----------------------------------------------------
        # Save figures without displaying them
        # ----------------------------------------------------
        if SHOW_FIGS and (
            i % 1 == 0
            or i in [1, 50, 100, 200]
        ):
            fig_contour(
                i,
                x_hat.cpu().numpy(),
                x_reparam_new.cpu().numpy(),
                None,
                None,
                xlim=xlim,
                ylim=ylim,
            )

            fig_save(
                i * N_sub,
                x_hat.cpu().numpy(),
                x_new.cpu().numpy(),
                x_new.cpu().numpy(),
                x_reparam_new.cpu().numpy(),
                xlim=xlim,
                ylim=ylim,
                force1=-perpG_Mg.cpu().numpy(),
                force2=-g_ref.cpu().numpy(),
                force3=-Mg_perp.cpu().numpy(),
                figure_path=figure_path,
            )

            torch.save(
                model.state_dict(),
                os.path.join(
                    folder_path,
                    f"model_MFEP_"
                    f"{mycase}_{casenum}_{i}.pth",
                ),
            )

            plt.close("all")

        # ----------------------------------------------------
        # Save global sequences
        # ----------------------------------------------------
        data["index_seq"] = np.asarray(
            index_seq
        )

        data["I_perp_seq"] = np.asarray(
            loss_perp_seq
        )

        data["I_perp_euclid_mean_seq"] = np.asarray(
            loss_perp_euclid_mean_seq
        )

        data["fmax_seq"] = np.asarray(
            loss_fmax_seq
        )

        data["reparam_resid_seq"] = np.asarray(
            loss_reparam_seq
        )

        data["cos_seq"] = np.asarray(
            loss_cos_seq
        )

        data[f"path_iter_{i}"] = (
            x_reparam_new
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Reference path data
        # ----------------------------------------------------
        data[f"phi_ref_iter_{i}"] = (
            x_hat
            .cpu()
            .numpy()
        )

        # Existing free-energy gradient storage
        data[f"g_ref_iter_{i}"] = (
            g_ref
            .cpu()
            .numpy()
        )

        # ====================================================
        # New: store reference-path metric M and F
        #
        # M_ref_iter_i:
        #     shape (N, 2, 2)
        #
        # F_ref_iter_i:
        #     shape (N, 2)
        #     same numerical value as g_ref_iter_i
        #     units after conversion: kcal/mol/rad
        # ====================================================
        data[f"M_ref_iter_{i}"] = (
            M_all
            .detach()
            .cpu()
            .numpy()
        )

        data[f"F_ref_iter_{i}"] = (
            g_ref
            .detach()
            .cpu()
            .numpy()
        )

        data[f"perpG_Mg_iter_{i}"] = (
            perpG_Mg_ref
            .cpu()
            .numpy()
        )

        data[f"norm_perpG_Mg_iter_{i}"] = (
            torch.linalg.norm(
                perpG_Mg_ref,
                dim=1,
            )
            .cpu()
            .numpy()
        )

        perpG_Mg_ref_G_sq = torch.einsum(
            "ni,nij,nj->n",
            perpG_Mg_ref,
            G_all,
            perpG_Mg_ref,
        ).clamp_min(0.0)

        data[f"norm_perpG_Mg_G_iter_{i}"] = (
            torch.sqrt(
                perpG_Mg_ref_G_sq
            )
            .cpu()
            .numpy()
        )

        data[f"node_weights_iter_{i}"] = (
            node_weights
            .cpu()
            .numpy()
        )

        np.savez(
            os.path.join(
                folder_path,
                f"{mycase}_{casenum}.npz",
            ),
            **data,
        )

    # --------------------------------------------------------
    # Final model
    # --------------------------------------------------------
    torch.save(
        model.state_dict(),
        os.path.join(
            folder_path,
            f"model_MFEP_"
            f"{mycase}_{casenum}_final.pth",
        ),
    )

    return {
        "N_sub": N_sub,
        "alpha_IVP": alpha_IVP,
        "eta_in": eta_in,
        "batches": batches,

        "final_I_perp": loss_perp_seq[-1],
        "min_I_perp": float(
            np.min(loss_perp_seq)
        ),

        "final_I_perp_euclid_mean": (
            loss_perp_euclid_mean_seq[-1]
        ),
        "min_I_perp_euclid_mean": float(
            np.min(
                loss_perp_euclid_mean_seq
            )
        ),

        "final_fmax": loss_fmax_seq[-1],
        "final_cos": loss_cos_seq[-1],

        "min_fmax": float(
            np.min(loss_fmax_seq)
        ),
    }


# ============================================================
# Resume helpers
# ============================================================
def config_paths(
    N_sub,
    alpha_IVP,
    eta_in,
):
    mycase = "alanine"

    casenum = (
        f"MFEP_image_Adam_N{N}"
        f"_Nsub{N_sub}"
        f"_alpha{alpha_IVP}"
        f"_lr{eta_in}"
        f"_NEBreparam"
    )

    folder_path = os.path.join(
        base_path,
        casenum,
    )

    final_model = os.path.join(
        folder_path,
        f"model_MFEP_"
        f"{mycase}_{casenum}_final.pth",
    )

    npz_file = os.path.join(
        folder_path,
        f"{mycase}_{casenum}.npz",
    )

    return (
        mycase,
        casenum,
        folder_path,
        final_model,
        npz_file,
    )


def is_config_done(
    N_sub,
    alpha_IVP,
    eta_in,
):
    batches = 200 // N_sub

    (
        _,
        _,
        _,
        final_model,
        npz_file,
    ) = config_paths(
        N_sub,
        alpha_IVP,
        eta_in,
    )

    if os.path.exists(final_model):
        return True

    if os.path.exists(npz_file):
        try:
            with np.load(
                npz_file,
                allow_pickle=True,
            ) as loaded:
                idx = loaded["index_seq"]

                if (
                    idx.size > 0
                    and int(np.max(idx)) >= batches
                ):
                    return True

        except Exception:
            return False

    return False


def load_result_from_npz(
    N_sub,
    alpha_IVP,
    eta_in,
):
    batches = 200 // N_sub

    (
        _,
        _,
        _,
        _,
        npz_file,
    ) = config_paths(
        N_sub,
        alpha_IVP,
        eta_in,
    )

    with np.load(
        npz_file,
        allow_pickle=True,
    ) as loaded:
        I_perp_seq = np.asarray(
            loaded["I_perp_seq"],
            dtype=float,
        )

        if (
            "I_perp_euclid_mean_seq"
            in loaded.files
        ):
            I_perp_euclid_mean_seq = np.asarray(
                loaded["I_perp_euclid_mean_seq"],
                dtype=float,
            )
        else:
            I_perp_euclid_mean_seq = (
                I_perp_seq.copy()
            )

        fmax_seq = np.asarray(
            loaded["fmax_seq"],
            dtype=float,
        )

        cos_seq = np.asarray(
            loaded["cos_seq"],
            dtype=float,
        )

    return {
        "N_sub": N_sub,
        "alpha_IVP": alpha_IVP,
        "eta_in": eta_in,
        "batches": batches,

        "final_I_perp": float(
            I_perp_seq[-1]
        ),
        "min_I_perp": float(
            np.min(I_perp_seq)
        ),

        "final_I_perp_euclid_mean": float(
            I_perp_euclid_mean_seq[-1]
        ),
        "min_I_perp_euclid_mean": float(
            np.min(
                I_perp_euclid_mean_seq
            )
        ),

        "final_fmax": float(
            fmax_seq[-1]
        ),
        "final_cos": float(
            cos_seq[-1]
        ),
        "min_fmax": float(
            np.min(fmax_seq)
        ),
    }


# ============================================================
# Grid settings
# ============================================================
GRID_N_SUB = [4, 5, 10]
GRID_ALPHA = [0.1, 1, 10]
GRID_ETA = [0.03, 0.01, 0.003, 0.001]

FORCE_RERUN = False


# ============================================================
# Main
# ============================================================
def main():
    all_configs = [
        (ns, al, et)
        for ns in GRID_N_SUB
        for al in GRID_ALPHA
        for et in GRID_ETA
    ]

    done_flags = {
        cfg: is_config_done(*cfg)
        for cfg in all_configs
    }

    n_done = sum(
        done_flags.values()
    )

    print("=" * 78)

    print(
        f"[RESUME] 共 {len(all_configs)} 个配置; "
        f"检测到已完成 {n_done} 个, "
        f"待跑 {len(all_configs) - n_done} 个"
        + (
            "  (FORCE_RERUN=True, 将全部重跑)"
            if FORCE_RERUN
            else ""
        )
    )

    for cfg in all_configs:
        ns, al, et = cfg

        tag = (
            "DONE  跳过"
            if (
                done_flags[cfg]
                and not FORCE_RERUN
            )
            else "TODO  待跑"
        )

        print(
            f"    [{tag}]  "
            f"N_sub={ns:>2}  "
            f"alpha={al:>4}  "
            f"eta={et}"
        )

    print("=" * 78)
    # ======== 新增：交互询问 ========
    answer = input("检查完毕，是否开始运行？(y/n): ").strip().lower()
    if answer != 'y':
        print("运行已取消。")
        sys.exit(0)
    # ================================

    summary = []

    for cfg in all_configs:
        N_sub, alpha_IVP, eta_in = cfg

        if (
            done_flags[cfg]
            and not FORCE_RERUN
        ):
            print(
                f"[SKIP] "
                f"N_sub={N_sub} "
                f"alpha={alpha_IVP} "
                f"eta={eta_in} "
                f"已完成, 直接读取旧结果"
            )

            try:
                summary.append(
                    load_result_from_npz(
                        N_sub,
                        alpha_IVP,
                        eta_in,
                    )
                )

            except Exception as exc:
                print(
                    f"       "
                    f"(读取旧结果失败, "
                    f"汇总里将缺此项: {exc})"
                )

            continue

        result = run_experiment(
            N_sub,
            alpha_IVP,
            eta_in,
        )

        summary.append(result)

    print("\n" + "=" * 78)
    print("GRID SEARCH SUMMARY  (按 final_fmax 升序)")
    print("=" * 78)

    summary_sorted = sorted(
        summary,
        key=lambda r: r["final_fmax"],
    )

    for result in summary_sorted:
        print(
            f"N_sub={result['N_sub']:>2}  "
            f"alpha={result['alpha_IVP']:>4}  "
            f"eta={result['eta_in']:<6}  "
            f"| final_I_perp="
            f"{result['final_I_perp']:.3e}  "
            f"final_I_perp_euclid_mean="
            f"{result['final_I_perp_euclid_mean']:.3e}  "
            f"final_fmax="
            f"{result['final_fmax']:.3e}  "
            f"final_1-cos2="
            f"{result['final_cos']:.3e}  "
            f"min_fmax="
            f"{result['min_fmax']:.3e}"
        )

    np.savez(
        os.path.join(
            base_path,
            "GRID_SUMMARY.npz",
        ),
        summary=np.array(
            summary,
            dtype=object,
        ),
    )


if __name__ == "__main__":
    main()