import os
import time
import numpy as np
import torch

import matplotlib

# 非交互式后端：保存图片但不弹窗
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

try:
    from ase.mep.neb import NEB, NEBOptimizer
except ImportError:
    from ase.neb import NEB, NEBOptimizer


# ============================================================
# 复用现有 alanine restrained-MD oracle 和绘图函数
#
# 必须确保 IVM_shell.py 最后使用：
#
# if __name__ == "__main__":
#     main()
#
# 否则 import 时会直接启动 IVM 网格搜索。
# ============================================================
import IVM_shell as ala_ivm

from ase_mfep_neb import (
    install_central_difference_improved_tangent,
)


M_gradient = ala_ivm.M_gradient
fig_save = ala_ivm.fig_save

# 关闭 GROMACS 大量终端输出
if hasattr(ala_ivm, "GROMACS_QUIET"):
    ala_ivm.GROMACS_QUIET = True


# ============================================================
# 基本设置
# ============================================================
device = torch.device("cpu")
dtype = torch.float64

seed = 41

np.random.seed(seed)
torch.manual_seed(seed)

# 与 IVM 完全相同的端点
C1 = np.array(
    [-2.60866606, 2.77661136],
    dtype=float,
)

C2 = np.array(
    [-1.3304716, 0.99118795],
    dtype=float,
)

# 内部 images 数，与 IVM 相同
N = 20

xlim = (-np.pi, 0.0)
ylim = (0.0, np.pi)

base_path = (
    "/Users/wuzhiyou/Code/IVM/"
    "MFEP_ala/amber14sb_vacuo"
)


# ============================================================
# Restrained-MD 参数
#
# 与 IVM 中 gradient_md_use 的默认设置一致
# ============================================================
DISCARD_FRAC = 0.05
TMIN = None
TMAX = None


# ============================================================
# NEB / ODE 参数
# ============================================================

# 不向 NEB 传入 k，因此使用 ASE 默认 k=0.1
#
# 只用于文件名和结果记录。
ASE_DEFAULT_K = 0.1

# 最大 accepted ODE steps
MAX_NEB_STEPS = 200

# ODE 终止条件：
# ASE NEB 总残差小于此值时终止。
#
# 设置得较小可以使其尽量跑满 MAX_NEB_STEPS。
NEB_FMAX_TOL = 1e-8

# 是否强制覆盖已经完成的结果
FORCE_RERUN = False


# ============================================================
# 图片与保存设置
# ============================================================

# 保存 path 图片，但不显示
SAVE_FIGS = True

# 每隔多少个 accepted ODE steps 保存一次图片
SAVE_FIG_EVERY = 10

# 每个 accepted ODE state 都写入 npz
SAVE_EVERY_STATE = True


# ============================================================
# 全局 restrained-MD oracle 计数
#
# 每个 internal image 调用一次 M_gradient：
#     GLOBAL_IMAGE_MD_CALLS += 1
#
# N 次 image 调用等于一次完整路径 gradient sweep。
# ============================================================
GLOBAL_IMAGE_MD_CALLS = 0


# ============================================================
# CV <-> ASE Atoms
#
# 使用一个虚拟 H 原子存储二维 CV：
#
#     position[0,0] = phi
#     position[0,1] = psi
#     position[0,2] = 0
#
# 这里只把 ASE Atoms 当作二维变量容器。
# ============================================================
def atoms_to_q(atoms):
    positions = atoms.get_positions()

    return np.array(
        [
            positions[0, 0],
            positions[0, 1],
        ],
        dtype=float,
    )


def make_cv_atoms(q, calculator=None):
    q = np.asarray(
        q,
        dtype=float,
    ).reshape(2,)

    atoms = Atoms(
        "H",
        positions=[
            [
                q[0],
                q[1],
                0.0,
            ]
        ],
    )

    atoms.pbc = False

    if calculator is not None:
        atoms.calc = calculator

    return atoms


# ============================================================
# 端点 calculator
#
# 端点固定，不需要 restrained MD。
#
# 由于 method="improvedtangent" 的 ASE NEB 会读取端点能量，
# 这里返回常数零能量和零力。
#
# 自定义 tangent 不使用能量，并且 climb=False，
# 所以常数零能量不会参与实际 tangent 计算。
# ============================================================
class ZeroEndpointCalculator(Calculator):
    implemented_properties = [
        "energy",
        "forces",
    ]

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        super().calculate(
            atoms,
            properties,
            system_changes,
        )

        self.results["energy"] = 0.0

        self.results["forces"] = np.zeros(
            (len(atoms), 3),
            dtype=float,
        )


# ============================================================
# Alanine MFEP calculator
#
# M_gradient 返回：
#
#     M(q)
#     nabla F(q)
#
# 这里向 ASE 返回：
#
#     force = -M(q) nabla F(q)
#
# 因此 ASE NEB 内部会投影：
#
#     -P_E [M nabla F]
#
# 它的零残差条件与 MFEP 条件相同。
# ============================================================
class AlanineMFEPGradientCalculator(Calculator):
    implemented_properties = [
        "energy",
        "forces",
    ]

    def __init__(
        self,
        discard_frac=0.05,
        tmin=None,
        tmax=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.discard_frac = discard_frac
        self.tmin = tmin
        self.tmax = tmax

        self.md_calls = 0

        # 最近一次 oracle 结果
        self.last_q = None
        self.last_M = None
        self.last_F = None
        self.last_Mg = None

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        global GLOBAL_IMAGE_MD_CALLS

        super().calculate(
            atoms,
            properties,
            system_changes,
        )

        q = atoms_to_q(atoms)

        q_tensor = torch.tensor(
            q.reshape(1, 2),
            dtype=dtype,
            device=device,
        )

        # ----------------------------------------------------
        # M_gradient 返回：
        #
        # M:
        #     shape (2,2)
        #
        # F_orig:
        #     shape (2,)
        #     原始单位 kJ/mol/rad
        # ----------------------------------------------------
        M, F_orig = M_gradient(
            q_tensor,
            discard_frac=self.discard_frac,
            tmin=self.tmin,
            tmax=self.tmax,
        )

        M = np.asarray(
            M,
            dtype=float,
        ).reshape(2, 2)

        F_orig = np.asarray(
            F_orig,
            dtype=float,
        ).reshape(2,)

        # 与 IVM 的 gradient_md_use 一致：
        # kJ/mol -> kcal/mol
        F = F_orig / 4.184

        # Riemannian gradient
        Mg = M @ F

        ase_forces = np.zeros(
            (len(atoms), 3),
            dtype=float,
        )

        # ASE force = -gradient
        ase_forces[0, 0] = -Mg[0]
        ase_forces[0, 1] = -Mg[1]

        # 没有自由能值，使用常数零占位。
        # 自定义 tangent 不依赖 energy。
        self.results["energy"] = 0.0
        self.results["forces"] = ase_forces

        # 保存本次 reference/oracle 数据
        self.last_q = q.copy()
        self.last_M = M.copy()
        self.last_F = F.copy()
        self.last_Mg = Mg.copy()

        self.md_calls += 1
        GLOBAL_IMAGE_MD_CALLS += 1


# ============================================================
# 与 IVM 相同的直线初始路径
# ============================================================
def get_linear_initial_path(
    q0,
    q1,
    n_internal,
):
    t = np.linspace(
        0.0,
        1.0,
        n_internal + 2,
    )[1:-1]

    return (
        (1.0 - t[:, None]) * q0[None, :]
        + t[:, None] * q1[None, :]
    )


# ============================================================
# 从 ASE images 提取完整二维路径
# ============================================================
def images_to_path(images):
    return np.array(
        [
            atoms_to_q(image)
            for image in images
        ],
        dtype=float,
    )


# ============================================================
# 与 IVM 相同的中心差分 tangent
#
# 输入：
#     path_full: (N+2, 2)
#
# 输出：
#     tangent: (N, 2)
# ============================================================
def central_difference_tangent(path_full):
    path_full = np.asarray(
        path_full,
        dtype=float,
    )

    return (
        path_full[2:]
        - path_full[:-2]
    )


# ============================================================
# 构造 G-正交投影
#
# G = M^{-1}
#
# P_parallel = tau_G (G tau_G)^T
# P_perp = I - P_parallel
#
# 与 IVM 的 G_projectors 一致。
# ============================================================
def compute_G_projectors(
    tangent,
    M_all,
):
    tangent = np.asarray(
        tangent,
        dtype=float,
    )

    M_all = np.asarray(
        M_all,
        dtype=float,
    )

    n_images = tangent.shape[0]

    G_all = np.linalg.inv(
        M_all
    )

    P_parallel_all = np.zeros(
        (n_images, 2, 2),
        dtype=float,
    )

    P_perp_all = np.zeros(
        (n_images, 2, 2),
        dtype=float,
    )

    identity = np.eye(
        2,
        dtype=float,
    )

    for i in range(n_images):
        v = tangent[i]
        G_i = G_all[i]

        norm_G_sq = float(
            v @ G_i @ v
        )

        norm_G = np.sqrt(
            max(norm_G_sq, 1e-16)
        )

        tau_G = v / norm_G
        G_tau_G = G_i @ tau_G

        denom = float(
            tau_G @ G_tau_G
        )

        P_parallel = (
            np.outer(
                tau_G,
                G_tau_G,
            )
            / max(denom, 1e-16)
        )

        P_parallel_all[i] = (
            P_parallel
        )

        P_perp_all[i] = (
            identity
            - P_parallel
        )

    return (
        G_all,
        P_parallel_all,
        P_perp_all,
    )


# ============================================================
# Euclidean normal projection
#
# 用于记录 ASE NEB 实际使用的：
#
#     P_E (M nabla F)
# ============================================================
def compute_Euclidean_perp(
    vectors,
    tangent,
):
    vectors = np.asarray(
        vectors,
        dtype=float,
    )

    tangent = np.asarray(
        tangent,
        dtype=float,
    )

    tangent_norm = np.linalg.norm(
        tangent,
        axis=1,
        keepdims=True,
    )

    tau = tangent / (
        tangent_norm + 1e-16
    )

    inner = np.sum(
        tau * vectors,
        axis=1,
        keepdims=True,
    )

    return (
        vectors
        - inner * tau
    )


# ============================================================
# 计算与 IVM 一致的两个整路径指标
#
# 1. 论文中的 G-metric I_perp：
#
#    I_perp_G
#    = int |P_G M nabla F|_G^2 |phi'|_G ds
#
# 2. Euclidean residual integral：
#
#    I_perp_E
#    = int ||P_G M nabla F||_2^2 |phi'|_2 ds
#
# 注意第二个指标仍使用 P_G M nabla F，
# 这样才能与你之前计算 IVM 结果的 notebook 一致。
# ============================================================
def compute_integrated_metrics(
    path_full,
    M_all,
    F_all,
):
    path_full = np.asarray(
        path_full,
        dtype=float,
    )

    M_all = np.asarray(
        M_all,
        dtype=float,
    )

    F_all = np.asarray(
        F_all,
        dtype=float,
    )

    path_internal = path_full[1:-1]

    tangent = central_difference_tangent(
        path_full
    )

    (
        G_all,
        P_parallel_G_all,
        P_perp_G_all,
    ) = compute_G_projectors(
        tangent,
        M_all,
    )

    # --------------------------------------------------------
    # M nabla F
    # --------------------------------------------------------
    Mg_all = np.einsum(
        "nij,nj->ni",
        M_all,
        F_all,
    )

    # --------------------------------------------------------
    # G-normal component
    #
    # 与 IVM 中的 perpG_Mg 一致
    # --------------------------------------------------------
    perpG_Mg = np.einsum(
        "nij,nj->ni",
        P_perp_G_all,
        Mg_all,
    )

    # --------------------------------------------------------
    # Euclidean-normal component
    #
    # ASE NEB 实际使用的物理 normal force
    # --------------------------------------------------------
    perpE_Mg = compute_Euclidean_perp(
        Mg_all,
        tangent,
    )

    # --------------------------------------------------------
    # 相邻 segments
    # --------------------------------------------------------
    seg_minus = (
        path_internal
        - path_full[:-2]
    )

    seg_plus = (
        path_full[2:]
        - path_internal
    )

    # --------------------------------------------------------
    # G-arclength quadrature weights
    # --------------------------------------------------------
    seg_minus_G_sq = np.einsum(
        "ni,nij,nj->n",
        seg_minus,
        G_all,
        seg_minus,
    )

    seg_plus_G_sq = np.einsum(
        "ni,nij,nj->n",
        seg_plus,
        G_all,
        seg_plus,
    )

    seg_minus_G = np.sqrt(
        np.maximum(
            seg_minus_G_sq,
            0.0,
        )
    )

    seg_plus_G = np.sqrt(
        np.maximum(
            seg_plus_G_sq,
            0.0,
        )
    )

    node_weights_G = 0.5 * (
        seg_minus_G
        + seg_plus_G
    )

    # --------------------------------------------------------
    # Euclidean arclength quadrature weights
    # --------------------------------------------------------
    seg_minus_E = np.linalg.norm(
        seg_minus,
        axis=1,
    )

    seg_plus_E = np.linalg.norm(
        seg_plus,
        axis=1,
    )

    node_weights_E = 0.5 * (
        seg_minus_E
        + seg_plus_E
    )

    # --------------------------------------------------------
    # 论文中的 G-metric residual
    #
    # |r_i|_G^2 = r_i^T G_i r_i
    # --------------------------------------------------------
    perpG_norm_G_sq = np.einsum(
        "ni,nij,nj->n",
        perpG_Mg,
        G_all,
        perpG_Mg,
    )

    perpG_norm_G_sq = np.maximum(
        perpG_norm_G_sq,
        0.0,
    )

    I_perp_G = float(
        np.sum(
            perpG_norm_G_sq
            * node_weights_G
        )
    )

    # --------------------------------------------------------
    # Euclidean residual integral
    #
    # ||P_G M nabla F||_2^2 |phi'|_2
    # --------------------------------------------------------
    perpG_norm_E_sq = np.sum(
        perpG_Mg ** 2,
        axis=1,
    )

    I_perp_E = float(
        np.sum(
            perpG_norm_E_sq
            * node_weights_E
        )
    )

    # 与当前 IVM fmax 定义一致
    fmax_Gperp_Euclidean = float(
        np.max(
            np.sqrt(
                perpG_norm_E_sq
            )
        )
    )

    # ASE NEB 实际的物理法向力
    fmax_NEB_physical = float(
        np.max(
            np.linalg.norm(
                perpE_Mg,
                axis=1,
            )
        )
    )

    return {
        "G_all": G_all,
        "P_parallel_G_all": P_parallel_G_all,
        "P_perp_G_all": P_perp_G_all,
        "Mg_all": Mg_all,
        "perpG_Mg": perpG_Mg,
        "perpE_Mg": perpE_Mg,
        "node_weights_G": node_weights_G,
        "node_weights_E": node_weights_E,
        "perpG_norm_G_sq": perpG_norm_G_sq,
        "perpG_norm_E_sq": perpG_norm_E_sq,
        "I_perp_G": I_perp_G,
        "I_perp_E": I_perp_E,
        "fmax_Gperp_Euclidean": (
            fmax_Gperp_Euclidean
        ),
        "fmax_NEB_physical": (
            fmax_NEB_physical
        ),
    }


# ============================================================
# 收集当前 internal images 的 M、F 和 M F
#
# 在调用此函数之前，必须先调用 neb.get_forces()，
# 确保 calculator cache 对应当前路径。
# ============================================================
def collect_internal_oracle_data(images):
    M_list = []
    F_list = []
    Mg_list = []

    for image_index, image in enumerate(
        images[1:-1],
        start=1,
    ):
        calculator = image.calc

        if not isinstance(
            calculator,
            AlanineMFEPGradientCalculator,
        ):
            raise TypeError(
                f"Internal image {image_index} does not use "
                f"AlanineMFEPGradientCalculator."
            )

        if (
            calculator.last_q is None
            or calculator.last_M is None
            or calculator.last_F is None
            or calculator.last_Mg is None
        ):
            raise RuntimeError(
                f"No oracle data available for image "
                f"{image_index}. Call neb.get_forces() first."
            )

        current_q = atoms_to_q(
            image
        )

        if not np.allclose(
            current_q,
            calculator.last_q,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Cached oracle position for image "
                f"{image_index} is stale."
            )

        M_list.append(
            calculator.last_M.copy()
        )

        F_list.append(
            calculator.last_F.copy()
        )

        Mg_list.append(
            calculator.last_Mg.copy()
        )

    return (
        np.stack(
            M_list,
            axis=0,
        ),
        np.stack(
            F_list,
            axis=0,
        ),
        np.stack(
            Mg_list,
            axis=0,
        ),
    )


# ============================================================
# 创建 NEB images
# ============================================================
def build_neb_images():
    initial_internal = get_linear_initial_path(
        C1,
        C2,
        N,
    )

    images = []

    # 固定起点
    images.append(
        make_cv_atoms(
            C1,
            calculator=ZeroEndpointCalculator(),
        )
    )

    # 内部 images
    for q in initial_internal:
        images.append(
            make_cv_atoms(
                q,
                calculator=(
                    AlanineMFEPGradientCalculator(
                        discard_frac=DISCARD_FRAC,
                        tmin=TMIN,
                        tmax=TMAX,
                    )
                ),
            )
        )

    # 固定终点
    images.append(
        make_cv_atoms(
            C2,
            calculator=ZeroEndpointCalculator(),
        )
    )

    return images


# ============================================================
# 结果路径
# ============================================================
def neb_result_paths():
    casenum = (
        f"MFEP_ASE_NEB_ODE_N{N}"
        f"_kDefault"
        f"_centralImprovedTangent"
    )

    folder_path = os.path.join(
        base_path,
        casenum,
    )

    figure_path = os.path.join(
        folder_path,
        "figures",
    )

    result_file = os.path.join(
        folder_path,
        f"alanine_{casenum}.npz",
    )

    done_file = os.path.join(
        folder_path,
        "DONE",
    )

    return (
        casenum,
        folder_path,
        figure_path,
        result_file,
        done_file,
    )


# ============================================================
# 运行唯一一次 NEB ODE 实验
# ============================================================
def run_neb_experiment():
    global GLOBAL_IMAGE_MD_CALLS

    GLOBAL_IMAGE_MD_CALLS = 0

    (
        casenum,
        folder_path,
        figure_path,
        result_file,
        done_file,
    ) = neb_result_paths()

    os.makedirs(
        folder_path,
        exist_ok=True,
    )

    os.makedirs(
        figure_path,
        exist_ok=True,
    )

    if (
        os.path.exists(done_file)
        and os.path.exists(result_file)
        and not FORCE_RERUN
    ):
        print(
            "[SKIP] ASE NEB ODE 已完成："
        )
        print(result_file)
        return

    print("\n" + "=" * 80)
    print("Alanine MFEP — ASE NEB + adaptive ODE")
    print("=" * 80)
    print(f"N internal images : {N}")
    print(f"ASE default k     : {ASE_DEFAULT_K}")
    print("NEB method        : improvedtangent")
    print("Tangent override  : central difference, same as IVM")
    print("Optimizer         : NEBOptimizer(method='ode')")
    print(f"Maximum ODE steps : {MAX_NEB_STEPS}")
    print(f"fmax tolerance    : {NEB_FMAX_TOL}")
    print("=" * 80)

    images = build_neb_images()

    # --------------------------------------------------------
    # 不传 k：
    # 使用 ASE NEB 默认弹簧常数 k=0.1
    # --------------------------------------------------------
    neb = NEB(
        images,
        climb=False,
        parallel=False,
        method="improvedtangent",
        allow_shared_calculator=False,
    )

    # --------------------------------------------------------
    # 用中心差分替换 improved tangent 的 get_tangent，
    # 但保留 improved-tangent 的投影和弹簧公式。
    # --------------------------------------------------------
    install_central_difference_improved_tangent(
        neb
    )

    # --------------------------------------------------------
    # 自适应 ODE optimizer
    #
    # 不指定 alpha、rtol、C1、C2：
    # 使用 ASE 默认值。
    #
    # logfile=None：
    # 不打印 ASE optimizer 自带日志；
    # 我们使用下面的 record_state 打印自定义日志。
    # --------------------------------------------------------
    optimizer = NEBOptimizer(
        neb,
        method="ode",
        logfile=None,
        trajectory=None,
        verbose=0,
    )

    # ========================================================
    # 保存序列
    # ========================================================

    # 与 IVM 对比的主要横坐标：
    # 实际整条路径 gradient sweep 次数减去初始 sweep
    count_seq = []

    # ODE accepted step
    optimizer_step_seq = []

    # 单 image restrained-MD 调用总数
    image_md_calls_seq = []

    # 原始完整路径 gradient sweep 次数，包含初始 sweep
    path_gradient_sweeps_raw_seq = []

    # 完整路径：(N+2, 2)
    images_seq = []

    # 两个与 IVM 一致的整路径指标
    I_perp_seq = []
    I_perp_euclid_arc_seq = []

    # 诊断
    fmaxs = []
    fmax_neb_physical_seq = []
    neb_total_fmax_seq = []

    # Reference-path oracle 数据
    M_ref_seq = []
    F_ref_seq = []
    Mg_ref_seq = []

    # 投影量
    perpG_Mg_seq = []
    perpE_Mg_seq = []

    # 积分权重
    node_weights_G_seq = []
    node_weights_E_seq = []

    # G 投影矩阵
    P_perp_G_seq = []

    last_recorded_step = {
        "value": None,
    }

    start_time = time.perf_counter()

    # ========================================================
    # 保存当前已有序列
    # ========================================================
    def save_current_npz():
        np.savez(
            result_file,

            # ------------------------------------------------
            # 横坐标
            # ------------------------------------------------
            count_seq=np.asarray(
                count_seq,
                dtype=float,
            ),

            optimizer_step_seq=np.asarray(
                optimizer_step_seq,
                dtype=int,
            ),

            image_md_calls_seq=np.asarray(
                image_md_calls_seq,
                dtype=int,
            ),

            path_gradient_sweeps_raw_seq=np.asarray(
                path_gradient_sweeps_raw_seq,
                dtype=float,
            ),

            # ------------------------------------------------
            # 路径
            # shape:
            # (n_saved, N+2, 2)
            # ------------------------------------------------
            images_seq=np.asarray(
                images_seq,
                dtype=float,
            ),

            # ------------------------------------------------
            # 与 IVM 一致的指标
            # ------------------------------------------------
            I_perp_seq=np.asarray(
                I_perp_seq,
                dtype=float,
            ),

            I_perp_euclid_arc_seq=np.asarray(
                I_perp_euclid_arc_seq,
                dtype=float,
            ),

            # ------------------------------------------------
            # fmax
            # ------------------------------------------------
            fmaxs=np.asarray(
                fmaxs,
                dtype=float,
            ),

            fmax_neb_physical_seq=np.asarray(
                fmax_neb_physical_seq,
                dtype=float,
            ),

            neb_total_fmax_seq=np.asarray(
                neb_total_fmax_seq,
                dtype=float,
            ),

            # ------------------------------------------------
            # Reference M、F 和 M F
            # ------------------------------------------------
            M_ref_seq=np.asarray(
                M_ref_seq,
                dtype=float,
            ),

            F_ref_seq=np.asarray(
                F_ref_seq,
                dtype=float,
            ),

            Mg_ref_seq=np.asarray(
                Mg_ref_seq,
                dtype=float,
            ),

            # ------------------------------------------------
            # 投影 residual
            # ------------------------------------------------
            perpG_Mg_seq=np.asarray(
                perpG_Mg_seq,
                dtype=float,
            ),

            perpE_Mg_seq=np.asarray(
                perpE_Mg_seq,
                dtype=float,
            ),

            # ------------------------------------------------
            # 积分权重
            # ------------------------------------------------
            node_weights_G_seq=np.asarray(
                node_weights_G_seq,
                dtype=float,
            ),

            node_weights_E_seq=np.asarray(
                node_weights_E_seq,
                dtype=float,
            ),

            P_perp_G_seq=np.asarray(
                P_perp_G_seq,
                dtype=float,
            ),

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------
            k_spring=np.asarray(
                [ASE_DEFAULT_K],
                dtype=float,
            ),

            optimizer_method=np.asarray(
                ["ode"],
                dtype=str,
            ),

            neb_method=np.asarray(
                ["improvedtangent_central_difference"],
                dtype=str,
            ),

            N_internal=np.asarray(
                [N],
                dtype=int,
            ),

            endpoints=np.stack(
                [C1, C2],
                axis=0,
            ),
        )

    # ========================================================
    # 记录当前 ODE accepted state
    # ========================================================
    def record_state():
        step_index = int(
            optimizer.nsteps
        )

        # 同一个 accepted step 不重复保存
        if (
            last_recorded_step["value"]
            == step_index
        ):
            return

        # ----------------------------------------------------
        # 计算当前 NEB force。
        #
        # 如果当前位置已被 ODE force_function 计算过，
        # ASE calculator cache 会复用结果，不一定产生新的 MD。
        # ----------------------------------------------------
        neb_forces = neb.get_forces()

        path_full = images_to_path(
            images
        )

        (
            M_all,
            F_all,
            Mg_all_cached,
        ) = collect_internal_oracle_data(
            images
        )

        metrics = compute_integrated_metrics(
            path_full,
            M_all,
            F_all,
        )

        # ----------------------------------------------------
        # ASE 总 NEB force：
        # 包含 physical perpendicular force 和 spring force
        # ----------------------------------------------------
        neb_force_rows = np.asarray(
            neb_forces,
            dtype=float,
        ).reshape(-1, 3)

        neb_total_fmax = float(
            np.max(
                np.linalg.norm(
                    neb_force_rows,
                    axis=1,
                )
            )
        )

        # ----------------------------------------------------
        # 实际完整路径 gradient sweep 次数
        #
        # 一次 sweep = N 次 image M_gradient 调用
        # ----------------------------------------------------
        path_sweeps_raw = (
            GLOBAL_IMAGE_MD_CALLS
            / float(N)
        )

        # ----------------------------------------------------
        # 与 IVM count_seq 的 convention 对齐：
        #
        # 初始路径：
        #     count_seq = 0
        #
        # 因此减去初始化所需的第一次整路径 sweep。
        #
        # ODE 内部若为了接受一步多次调用 force，
        # 这些额外 oracle 调用会真实反映在 count_seq 中。
        # ----------------------------------------------------
        count_value = max(
            path_sweeps_raw - 1.0,
            0.0,
        )

        count_seq.append(
            count_value
        )

        optimizer_step_seq.append(
            step_index
        )

        image_md_calls_seq.append(
            GLOBAL_IMAGE_MD_CALLS
        )

        path_gradient_sweeps_raw_seq.append(
            path_sweeps_raw
        )

        images_seq.append(
            path_full.copy()
        )

        I_perp_seq.append(
            metrics["I_perp_G"]
        )

        I_perp_euclid_arc_seq.append(
            metrics["I_perp_E"]
        )

        fmaxs.append(
            metrics[
                "fmax_Gperp_Euclidean"
            ]
        )

        fmax_neb_physical_seq.append(
            metrics[
                "fmax_NEB_physical"
            ]
        )

        neb_total_fmax_seq.append(
            neb_total_fmax
        )

        M_ref_seq.append(
            M_all.copy()
        )

        F_ref_seq.append(
            F_all.copy()
        )

        Mg_ref_seq.append(
            metrics["Mg_all"].copy()
        )

        perpG_Mg_seq.append(
            metrics["perpG_Mg"].copy()
        )

        perpE_Mg_seq.append(
            metrics["perpE_Mg"].copy()
        )

        node_weights_G_seq.append(
            metrics[
                "node_weights_G"
            ].copy()
        )

        node_weights_E_seq.append(
            metrics[
                "node_weights_E"
            ].copy()
        )

        P_perp_G_seq.append(
            metrics[
                "P_perp_G_all"
            ].copy()
        )

        last_recorded_step["value"] = (
            step_index
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print(
            f"[NEB-ODE] "
            f"accepted_step={step_index:4d}  "
            f"count_seq={count_value:.1f}  "
            f"I_perp_G={metrics['I_perp_G']:.6e}  "
            f"I_perp_E={metrics['I_perp_E']:.6e}  "
            f"fmax_Gperp_E="
            f"{metrics['fmax_Gperp_Euclidean']:.6e}  "
            f"fmax_NEB_physical="
            f"{metrics['fmax_NEB_physical']:.6e}  "
            f"ASE_total_fmax={neb_total_fmax:.6e}  "
            f"image_MD_calls={GLOBAL_IMAGE_MD_CALLS}  "
            f"elapsed={elapsed:.1f}s"
        )

        # ----------------------------------------------------
        # 保存图片，不显示
        # ----------------------------------------------------
        if (
            SAVE_FIGS
            and (
                step_index == 0
                or step_index % SAVE_FIG_EVERY == 0
                or step_index == MAX_NEB_STEPS
            )
        ):
            try:
                fig_save(
                    step_index,
                    path_full[1:-1],
                    path_full[1:-1],
                    None,
                    path_full[1:-1],
                    xlim=xlim,
                    ylim=ylim,

                    # G-normal M grad F
                    force1=-metrics[
                        "perpG_Mg"
                    ],

                    # 原始 nabla F
                    force2=-F_all,

                    # ASE 实际 Euclidean-normal M grad F
                    force3=-metrics[
                        "perpE_Mg"
                    ],

                    figure_path=figure_path,
                )

            except Exception as exc:
                print(
                    f"[WARNING] fig_save failed "
                    f"at accepted step {step_index}: {exc}"
                )

            plt.close("all")

        if SAVE_EVERY_STATE:
            save_current_npz()

    # ========================================================
    # 首先保存初始路径
    #
    # 这会触发初始 N 个 image 的 restrained MD。
    # 保存时 count_seq 定义为 0。
    # ========================================================
    record_state()

    # 每个 ODE accepted step 记录一次
    optimizer.attach(
        record_state,
        interval=1,
    )

    # ========================================================
    # 执行 ASE adaptive ODE NEB
    # ========================================================
    converged = optimizer.run(
        fmax=NEB_FMAX_TOL,
        steps=MAX_NEB_STEPS,
    )

    # 防止最终状态尚未记录
    record_state()

    # 最终保存
    save_current_npz()

    elapsed_total = (
        time.perf_counter()
        - start_time
    )

    with open(
        done_file,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "completed\n"
            f"converged={converged}\n"
            f"optimizer=ode\n"
            f"k={ASE_DEFAULT_K}\n"
            f"accepted_steps={optimizer.nsteps}\n"
            f"image_md_calls={GLOBAL_IMAGE_MD_CALLS}\n"
            f"path_gradient_sweeps="
            f"{GLOBAL_IMAGE_MD_CALLS / float(N)}\n"
            f"elapsed_seconds={elapsed_total}\n"
        )

    print("\n" + "=" * 80)
    print("ASE NEB ODE finished")
    print("=" * 80)
    print(f"Converged              : {converged}")
    print(f"Accepted ODE steps     : {optimizer.nsteps}")
    print(f"Image MD calls         : {GLOBAL_IMAGE_MD_CALLS}")
    print(
        f"Path gradient sweeps   : "
        f"{GLOBAL_IMAGE_MD_CALLS / float(N):.1f}"
    )
    print(f"Elapsed time (seconds) : {elapsed_total:.1f}")
    print(f"Result file            : {result_file}")
    print(f"Figure folder          : {figure_path}")
    print("=" * 80)


# ============================================================
# 主程序
# ============================================================
def main():
    run_neb_experiment()


if __name__ == "__main__":
    main()