import numpy as np

try:
    # 当前 ASE 路径
    from ase.mep.neb import ImprovedTangentMethod
except ImportError:
    # 兼容较旧 ASE
    from ase.neb import ImprovedTangentMethod


class CentralDifferenceImprovedTangentMethod(
    ImprovedTangentMethod
):
    """
    使用中心差分 tangent，但保留 ASE ImprovedTangentMethod
    的 NEB 力投影和切向弹簧力。

    对第 i 个内部 image：

        tangent_i
        = phi_{i+1} - phi_{i-1}

        = (phi_i - phi_{i-1})
          + (phi_{i+1} - phi_i)

        = spring1.t + spring2.t

    这与 IVM 中的：

        X_full[2:] - X_full[:-2]

    完全一致。

    从 ImprovedTangentMethod 继承：

        adjust_positions()
        add_image_force()

    因此兼容 ASE BaseNEB.set_positions() 和 NEBOptimizer(method="ode")。
    """

    def __init__(self, neb):
        super().__init__(neb)

    def get_tangent(
        self,
        state,
        spring1,
        spring2,
        i,
    ):
        """
        返回归一化的中心差分 tangent。

        Parameters
        ----------
        state
            ASE NEBState。中心差分不使用能量信息。

        spring1
            image i-1 到 image i 的 spring。

        spring2
            image i 到 image i+1 的 spring。

        i
            当前内部 image 的编号。

        Returns
        -------
        tangent : np.ndarray
            与单个 ASE image positions 相同的形状，
            通常为 (natoms, 3)。
        """
        tangent = (
            np.asarray(
                spring1.t,
                dtype=float,
            ).copy()
            +
            np.asarray(
                spring2.t,
                dtype=float,
            ).copy()
        )

        norm = np.linalg.norm(
            tangent
        )

        if norm > 1e-16:
            tangent /= norm
        else:
            # 极端情况下中心差分为零。
            # 优先退化到 forward spring。
            tangent = np.asarray(
                spring2.t,
                dtype=float,
            ).copy()

            norm = np.linalg.norm(
                tangent
            )

            if norm <= 1e-16:
                # 若 forward spring 也为零，
                # 再退化到 backward spring。
                tangent = np.asarray(
                    spring1.t,
                    dtype=float,
                ).copy()

                norm = np.linalg.norm(
                    tangent
                )

            if norm > 1e-16:
                tangent /= norm
            else:
                # 三个 image 完全重合时返回零向量，
                # 避免除零产生 NaN。
                tangent[...] = 0.0

        return tangent


def install_central_difference_improved_tangent(neb):
    """
    将 ASE NEB 的 tangent method 替换为中心差分版本。

    只修改 tangent 的计算方式：

        tangent_i = phi_{i+1} - phi_{i-1}

    其他内容继续使用 ASE ImprovedTangentMethod：

        - 去除真实力的切向分量；
        - 加入 improved NEB 切向弹簧力；
        - adjust_positions 默认保持 positions 不变。

    Parameters
    ----------
    neb
        已创建的 ASE NEB 对象。

    Returns
    -------
    neb
        修改后的同一个 NEB 对象。

    Example
    -------
    neb = NEB(
        images,
        climb=False,
        parallel=False,
        method="improvedtangent",
    )

    install_central_difference_improved_tangent(neb)
    """
    neb.neb_method = (
        CentralDifferenceImprovedTangentMethod(
            neb
        )
    )

    return neb