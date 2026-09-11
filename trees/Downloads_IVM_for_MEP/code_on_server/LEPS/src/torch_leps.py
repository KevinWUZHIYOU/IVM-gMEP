"""Pure-PyTorch LEPS potential used by the direct-autodiff gMAM run."""

from __future__ import annotations

import torch

from leps_core import (
    A_LEPS,
    B_LEPS,
    C_LEPS,
    D_AB,
    D_AC,
    D_BC,
    MORSE_ALPHA,
    R0,
)


def _q(r, dissociation):
    displacement = r - float(R0)
    exp_one = torch.exp(-float(MORSE_ALPHA) * displacement)
    exp_two = exp_one * exp_one
    return 0.5 * float(dissociation) * (1.5 * exp_two - exp_one)


def _j(r, dissociation):
    displacement = r - float(R0)
    exp_one = torch.exp(-float(MORSE_ALPHA) * displacement)
    exp_two = exp_one * exp_one
    return 0.25 * float(dissociation) * (exp_two - 6.0 * exp_one)


def torch_leps_energy(points: torch.Tensor) -> torch.Tensor:
    """Return the historical LEPS energy for a tensor with final dimension 2."""
    if points.shape[-1] != 2:
        raise ValueError("points must have final dimension 2")
    r_ab = points[..., 0]
    r_bc = points[..., 1]
    r_ac = r_ab + r_bc

    q_ab = _q(r_ab, D_AB)
    q_bc = _q(r_bc, D_BC)
    q_ac = _q(r_ac, D_AC)
    j_ab = _j(r_ab, D_AB)
    j_bc = _j(r_bc, D_BC)
    j_ac = _j(r_ac, D_AC)

    pa = 1.0 + float(A_LEPS)
    pb = 1.0 + float(B_LEPS)
    pc = 1.0 + float(C_LEPS)
    radicand = (
        j_ab**2 / pa**2
        + j_bc**2 / pb**2
        + j_ac**2 / pc**2
        - j_ab * j_bc / (pa * pb)
        - j_bc * j_ac / (pb * pc)
        - j_ab * j_ac / (pa * pc)
    )
    return (
        q_ab / pa
        + q_bc / pb
        + q_ac / pc
        - torch.sqrt(torch.clamp(radicand, min=0.0))
    )


def torch_leps_energy_force(
    points: torch.Tensor,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return energy and force, retaining the Hessian graph when requested."""
    if not points.requires_grad:
        raise ValueError("points must require gradients")
    energy = torch_leps_energy(points)
    gradient = torch.autograd.grad(
        energy.sum(),
        points,
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    return energy, -gradient
