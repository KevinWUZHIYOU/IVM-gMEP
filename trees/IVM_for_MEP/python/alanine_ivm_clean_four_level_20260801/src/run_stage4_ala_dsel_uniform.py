#!/usr/bin/env python3
"""Canonical entry point for the Stage-IV Ala DSEL accelerator campaign."""

from run_stage4_discrete_b040_uniform import (  # noqa: F401
    EXPECTED_IMPLEMENTATION_ID,
    POLICY_VERSION,
    assert_frozen_outer_policy,
    command_options,
    command_value,
    main,
    phase_a_command,
    phase_b_command,
    runner_base,
)


if __name__ == "__main__":
    main()
