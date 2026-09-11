#!/usr/bin/env python3
"""Canonical entry point for standalone discrete-coordinate Ala IVM v2."""

from run_stage4_ala_dsel_single_fixedalpha import (  # noqa: F401,F403
    COMMON_INITIAL,
    DSEL_COMMAND_OPTIONS,
    EXPECTED_IMPLEMENTATION_ID,
    POLICY_VERSION,
    REPARAM_POLICY_ID,
    STOP_RULE_ID,
    assert_matched_pair_commands,
    assert_single_run_policy,
    command_options,
    command_value,
    complete_candidate_calls,
    main,
    run_label,
    single_run_command,
)


if __name__ == "__main__":
    main()
