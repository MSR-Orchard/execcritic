"""
Mini-SWE-Agent integration for Slime RL training.

This package integrates mini-swe-agent with slime's RL training framework,
enabling training on SWE-bench coding tasks.

Main components:
- swe_wrapper: Adapter between mini-swe-agent and slime
- swe_generate: Custom generate function for SWE tasks
- swe_reward: Reward computation for SWE tasks

Usage:
    python train.py \\
        --custom-generate-function-path slime.examples.mini_swe.swe_generate.generate \\
        --custom-rm-path slime.examples.mini_swe.swe_reward.reward_func \\
        --yaml-path slime/examples/mini_swe/configs/slime_train.yaml
"""

__version__ = "0.1.0"

__all__ = [
    "generate",
    "reward_func",
    "AgentConfig",
    "SWEAgent",
    "create_environment",
    "load_config",
]


def __getattr__(name):
    if name == "generate":
        from .swe_generate import generate

        return generate
    if name == "reward_func":
        from .swe_reward import reward_func

        return reward_func
    if name in {"AgentConfig", "SWEAgent", "create_environment", "load_config"}:
        from . import swe_wrapper

        return getattr(swe_wrapper, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
