#!/usr/bin/env python3

"""This is the central entry point to the mini-extra script. Use subcommands
to invoke other command line utilities like running on benchmarks, editing config,
inspecting trajectories, etc.
"""

import sys
from importlib import import_module

from rich.console import Console

subcommands = [
    ("minisweagent.run.utilities.config", ["config"], "Manage the global config file"),
    ("minisweagent.run.utilities.inspector", ["inspect", "i", "inspector"], "Run inspector (browse trajectories)"),
    ("minisweagent.run.benchmarks.swebench", ["swebench"], "Evaluate on SWE-bench or SWE-bench Pro (batch mode)"),
    ("minisweagent.run.benchmarks.swebench_single", ["swebench-single"], "Evaluate on SWE-bench or SWE-bench Pro (single instance)"),
    ("minisweagent.run.benchmarks.swerebench", ["swerebench"], "Evaluate on SWE-rebench (batch mode)"),
    ("minisweagent.run.benchmarks.gentest", ["gentest"], "Generate regression tests as a native mini-swe-agent benchmark"),
    (
        "minisweagent.run.benchmarks.patch_gentest",
        ["patch-gentest"],
        "Generate patch-conditioned regression tests for patch groups",
    ),
    ("minisweagent.run.benchmarks.swerebench_verify", ["swerebench-verify"], "Verify patches for SWE-rebench"),
    ("minisweagent.run.benchmarks.scaleswe", ["scaleswe"], "Evaluate on Scale-SWE (batch mode)"),
    ("minisweagent.run.benchmarks.swerebench_verify_azure_modal", ["swerebench-verify-azure-modal"], "Verify trajectory patches for SWE-rebench via Azure Modal"),
    ("minisweagent.run.benchmarks.swebench_verify_azure_modal", ["swebench-verify-azure-modal"], "Verify trajectory patches for SWE-bench via Azure Modal"),
    ("minisweagent.run.benchmarks.swebench_verify_azure_modal_cli", ["swebench-verify-azure-modal-cli"], "Verify trajectory patches for SWE-bench via Azure Modal (uses swebench harness)"),
    ("minisweagent.run.benchmarks.swebench_multilingual_verify_azure_modal", ["swebench-multilingual-verify-azure-modal"], "Verify trajectory patches for SWE-bench Multilingual via Azure Modal"),
    ("minisweagent.run.benchmarks.scaleswe_verify_azure_modal", ["scaleswe-verify-azure-modal"], "Verify trajectory patches for Scale-SWE via Azure Modal"),
]


def get_docstring() -> str:
    lines = [
        "This is the [yellow]central entry point for all extra commands[/yellow] from mini-swe-agent.",
        "",
        "Available sub-commands:",
        "",
    ]
    for _, aliases, description in subcommands:
        alias_text = " or ".join(f"[bold green]{alias}[/bold green]" for alias in aliases)
        lines.append(f"  {alias_text}: {description}")
    return "\n".join(lines)


def main():
    args = sys.argv[1:]

    if len(args) == 0 or len(args) == 1 and args[0] in ["-h", "--help"]:
        return Console().print(get_docstring())

    for module_path, aliases, _ in subcommands:
        if args[0] in aliases:
            return import_module(module_path).app(args[1:], prog_name=f"mini-extra {aliases[0]}")

    return Console().print(get_docstring())


if __name__ == "__main__":
    main()
