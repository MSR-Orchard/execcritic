"""Parse actions & format observations with toolcalls"""

import json
import time

from jinja2 import StrictUndefined, Template

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}

WRITE_GENERATED_TEST_TOOL_NAME = "write_generated_test"
WRITE_GENERATED_TEST_ALLOWED_FIELDS = (
    "test_code",
    "test_file_path",
    "test_command",
    "command_evidence",
    "notes",
    "oracle",
)
WRITE_GENERATED_TEST_TOOL = {
    "type": "function",
    "function": {
        "name": WRITE_GENERATED_TEST_TOOL_NAME,
        "description": (
            "Write the full generated regression test source and its repository-inferred execution contract. "
            "Choose the test path and exact focused command from repository evidence; the harness will run it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "test_code": {
                    "type": "string",
                    "description": "The complete Python source for the generated regression test file.",
                },
                "test_file_path": {
                    "type": "string",
                    "description": (
                        "New, currently nonexistent test path relative to /testbed, inferred from repository conventions."
                    ),
                },
                "test_command": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Exact repository-native command that runs only the generated test.",
                },
                "command_evidence": {
                    "type": "string",
                    "description": (
                        "Repository evidence supporting the path and command, such as CI/config files or "
                        "adjacent tests."
                    ),
                },
                "notes": {
                    "type": "string",
                    "description": "Optional short note about what behavior the test covers.",
                },
                "oracle": {
                    "type": "object",
                    "description": (
                        "Optional structured oracle contract: the public entrypoint, public integration path, "
                        "triggering input x, expected observable output y, evidence for y, and non-assertions."
                    ),
                    "properties": {
                        "entrypoint": {
                            "type": "string",
                            "description": "Public API, command, object, or documented behavior under test.",
                        },
                        "public_integration_path": {
                            "type": "string",
                            "description": (
                                "Concrete user-facing path used by the test, such as public API -> object "
                                "method, CLI command, framework runner, documented workflow, or adjacent "
                                "integration-test pattern. Do not name private helpers or implementation modules."
                            ),
                        },
                        "trigger_input": {
                            "type": "string",
                            "description": "The input/state x that triggers the bug behavior.",
                        },
                        "expected_output": {
                            "type": "string",
                            "description": "The expected observable output/behavior y for a correct implementation.",
                        },
                        "evidence": {
                            "type": "string",
                            "description": (
                                "Why y is correct, e.g. issue_explicit, docs, existing_tests, or public_invariant."
                            ),
                        },
                        "non_assertions": {
                            "type": "string",
                            "description": "Implementation details or output fragments intentionally not pinned.",
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "required": ["test_code"],
            "additionalProperties": False,
        },
    },
}

SUBMIT_GENERATED_TEST_TOOL_NAME = "submit_generated_test"
SUBMIT_GENERATED_TEST_TOOL = {
    "type": "function",
    "function": {
        "name": SUBMIT_GENERATED_TEST_TOOL_NAME,
        "description": (
            "Submit the latest generated-test candidate after reviewing its harness-run self-test result. "
            "The execution_hash must match the exact latest test contents, path, and focused command."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "execution_hash": {
                    "type": "string",
                    "description": "Execution hash reported by the self-test feedback for the candidate to submit.",
                }
            },
            "required": ["execution_hash"],
            "additionalProperties": False,
        },
    },
}

GROUNDING_PLAN_TOOL_NAME = "grounding_plan"
GROUNDING_PLAN_REQUIRED_FIELDS = (
    "issue_summary",
    "public_entrypoint",
    "public_integration_path",
    "trigger_input",
    "expected_output",
    "evidence",
    "test_strategy",
    "exploration_plan",
)
GROUNDING_PLAN_OPTIONAL_FIELDS = (
    "candidate_patch_hypothesis",
    "candidate_blind_spots",
    "discriminating_trigger_inputs",
    "why_this_test_should_distinguish_patch",
)
GROUNDING_PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": GROUNDING_PLAN_TOOL_NAME,
        "description": (
            "Record the initial generated-test grounding plan before any repository inspection or test writing. "
            "Use best current hypotheses from the issue, and state unknowns you will resolve by inspection."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "issue_summary": {
                    "type": "string",
                    "description": "Short summary of the bug behavior described by the issue.",
                },
                "public_entrypoint": {
                    "type": "string",
                    "description": (
                        "Likely public API, command, object, or documented behavior to test. "
                        "If uncertain, name the candidate and what must be inspected."
                    ),
                },
                "public_integration_path": {
                    "type": "string",
                    "description": (
                        "Candidate public integration path from a user-facing API, command, framework runner, "
                        "documented object, or adjacent-test workflow to the issue behavior."
                    ),
                },
                "trigger_input": {
                    "type": "string",
                    "description": "The input/state x that should trigger the bug.",
                },
                "expected_output": {
                    "type": "string",
                    "description": "The expected observable output/behavior y for a correct fix.",
                },
                "evidence": {
                    "type": "string",
                    "description": "Why y is believed correct: issue text, docs, existing tests, or public invariant.",
                },
                "candidate_patch_hypothesis": {
                    "type": "string",
                    "description": (
                        "For patch-conditioned generation, what behavior this candidate patch appears to fix "
                        "or risk breaking, stated without using the patch text as the oracle."
                    ),
                },
                "candidate_blind_spots": {
                    "type": "string",
                    "description": (
                        "For patch-conditioned generation, issue-backed edge cases, branches, or variants the "
                        "candidate patch may miss or overfit."
                    ),
                },
                "discriminating_trigger_inputs": {
                    "type": "string",
                    "description": (
                        "Concrete issue-backed input(s) or state(s) that should distinguish a correct patch "
                        "from this candidate patch."
                    ),
                },
                "why_this_test_should_distinguish_patch": {
                    "type": "string",
                    "description": (
                        "Why the planned public-behavior test should pass a correct patch but fail an incomplete "
                        "or overfit candidate patch."
                    ),
                },
                "test_strategy": {
                    "type": "string",
                    "description": "How the generated test will exercise behavior without relying on implementation details.",
                },
                "exploration_plan": {
                    "type": "string",
                    "description": "Specific files/APIs/tests to inspect next to validate imports, fixtures, and x->y mapping.",
                },
                "non_assertions": {
                    "type": "string",
                    "description": "Implementation details, exact wording, or incidental values that should not be pinned.",
                },
            },
            "required": list(GROUNDING_PLAN_REQUIRED_FIELDS),
            "additionalProperties": False,
        },
    },
}

# --- Resolve-style test_patch tools (RESOLVE_STYLE_TESTPATCH_DESIGN.md) ---
WRITE_TEST_PATCH_TOOL_NAME = "write_test_patch"
WRITE_TEST_PATCH_ALLOWED_FIELDS = ("test_command",)
WRITE_TEST_PATCH_TOOL = {
    "type": "function",
    "function": {
        "name": WRITE_TEST_PATCH_TOOL_NAME,
        "description": (
            "Submit your current edits. FIRST edit the repository files directly in the workspace with bash "
            "(sed/python/cat) — you may change tests, fixtures, or product source. THEN call this tool: the "
            "harness runs `git diff` to capture your edits as a patch, applies it onto an immutable buggy Base, "
            "executes your focused test_command in the repo's official environment, and reports whether it "
            "cleanly fails. Do NOT paste a diff; the harness derives it from your workspace edits."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "test_command": {
                    "type": "string",
                    "description": (
                        "Exact focused command to execute after applying the patch, chosen from repository "
                        "evidence and validated during exploration."
                    ),
                },
            },
            "required": ["test_command"],
            "additionalProperties": False,
        },
    },
}

SUBMIT_TEST_PATCH_TOOL_NAME = "submit_test_patch"
SUBMIT_TEST_PATCH_TOOL = {
    "type": "function",
    "function": {
        "name": SUBMIT_TEST_PATCH_TOOL_NAME,
        "description": (
            "Finalize and submit the latest written test_patch as your answer, after reviewing its per-node "
            "harness feedback. The patch_hash must match the latest write_test_patch result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patch_hash": {
                    "type": "string",
                    "description": "The patch_hash reported by the latest write_test_patch feedback.",
                }
            },
            "required": ["patch_hash"],
            "additionalProperties": False,
        },
    },
}

EXTRA_TOOLS = {
    GROUNDING_PLAN_TOOL_NAME: GROUNDING_PLAN_TOOL,
    WRITE_GENERATED_TEST_TOOL_NAME: WRITE_GENERATED_TEST_TOOL,
    SUBMIT_GENERATED_TEST_TOOL_NAME: SUBMIT_GENERATED_TEST_TOOL,
    WRITE_TEST_PATCH_TOOL_NAME: WRITE_TEST_PATCH_TOOL,
    SUBMIT_TEST_PATCH_TOOL_NAME: SUBMIT_TEST_PATCH_TOOL,
}


def tools_for_names(extra_tools: list[str] | None = None) -> list[dict]:
    tools = [BASH_TOOL]
    for name in extra_tools or []:
        tool = EXTRA_TOOLS.get(str(name).strip())
        if tool and tool not in tools:
            tools.append(tool)
    return tools


def parse_toolcall_actions(
    tool_calls: list, *, format_error_template: str, extra_tools: list[str] | None = None
) -> list[dict]:
    """Parse tool calls from the response. Raises FormatError if unknown tool or invalid args."""
    allowed_extra_tools = {str(name).strip() for name in extra_tools or []}
    if not tool_calls:
        raise FormatError(
            {
                "role": "user",
                "content": Template(format_error_template, undefined=StrictUndefined).render(
                    error="No tool calls found in the response. Every response MUST include at least one tool call.",
                    actions=[],
                ),
                "extra": {"interrupt_type": "FormatError"},
            }
        )
    actions = []
    for tool_call in tool_calls:
        error_msg = ""
        args = {}
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as e:
            error_msg = f"Error parsing tool call arguments: {e}."
        tool_name = tool_call.function.name
        if tool_name == "bash":
            if not isinstance(args, dict) or "command" not in args:
                error_msg += "Missing 'command' argument in bash tool call."
            if error_msg:
                _raise_format_error(format_error_template, error_msg)
            actions.append({"tool": "bash", "command": args["command"], "tool_call_id": tool_call.id})
            continue
        if tool_name == WRITE_GENERATED_TEST_TOOL_NAME and tool_name in allowed_extra_tools:
            if not isinstance(args, dict) or not isinstance(args.get("test_code"), str):
                error_msg += "Missing string 'test_code' argument in write_generated_test tool call."
            invalid_contract_fields = [
                field
                for field in ("test_file_path", "test_command", "command_evidence")
                if isinstance(args, dict) and field in args and not isinstance(args.get(field), str)
            ]
            if invalid_contract_fields:
                error_msg += f" Execution-contract field(s) must be strings: {', '.join(invalid_contract_fields)}."
            oracle = args.get("oracle") if isinstance(args, dict) else None
            if oracle is not None and not isinstance(oracle, dict):
                error_msg += " Optional 'oracle' argument must be an object."
            if isinstance(args, dict):
                extra_fields = [k for k in args if k not in WRITE_GENERATED_TEST_ALLOWED_FIELDS]
                if extra_fields:
                    error_msg += (
                        f" Unexpected argument(s) {', '.join(sorted(extra_fields))} in write_generated_test tool call; "
                        f"only {', '.join(WRITE_GENERATED_TEST_ALLOWED_FIELDS)} are allowed."
                    )
            if error_msg:
                _raise_format_error(format_error_template, error_msg)
            actions.append(
                {
                    "tool": WRITE_GENERATED_TEST_TOOL_NAME,
                    "test_code": args["test_code"],
                    **{
                        field: args[field]
                        for field in ("test_file_path", "test_command", "command_evidence")
                        if field in args
                    },
                    "notes": args.get("notes", "") if isinstance(args, dict) else "",
                    "oracle": oracle or {},
                    "tool_call_id": tool_call.id,
                }
            )
            continue
        if tool_name == SUBMIT_GENERATED_TEST_TOOL_NAME and tool_name in allowed_extra_tools:
            if not isinstance(args, dict) or not isinstance(args.get("execution_hash"), str):
                error_msg += "Missing string 'execution_hash' argument in submit_generated_test tool call."
            if isinstance(args, dict):
                extra_fields = [key for key in args if key != "execution_hash"]
                if extra_fields:
                    error_msg += (
                        f" Unexpected argument(s) {', '.join(sorted(extra_fields))} in "
                        "submit_generated_test tool call; only execution_hash is allowed."
                    )
            if error_msg:
                _raise_format_error(format_error_template, error_msg)
            actions.append(
                {
                    "tool": SUBMIT_GENERATED_TEST_TOOL_NAME,
                    "execution_hash": args["execution_hash"],
                    "tool_call_id": tool_call.id,
                }
            )
            continue
        if tool_name == WRITE_TEST_PATCH_TOOL_NAME and tool_name in allowed_extra_tools:
            if (
                not isinstance(args, dict)
                or not isinstance(args.get("test_command"), str)
                or not args["test_command"].strip()
            ):
                error_msg += "Missing non-empty string 'test_command' argument in write_test_patch tool call."
            if isinstance(args, dict):
                extra_fields = [k for k in args if k not in WRITE_TEST_PATCH_ALLOWED_FIELDS]
                if extra_fields:
                    error_msg += (
                        f" Unexpected argument(s) {', '.join(sorted(extra_fields))} in write_test_patch tool call; "
                        f"only {', '.join(WRITE_TEST_PATCH_ALLOWED_FIELDS)} are allowed. Do NOT pass a diff; edit "
                        "files with bash first, then call write_test_patch with just test_command."
                    )
            if error_msg:
                _raise_format_error(format_error_template, error_msg)
            actions.append(
                {
                    "tool": WRITE_TEST_PATCH_TOOL_NAME,
                    "test_command": args["test_command"],
                    "tool_call_id": tool_call.id,
                }
            )
            continue
        if tool_name == SUBMIT_TEST_PATCH_TOOL_NAME and tool_name in allowed_extra_tools:
            if not isinstance(args, dict) or not isinstance(args.get("patch_hash"), str):
                error_msg += "Missing string 'patch_hash' argument in submit_test_patch tool call."
            if isinstance(args, dict):
                extra_fields = [key for key in args if key != "patch_hash"]
                if extra_fields:
                    error_msg += (
                        f" Unexpected argument(s) {', '.join(sorted(extra_fields))} in "
                        "submit_test_patch tool call; only patch_hash is allowed."
                    )
            if error_msg:
                _raise_format_error(format_error_template, error_msg)
            actions.append(
                {
                    "tool": SUBMIT_TEST_PATCH_TOOL_NAME,
                    "patch_hash": args["patch_hash"],
                    "tool_call_id": tool_call.id,
                }
            )
            continue
        if tool_name == GROUNDING_PLAN_TOOL_NAME and tool_name in allowed_extra_tools:
            missing = [
                field
                for field in GROUNDING_PLAN_REQUIRED_FIELDS
                if not isinstance(args, dict) or not isinstance(args.get(field), str)
            ]
            if missing:
                error_msg += f"Missing string grounding_plan field(s): {', '.join(missing)}."
            if isinstance(args, dict) and args.get("non_assertions") is not None and not isinstance(
                args.get("non_assertions"), str
            ):
                error_msg += " Optional 'non_assertions' argument must be a string."
            if error_msg:
                _raise_format_error(format_error_template, error_msg)
            actions.append(
                {
                    "tool": GROUNDING_PLAN_TOOL_NAME,
                    **{field: args.get(field, "") for field in GROUNDING_PLAN_REQUIRED_FIELDS},
                    **{
                        field: args.get(field, "")
                        for field in GROUNDING_PLAN_OPTIONAL_FIELDS
                        if isinstance(args.get(field), str)
                    },
                    "non_assertions": args.get("non_assertions", "") if isinstance(args, dict) else "",
                    "tool_call_id": tool_call.id,
                }
            )
            continue
        error_msg += f"Unknown tool '{tool_name}'."
        _raise_format_error(format_error_template, error_msg)
    return actions


def _raise_format_error(format_error_template: str, error_msg: str) -> None:
    raise FormatError(
        {
            "role": "user",
            "content": Template(format_error_template, undefined=StrictUndefined).render(
                actions=[], error=error_msg.strip()
            ),
            "extra": {"interrupt_type": "FormatError"},
        }
    )


def format_toolcall_observation_messages(
    *,
    actions: list[dict],
    outputs: list[dict],
    observation_template: str,
    template_vars: dict | None = None,
    multimodal_regex: str = "",
) -> list[dict]:
    """Format execution outputs into tool result messages."""
    not_executed = {"output": "", "returncode": -1, "exception_info": "action was not executed"}
    padded_outputs = outputs + [not_executed] * (len(actions) - len(outputs))
    results = []
    for action, output in zip(actions, padded_outputs):
        content = Template(observation_template, undefined=StrictUndefined).render(
            output=output, **(template_vars or {})
        )
        msg = {
            "content": content,
            "extra": {
                "raw_output": output.get("output", ""),
                "returncode": output.get("returncode"),
                "timestamp": time.time(),
                "exception_info": output.get("exception_info"),
                **output.get("extra", {}),
            },
        }
        if "tool_call_id" in action:
            msg["tool_call_id"] = action["tool_call_id"]
            msg["role"] = "tool"
        else:
            msg["role"] = "user"  # human issued commands
        if multimodal_regex:
            msg = expand_multimodal_content(msg, pattern=multimodal_regex)
        results.append(msg)
    return results
