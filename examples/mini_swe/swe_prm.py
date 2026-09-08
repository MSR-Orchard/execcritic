"""
LLM-as-a-Judge process-reward grader for mini-swe trajectories.

Grades a *whole* SWE-bench-style agent trajectory on PROCESS QUALITY (independent
of whether tests ultimately passed) along 6 rubric dimensions + 2 integrity flags,
then aggregates the per-dimension scores into a single scalar in [0, 1].

Trajectory format (one file per instance):
    {rollouts_dir}/{instance_id}/{instance_id}.traj.json
    {
      "info": {"exit_status": "Submitted", "submission": "<final patch>",
               "model_stats": {"api_calls": N}, ...},
      "messages": [ {role: system}, {role: user, content: <task>},
                    {role: assistant, reasoning_content, tool_calls, extra.actions},
                    {role: tool, content: "<returncode>..</returncode><output>..</output>"},
                    ... , {role: exit, content: <final patch>} ],
      "instance_id": "..."
    }

Usage (grade a directory of rollouts):
    export PRM_API_KEY=...                 # required unless --dry-run
    export PRM_MODEL=gpt-4o-mini           # or your Azure deployment name
    # OpenAI-compatible endpoint (vLLM / OpenAI / proxy):
    export PRM_API_BASE_URL=https://api.openai.com/v1
    # OR Azure OpenAI:
    #   export PRM_API_TYPE=azure
    #   export PRM_API_BASE_URL=https://<resource>.openai.azure.com
    #   export PRM_API_VERSION=2024-08-01-preview

    python -m examples.mini_swe.ideas.process_grader.swe_prm \
        --rollouts-dir /path/to/rollouts \
        --out examples/mini_swe/ideas/process_grader/grades_iter1549.jsonl \
        --limit 10

    # Inspect the prompt without calling the API:
    python -m ...swe_prm --rollouts-dir <dir> --limit 1 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("swe_prm")

# ---------------------------------------------------------------------------
# Rubric / aggregation configuration
# ---------------------------------------------------------------------------

# Per-dimension weights (sum to 1.0). D1/D5 carry the most signal because they
# are closest to "did the agent actually work toward a correct fix".
# D2 (Bug Localization) and D4 (Action Non-Redundancy) were removed (Option-A):
# D2 was too noisy without a gold-patch oracle, D4 was too easily confused with
# legitimate exploration. Their weight was redistributed to D1/D5/D3/D6.
DIM_WEIGHTS: dict[str, float] = {
    "D1": 0.40,  # Goal-Directed Progress
    "D3": 0.20,  # Observation Grounding
    "D5": 0.30,  # Test-Driven Verification
    "D6": 0.10,  # Edit Quality & Minimality
}

# Allowed discrete values per dimension (used to clamp/validate judge output).
# All dims now share the same 5-level scale to maximize J's discrimination.
DIM_ALLOWED: dict[str, list[float]] = {
    "D1": [0.0, 0.25, 0.5, 0.75, 1.0],
    "D3": [0.0, 0.25, 0.5, 0.75, 1.0],
    "D5": [0.0, 0.25, 0.5, 0.75, 1.0],
    "D6": [0.0, 0.25, 0.5, 0.75, 1.0],
}

# If an integrity flag fires, the aggregate is capped to this value.
FLAG_CAP = 0.1

SYSTEM_PROMPT = """You are a senior software engineer acting as a strict, calibrated grader.
You will evaluate an AI coding agent's ENTIRE trajectory on a SWE-bench style
bug-fix task. The agent runs bash commands in a sandbox to localize and fix a
bug, then (ideally) submits a patch.

Grade the PROCESS QUALITY — how soundly the agent worked — as much as possible
INDEPENDENT of whether tests ultimately passed. A lucky pass with sloppy process
should score low; a rigorous process that just missed should score high.

Be calibrated and resist grade inflation: reserve 1.0 for clearly excellent,
use 0.5 for genuinely mixed, 0.0 for absent/bad. Score ONLY with the allowed
discrete values, using the anchors below. Every dimension uses the same
5-level scale: [0.0, 0.25, 0.5, 0.75, 1.0].

If an [ORACLE] block is provided in the user message, you may compare the
agent's edited files / tests against the reference (gold) patch for D5 and D6.
Use it as grounding only — still grade PROCESS, not exact-match of edits.

D1 Goal-Directed Progress [0.0, 0.25, 0.5, 0.75, 1.0]
   1.0  consistently advanced a coherent hypothesis, refined by evidence.
   0.75 mostly on-track, minor detours quickly corrected.
   0.5  mixed: real progress but notable wandering or dead-ends.
   0.25 mostly drifting / clung to a wrong hypothesis despite evidence.
   0.0  never approached the actual problem.

D3 Observation Grounding [0.0, 0.25, 0.5, 0.75, 1.0]
   1.0  actions consistently build on concrete prior outputs (paths, errors, line numbers).
   0.75 mostly grounded; a few actions were speculative but not contradictory.
   0.5  partial: grounded sometimes, ignored evidence other times.
   0.25 mostly ungrounded; occasionally used real outputs.
   0.0  repeatedly ignored or contradicted what observations showed.

D5 Test-Driven Verification [0.0, 0.25, 0.5, 0.75, 1.0]
   1.0  reproduced the bug AND ran the RELEVANT failing test(s) to confirm the fix before submitting.
   0.75 ran the relevant tests but only after the fix (no upfront reproduction), OR
        reproduced + ran one of multiple relevant tests.
   0.5  ran some tests but not the ones relevant to this bug, or verified only partially.
   0.25 ran a token test (e.g. import / smoke) clearly unrelated to the bug.
   0.0  submitted without running any relevant test (or never verified at all).
   Note: running an arbitrary unrelated test does NOT count as verification.
   If an [ORACLE] gold_test_files list is given, "relevant" = those files.

D6 Edit Quality & Minimality [0.0, 0.25, 0.5, 0.75, 1.0]
   1.0  final patch is focused, minimal, well-formed; no unrelated changes;
        if oracle present, edits the SAME file(s) as the gold patch.
   0.75 mostly minimal; touches the right file(s) but includes a couple of
        unnecessary edits OR misses one peripheral file.
   0.5  partially correct area: edits at least one gold file but also unrelated
        files, OR edits a plausible nearby file but misses the gold one.
   0.25 edits in wrong file(s) but plausibly close; or correct file but
        sprawling / malformed changes.
   0.0  sprawling, malformed, unrelated changes, OR no usable edit produced.
   If the trajectory was truncated with no submission, grade the edits actually
   attempted (do not auto-assign 0.0 for lack of a final submission alone).

FLAGS (boolean):
- harmful: destructive without necessity (rm -rf source, git reset --hard
  discarding own fix, reverting own patch, or EDITING/DELETING TESTS to pass).
- hallucinated: invents non-existent files, symbols, APIs, or imagined output.
If EITHER flag is true, the trajectory has a serious integrity violation and
should be scored very low regardless of other dimensions.

Think first, then score. Output ONLY a JSON object, no other text, with keys in
THIS order. Do NOT include D2 or D4 (they were intentionally removed):
{
  "reasoning": "<=3 short sentences citing concrete evidence from the trajectory",
  "harmful": <bool>,
  "hallucinated": <bool>,
  "scores": {"D1": <v>, "D3": <v>, "D5": <v>, "D6": <v>}
}
"""

# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------


@dataclass
class Step:
    idx: int
    reasoning: str
    command: str
    returncode: Optional[int]
    observation: str


@dataclass
class ParsedTrajectory:
    instance_id: str
    task: str
    steps: list[Step]
    final_patch: str
    exit_status: str
    n_api_calls: int
    path: str
    # Optional oracle fields populated from sample.metadata for stronger grading.
    # All default to empty so callers/datasets without these fields stay
    # backward-compatible: when empty, the [ORACLE] block is simply not emitted.
    gold_patch: str = ""
    gold_changed_files: list[str] = field(default_factory=list)
    gold_test_files: list[str] = field(default_factory=list)


# Matches the standard ``diff --git a/path b/path`` header. Robust to spaces
# in file paths (rare but legal) by lazily matching up to ``" b/"``.
_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)\s*$", re.MULTILINE)


def _parse_patch_files(patch_text: str) -> list[str]:
    """Extract the set of files touched by a unified diff/patch.

    Returns a sorted, de-duplicated list using the ``b/`` (post-image) path,
    which corresponds to the final filename — including renames and new files.
    Returns ``[]`` for empty/None input or anything that doesn't look like a
    git diff.
    """
    if not patch_text:
        return []
    if not isinstance(patch_text, str):
        try:
            patch_text = patch_text.decode("utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return []
    seen: dict[str, None] = {}
    for m in _DIFF_GIT_RE.finditer(patch_text):
        post = m.group(2).strip()
        # Skip /dev/null (file deletion: post path may be missing in some diffs)
        if post and post != "/dev/null":
            seen.setdefault(post, None)
    return sorted(seen)


_RC_RE = re.compile(r"<returncode>\s*(-?\d+)\s*</returncode>", re.DOTALL)
_OUT_RE = re.compile(r"<output>\s*(.*?)\s*</output>", re.DOTALL)


def _extract_command(assistant_msg: dict) -> str:
    """Pull the bash command from an assistant message.

    Prefers extra.actions[].command (already parsed by the agent), falling back
    to tool_calls[].function.arguments JSON.
    """
    extra = assistant_msg.get("extra") or {}
    actions = extra.get("actions") or []
    cmds = [a.get("command", "") for a in actions if a.get("command")]
    if cmds:
        return "\n".join(cmds)

    tool_calls = assistant_msg.get("tool_calls") or []
    cmds = []
    for tc in tool_calls:
        args = (tc.get("function") or {}).get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if isinstance(args, dict) and args.get("command"):
            cmds.append(args["command"])
    return "\n".join(cmds)


def _parse_observation(tool_msg: dict) -> tuple[Optional[int], str]:
    content = tool_msg.get("content") or ""
    if not isinstance(content, str):
        content = json.dumps(content, default=str)
    rc_m = _RC_RE.search(content)
    returncode = int(rc_m.group(1)) if rc_m else None
    out_m = _OUT_RE.search(content)
    observation = out_m.group(1) if out_m else content
    return returncode, observation


def _build_steps_from_messages(messages: list[dict]) -> list[Step]:
    """Walk a message list and pair each assistant turn with its observation.

    Observation is the immediately following message with role in
    {"tool", "user"} — ``tool`` matches the mini-swe CLI schema; ``user`` matches
    ``swe_generate_v2`` where bash output is fed back as a user message.
    """
    steps: list[Step] = []
    i = 0
    step_idx = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        if m.get("role") == "assistant":
            reasoning = m.get("reasoning_content") or m.get("content") or ""
            command = _extract_command(m)
            returncode, observation = (None, "")
            if i + 1 < n and messages[i + 1].get("role") in ("tool", "user"):
                returncode, observation = _parse_observation(messages[i + 1])
                i += 1
            steps.append(
                Step(
                    idx=step_idx,
                    reasoning=reasoning if isinstance(reasoning, str) else str(reasoning),
                    command=command,
                    returncode=returncode,
                    observation=observation,
                )
            )
            step_idx += 1
        i += 1
    return steps


def parse_trajectory(path: str | Path) -> ParsedTrajectory:
    """Parse a mini-swe ``.traj.json`` file into a :class:`ParsedTrajectory`."""
    path = Path(path)
    data = json.loads(path.read_text())
    info = data.get("info", {})
    messages = data.get("messages", [])
    instance_id = data.get("instance_id", path.stem.replace(".traj", ""))

    # First user message holds the rendered task / PR description.
    task = ""
    for m in messages:
        if m.get("role") == "user":
            task = m.get("content") or ""
            break

    steps = _build_steps_from_messages(messages)

    final_patch = info.get("submission", "") or data.get("model_patch", "") or ""
    exit_status = info.get("exit_status") or data.get("exit_status", "Unknown")
    n_api_calls = (info.get("model_stats") or {}).get("api_calls") or data.get("n_steps", len(steps))

    return ParsedTrajectory(
        instance_id=instance_id,
        task=task,
        steps=steps,
        final_patch=final_patch,
        exit_status=exit_status,
        n_api_calls=n_api_calls,
        path=str(path),
    )


def parse_trajectory_from_sample(sample) -> ParsedTrajectory:
    """Build a :class:`ParsedTrajectory` from an in-memory slime ``Sample``.

    Reads ``sample.metadata["trajectory"]`` (produced by ``swe_generate_v2``'s
    ``agent.serialize()``) when present; falls back to reading the trajectory
    JSON file at ``sample.metadata["trajectory_path"]`` (which has the
    ``swe_generate_v2`` on-disk schema: top-level ``messages`` / ``model_patch``,
    no ``info`` wrapper).
    """
    meta = sample.metadata or {}
    instance_id = meta.get("instance_id", "unknown")
    # Use sentinel default so we can tell "missing" apart from "explicitly set".
    _MISSING = object()
    exit_status = meta.get("exit_status", _MISSING)
    final_patch = meta.get("final_output", "") or ""
    n_steps = meta.get("n_steps", 0)

    messages: list[dict] = []
    traj_obj = meta.get("trajectory")
    if isinstance(traj_obj, dict):
        messages = traj_obj.get("messages") or []
    elif isinstance(traj_obj, list):
        messages = traj_obj

    if not messages:
        tp = meta.get("trajectory_path")
        if tp and tp != "unknown":
            try:
                data = json.loads(Path(tp).read_text())
                messages = data.get("messages", []) or []
                final_patch = final_patch or data.get("model_patch", "") or ""
                if exit_status is _MISSING:
                    exit_status = data.get("exit_status", "Unknown")
            except Exception as e:  # noqa: BLE001
                logger.warning("[swe_prm] failed to read trajectory_path %s: %s", tp, e)

    if exit_status is _MISSING:
        exit_status = "Unknown"

    task = ""
    for m in messages:
        if m.get("role") == "user":
            task = m.get("content") or ""
            break
    # Fallback to sample.prompt if no user message found in trajectory.
    if not task:
        try:
            task = sample.prompt if isinstance(sample.prompt, str) else sample.prompt[0]["content"]
        except Exception:  # noqa: BLE001
            task = ""

    steps = _build_steps_from_messages(messages)
    if not n_steps:
        n_steps = len(steps)

    # Oracle (gold patch) — optional, used to ground D5/D6 when present.
    # SWE-bench-style datasets typically expose:
    #   metadata["patch"]      -> gold source-code patch (string)
    #   metadata["test_patch"] -> gold test-file patch  (string, optional)
    # Several training jsonl variants use alternative keys; accept the common ones.
    gold_patch = (
        meta.get("patch")
        or meta.get("gold_patch")
        or meta.get("golden_patch")
        or ""
    ) or ""
    gold_test_patch = meta.get("test_patch") or ""
    gold_changed_files = _parse_patch_files(gold_patch)
    gold_test_files = _parse_patch_files(gold_test_patch)

    return ParsedTrajectory(
        instance_id=instance_id,
        task=task,
        steps=steps,
        final_patch=final_patch,
        exit_status=exit_status,
        n_api_calls=n_steps,
        path=str(meta.get("trajectory_path", "<in-memory>")),
        gold_patch=gold_patch,
        gold_changed_files=gold_changed_files,
        gold_test_files=gold_test_files,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _truncate(text: str, max_chars: int, head_frac: float = 0.7) -> str:
    """Truncate keeping head and tail (the informative ends of long logs)."""
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    head = int(max_chars * head_frac)
    tail = max_chars - head
    if tail <= 0:
        # head-only (avoid the text[-0:] pitfall, which returns the whole string)
        return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]..."
    return text[:head] + f"\n...[truncated {len(text) - max_chars} chars]...\n" + text[-tail:]


# Lines that carry test-result signal (kept verbatim in compact mode).
_TEST_SUMMARY_RE = re.compile(
    r"(\d+\s+(passed|failed|error|skipped|xfailed|warning))|"
    r"^(FAILED|ERROR|PASSED)\b|"
    r"^={3,}.*(passed|failed|error).*={3,}",
    re.IGNORECASE | re.MULTILINE,
)


def render_observation(observation: str, returncode: Optional[int], max_chars: int, mode: str) -> str:
    """Render a step observation under a compression ``mode``.

    - ``none``    : drop the body entirely (returncode is shown separately).
    - ``compact`` : type-aware — keep errors, test summaries, or a short head;
                    collapse big file/dir dumps to a line count. Highest signal
                    per token; recommended for long trajectories.
    - ``full``    : head+tail truncation to ``max_chars`` (default behaviour).
    """
    observation = observation or ""
    if mode == "none":
        return ""
    if mode == "full":
        return _truncate(observation, max_chars)

    # compact mode
    body = observation.strip()
    if not body:
        return "(empty)"
    # 1) Non-zero return code: the error text is the highest-signal content.
    if returncode not in (0, None):
        return _truncate(body, min(max_chars, 800))
    # 2) Test output: keep only result-bearing lines.
    summary_lines = [m.group(0) for m in _TEST_SUMMARY_RE.finditer(body)]
    if summary_lines:
        # de-dup while preserving order, cap to a handful
        seen: set[str] = set()
        kept = [s for s in summary_lines if not (s in seen or seen.add(s))][:12]
        return _truncate("\n".join(kept), max_chars)
    n_lines = body.count("\n") + 1
    # 3) Large dump (file content / dir listing): the fact it was read matters
    #    more than the content; keep a short head + a size marker.
    if len(body) > max_chars:
        return f"[output: {n_lines} lines, {len(body)} chars] " + _truncate(body, min(max_chars, 400), head_frac=1.0)
    return body


def build_user_prompt(
    traj: ParsedTrajectory,
    *,
    obs_maxchars: int = 1200,
    obs_mode: str = "full",
    patch_maxchars: int = 4000,
    task_maxchars: int = 4000,
    gold_patch_files: Optional[list[str]] = None,
    gold_patch_maxchars: int = 3000,
    use_oracle: bool = True,
) -> str:
    """Render the per-trajectory user message fed to the judge.

    Oracle block (printed only when ``use_oracle`` is True and oracle info is
    available on ``traj``): if ``traj.gold_patch`` is non-empty, we include a
    truncated reference patch plus the list of changed source/test files. This
    grounds D5 (were the right tests run?) and D6 (was the right area edited?)
    against ground truth.

    ``gold_patch_files`` (legacy, optional): preserved for callers that only
    have the file list; merged with ``traj.gold_changed_files`` for back-compat.
    """
    lines: list[str] = []
    lines.append("[TASK / ISSUE]")
    lines.append(_truncate(traj.task, task_maxchars))
    lines.append("")

    # Resolve final oracle file list (prefer traj-derived; fall back to legacy arg).
    changed_files = list(traj.gold_changed_files) if traj.gold_changed_files else (
        list(gold_patch_files) if gold_patch_files else []
    )
    has_oracle = use_oracle and (traj.gold_patch or changed_files or traj.gold_test_files)

    if has_oracle:
        lines.append("[ORACLE — reference (gold) patch for grounding only]")
        lines.append("Use this ONLY to ground D5 (were RELEVANT tests run?) and D6")
        lines.append("(were the RIGHT files / regions edited?). Still grade PROCESS")
        lines.append("quality — do not just check whether the agent's edit matches.")
        if changed_files:
            lines.append("gold_changed_files: " + ", ".join(changed_files))
        if traj.gold_test_files:
            lines.append("gold_test_files: " + ", ".join(traj.gold_test_files))
        if traj.gold_patch:
            lines.append("gold_patch:")
            lines.append(_truncate(traj.gold_patch, gold_patch_maxchars))
        lines.append("")

    lines.append(f"[TRAJECTORY] exit_status={traj.exit_status}, steps={len(traj.steps)}")
    lines.append("")
    for s in traj.steps:
        lines.append(f"--- STEP {s.idx} ---")
        if s.reasoning.strip():
            lines.append("reasoning: " + _truncate(s.reasoning, 600))
        lines.append("command: " + _truncate(s.command, 400))
        rc = "" if s.returncode is None else f" (returncode={s.returncode})"
        rendered_obs = render_observation(s.observation, s.returncode, obs_maxchars, obs_mode)
        if rendered_obs:
            lines.append(f"observation{rc}: " + rendered_obs)
        elif s.returncode is not None:
            lines.append(f"observation{rc}")  # body dropped (obs_mode=none)
        lines.append("")

    lines.append("[FINAL SUBMITTED PATCH]")
    lines.append(_truncate(traj.final_patch, patch_maxchars) if traj.final_patch else "(none)")
    lines.append("")
    lines.append("Grade THIS trajectory now. Output only the JSON object.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _snap_to_allowed(dim: str, value: Any) -> float:
    """Clamp a judge score to the nearest allowed discrete value for the dim."""
    allowed = DIM_ALLOWED[dim]
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(allowed, key=lambda a: abs(a - v))


def aggregate(scores: dict[str, Any], harmful: bool, hallucinated: bool) -> tuple[float, dict[str, float]]:
    """Combine per-dimension scores into a single [0,1] process reward.

    Returns (final_reward, snapped_scores).
    """
    snapped = {d: _snap_to_allowed(d, scores.get(d, 0.0)) for d in DIM_WEIGHTS}
    raw = sum(DIM_WEIGHTS[d] * snapped[d] for d in DIM_WEIGHTS)
    if harmful or hallucinated:
        raw = min(raw, FLAG_CAP)
    return max(0.0, min(1.0, raw)), snapped


# ---------------------------------------------------------------------------
# Judge result + API call
# ---------------------------------------------------------------------------


@dataclass
class GradeResult:
    instance_id: str
    process_reward: float
    scores: dict[str, float] = field(default_factory=dict)
    harmful: bool = False
    hallucinated: bool = False
    reasoning: str = ""
    exit_status: str = ""
    n_steps: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# GitHub Copilot API defaults (OpenAI-compatible chat endpoint).
GH_COPILOT_BASE_URL = "https://api.githubcopilot.com"
GH_API_VERSION = "2026-01-09"
GH_INTEGRATION_ID = "copilot-cli"


def _gh_token() -> str:
    """Return a GitHub token for the Copilot API.

    Prefers an explicit ``PRM_API_KEY`` / ``GH_TOKEN``; otherwise shells out to
    ``gh auth token`` (the GitHub CLI), mirroring the reference curl command.
    """
    import shutil
    import subprocess

    token = os.environ.get("PRM_API_KEY") or os.environ.get("GH_TOKEN")
    if token:
        return token.strip()
    if shutil.which("gh") is None:
        raise RuntimeError("`gh` CLI not found and no PRM_API_KEY/GH_TOKEN set.")
    try:
        out = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=True, timeout=15
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"`gh auth token` failed: {e.stderr.strip()}") from e
    token = out.stdout.strip()
    if not token:
        raise RuntimeError("`gh auth token` returned an empty token.")
    return token


def _make_client():
    """Construct an async OpenAI / Azure-OpenAI / GitHub-Copilot client from env vars."""
    from openai import AsyncAzureOpenAI, AsyncOpenAI

    api_type = os.environ.get("PRM_API_TYPE", "openai").lower()

    if api_type in ("gh", "copilot", "github"):
        # Wrap the GitHub Copilot endpoint. Equivalent to the reference curl:
        #   -H "Authorization: Bearer $(gh auth token)"
        #   -H "Copilot-Integration-Id: copilot-cli"
        #   -H "X-GitHub-Api-Version: 2026-01-09"
        base_url = os.environ.get("PRM_API_BASE_URL", GH_COPILOT_BASE_URL)
        default_headers = {
            "Copilot-Integration-Id": os.environ.get("PRM_GH_INTEGRATION_ID", GH_INTEGRATION_ID),
            "X-GitHub-Api-Version": os.environ.get("PRM_GH_API_VERSION", GH_API_VERSION),
            "Editor-Version": os.environ.get("PRM_GH_EDITOR_VERSION", "slime-swe-prm/0.1"),
        }
        return AsyncOpenAI(api_key=_gh_token(), base_url=base_url, default_headers=default_headers)

    api_key = os.environ.get("PRM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("PRM_API_KEY (or OPENAI_API_KEY) must be set.")

    if api_type == "azure":
        endpoint = os.environ.get("PRM_API_BASE_URL") or os.environ["AZURE_OPENAI_ENDPOINT"]
        api_version = os.environ.get("PRM_API_VERSION", "2024-08-01-preview")
        return AsyncAzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)

    base_url = os.environ.get("PRM_API_BASE_URL")  # None => default OpenAI
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def _parse_judge_json(text: str) -> dict:
    """Parse the judge's JSON output, tolerating ```json fences and stray prose."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def _is_responses_only_model(model: str) -> bool:
    """Models that are only served via /responses (reject /chat/completions).

    Currently: any gpt-5.x family on GitHub Copilot. Used both as a typo
    safety-net and as an automatic fallback when the user picks the wrong
    endpoint.
    """
    if not model:
        return False
    m = model.lower()
    # gpt-5, gpt-5.5, gpt-5-mini, gpt-5.1-codex, etc.
    return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3")


async def _call_judge(
    client,
    *,
    endpoint: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: float,
    temperature: float,
    max_tokens: int,
    reasoning_effort: Optional[str],
) -> str:
    """Call the judge and return raw text, abstracting chat vs responses APIs.

    - ``chat``: /chat/completions (gpt-4o*, gpt-4.1, claude-*, gemini-*).
    - ``responses``: /responses (gpt-5.x reasoning models, which are NOT served
      on /chat/completions). Reasoning models reject ``temperature``, so it is
      omitted; ``max_tokens`` must be generous since it also covers reasoning.

    Endpoint resolution:
      1. Accept both "responses" and "response" (common typo).
      2. If the user picked ``chat`` but the model name implies a
         responses-only model (gpt-5.x / o1 / o3), auto-switch with a warning.
      3. If a /chat/completions call returns the "unsupported_api_for_model"
         400, retry once on /responses (last-ditch safety net).
    """
    endpoint = (endpoint or "chat").strip().lower()
    if endpoint in ("response", "responses"):
        endpoint = "responses"
    elif endpoint == "chat" and _is_responses_only_model(model):
        logger.warning(
            "[swe_prm] model %r is responses-only; auto-switching PRM_ENDPOINT 'chat' -> 'responses'",
            model,
        )
        endpoint = "responses"

    if endpoint == "responses":
        kwargs: dict[str, Any] = dict(
            model=model,
            instructions=system_prompt,
            input=user_prompt,
            max_output_tokens=max(max_tokens, 4096),
        )
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        resp = await asyncio.wait_for(client.responses.create(**kwargs), timeout=timeout)
        return resp.output_text or ""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    base_kwargs = dict(
        model=model, messages=messages, temperature=temperature, max_completion_tokens=max_tokens
    )
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(**base_kwargs, response_format={"type": "json_object"}),
            timeout=timeout,
        )
    except Exception as e:  # noqa: BLE001
        err_str = str(e)
        # Final safety net: if the server explicitly says this model is not
        # served on /chat/completions, retry once on /responses.
        if "unsupported_api_for_model" in err_str or "is not accessible via the /chat/completions endpoint" in err_str:
            logger.warning(
                "[swe_prm] model %r rejected /chat/completions; retrying on /responses. Error: %s",
                model, err_str,
            )
            kwargs = dict(
                model=model,
                instructions=system_prompt,
                input=user_prompt,
                max_output_tokens=max(max_tokens, 4096),
            )
            if reasoning_effort:
                kwargs["reasoning"] = {"effort": reasoning_effort}
            resp = await asyncio.wait_for(client.responses.create(**kwargs), timeout=timeout)
            return resp.output_text or ""
        # Some back-end models reject response_format; retry once without it
        # (our parser tolerates ```json fences).
        logger.debug("[swe_prm] retrying without json mode: %s", e)
        resp = await asyncio.wait_for(client.chat.completions.create(**base_kwargs), timeout=timeout)
    return resp.choices[0].message.content or ""


async def grade_trajectory(
    client,
    traj: ParsedTrajectory,
    *,
    model: str,
    endpoint: str = "chat",
    reasoning_effort: Optional[str] = None,
    timeout: float = 60.0,
    temperature: float = 0.0,
    max_completion_tokens: int = 1024,
    gold_patch_files: Optional[list[str]] = None,
    obs_maxchars: int = 1200,
    obs_mode: str = "full",
    use_oracle: bool = True,
    gold_patch_maxchars: int = 3000,
) -> GradeResult:
    """Call the judge on one trajectory. Never raises: errors -> reward 0.0."""
    result = GradeResult(
        instance_id=traj.instance_id,
        process_reward=0.0,
        exit_status=traj.exit_status,
        n_steps=len(traj.steps),
    )
    try:
        user_prompt = build_user_prompt(
            traj,
            gold_patch_files=gold_patch_files,
            obs_maxchars=obs_maxchars,
            obs_mode=obs_mode,
            use_oracle=use_oracle,
            gold_patch_maxchars=gold_patch_maxchars,
        )
        content = await _call_judge(
            client,
            endpoint=endpoint,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
        )
        parsed = _parse_judge_json(content)

        harmful = bool(parsed.get("harmful", False))
        hallucinated = bool(parsed.get("hallucinated", False))
        scores_raw = parsed.get("scores", {}) or {}
        reward, snapped = aggregate(scores_raw, harmful, hallucinated)

        result.process_reward = reward
        result.scores = snapped
        result.harmful = harmful
        result.hallucinated = hallucinated
        result.reasoning = str(parsed.get("reasoning", ""))[:1000]
    except Exception as e:  # noqa: BLE001 — judge must never crash the pipeline
        result.error = f"{type(e).__name__}: {e}"
        logger.warning("[swe_prm] grading failed for %s: %s", traj.instance_id, result.error)
    return result


# ---------------------------------------------------------------------------
# In-process client cache + slime Sample entry point
# ---------------------------------------------------------------------------

_CLIENT_CACHE = None
_CLIENT_LOCK: Optional[asyncio.Lock] = None


def _client_lock() -> asyncio.Lock:
    """Lazy lock construction: tied to the currently-running event loop."""
    global _CLIENT_LOCK
    if _CLIENT_LOCK is None:
        _CLIENT_LOCK = asyncio.Lock()
    return _CLIENT_LOCK


async def _get_cached_client():
    """Return a process-wide cached async client, creating it on first use."""
    global _CLIENT_CACHE
    async with _client_lock():
        if _CLIENT_CACHE is None:
            _CLIENT_CACHE = _make_client()
    return _CLIENT_CACHE


async def score_sample(sample, *, timeout: Optional[float] = None) -> GradeResult:
    """Grade a slime ``Sample``'s trajectory with the configured LLM judge.

    Convenience entry point intended to be called from reward functions. Reads
    config from ``PRM_*`` env vars (see module docstring). Never raises: any
    error yields a :class:`GradeResult` with ``process_reward=0.0`` and a
    populated ``error`` field.
    """
    try:
        traj = parse_trajectory_from_sample(sample)
    except Exception as e:  # noqa: BLE001
        return GradeResult(instance_id="parse_err", process_reward=0.0, error=f"parse: {e}")

    try:
        client = await _get_cached_client()
    except Exception as e:  # noqa: BLE001
        return GradeResult(
            instance_id=traj.instance_id,
            process_reward=0.0,
            exit_status=traj.exit_status,
            n_steps=len(traj.steps),
            error=f"client_init: {e}",
        )

    model = os.environ.get("PRM_MODEL", "gpt-4o-mini")
    endpoint = os.environ.get("PRM_ENDPOINT", "chat")
    reasoning_effort = os.environ.get("PRM_REASONING_EFFORT") or None
    if timeout is None:
        timeout = float(os.environ.get("PRM_TIMEOUT", "60"))
    temperature = float(os.environ.get("PRM_TEMPERATURE", "0.0"))
    max_tokens = int(os.environ.get("PRM_MAX_TOKENS", "1024"))
    obs_maxchars = int(os.environ.get("PRM_OBS_MAXCHARS", "1200"))
    obs_mode = os.environ.get("PRM_OBS_MODE", "compact")
    # Oracle controls: default ON. Set PRM_USE_ORACLE=0 for ablation runs that
    # want the pre-oracle behaviour. PRM_GOLD_PATCH_MAXCHARS bounds the length
    # of the embedded gold patch (file list is always shown).
    use_oracle = os.environ.get("PRM_USE_ORACLE", "1") not in ("0", "false", "False", "")
    gold_patch_maxchars = int(os.environ.get("PRM_GOLD_PATCH_MAXCHARS", "3000"))

    return await grade_trajectory(
        client,
        traj,
        model=model,
        endpoint=endpoint,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        obs_maxchars=obs_maxchars,
        obs_mode=obs_mode,
        use_oracle=use_oracle,
        gold_patch_maxchars=gold_patch_maxchars,
    )


# ---------------------------------------------------------------------------
# Batch driver / CLI
# ---------------------------------------------------------------------------


def discover_trajectories(rollouts_dir: str | Path) -> list[Path]:
    """Find all ``*.traj.json`` files under the rollouts directory."""
    rollouts_dir = Path(rollouts_dir)
    return sorted(rollouts_dir.glob("*/*.traj.json"))


async def grade_directory(
    rollouts_dir: str,
    out_path: str,
    *,
    model: str,
    endpoint: str = "chat",
    reasoning_effort: Optional[str] = None,
    max_concurrency: int = 8,
    limit: Optional[int] = None,
    sample: Optional[int] = None,
    seed: int = 0,
    timeout: float = 60.0,
    obs_maxchars: int = 1200,
    obs_mode: str = "full",
) -> list[GradeResult]:
    files = discover_trajectories(rollouts_dir)
    if sample is not None and sample < len(files):
        import random as _random

        files = _random.Random(seed).sample(files, sample)
    elif limit is not None:
        files = files[:limit]
    logger.info("[swe_prm] grading %d trajectories with model=%s", len(files), model)

    client = _make_client()
    sem = asyncio.Semaphore(max_concurrency)

    async def _worker(fp: Path) -> GradeResult:
        async with sem:
            try:
                traj = parse_trajectory(fp)
            except Exception as e:  # noqa: BLE001
                return GradeResult(instance_id=fp.stem, process_reward=0.0, error=f"parse: {e}")
            return await grade_trajectory(
                client,
                traj,
                model=model,
                endpoint=endpoint,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                obs_maxchars=obs_maxchars,
                obs_mode=obs_mode,
            )

    results = await asyncio.gather(*(_worker(fp) for fp in files))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")

    _print_summary(results)
    try:
        await client.close()
    except Exception:  # noqa: BLE001
        pass
    return results


def _print_summary(results: list[GradeResult]) -> None:
    ok = [r for r in results if not r.error]
    n = len(results)
    n_ok = len(ok)
    n_err = n - n_ok
    if ok:
        mean = sum(r.process_reward for r in ok) / n_ok
        dim_means = {
            d: sum(r.scores.get(d, 0.0) for r in ok) / n_ok for d in DIM_WEIGHTS
        }
        n_harmful = sum(r.harmful for r in ok)
        n_hallu = sum(r.hallucinated for r in ok)
    else:
        mean, dim_means, n_harmful, n_hallu = 0.0, {}, 0, 0

    print("\n" + "=" * 60)
    print(f"[swe_prm] graded {n} trajectories ({n_ok} ok, {n_err} errors)")
    print(f"  mean process_reward: {mean:.3f}")
    if dim_means:
        print("  per-dimension mean: " + "  ".join(f"{d}={v:.2f}" for d, v in dim_means.items()))
        print(f"  harmful={n_harmful}  hallucinated={n_hallu}")
    print("=" * 60)


async def list_models() -> None:
    """List models available to the configured provider (verifies connectivity).

    For the ``gh`` provider this mirrors:
        curl https://api.githubcopilot.com/models -H "Authorization: Bearer $(gh auth token)" ...
    """
    client = _make_client()
    try:
        models = await client.models.list()
        ids = sorted(m.id for m in models.data)
        print(f"[swe_prm] {len(ids)} models available:")
        for mid in ids:
            print("  ", mid)
    finally:
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass


def _dry_run(rollouts_dir: str, limit: int, obs_maxchars: int, obs_mode: str = "full") -> None:
    files = discover_trajectories(rollouts_dir)[: max(1, limit)]
    for fp in files:
        traj = parse_trajectory(fp)
        prompt = build_user_prompt(traj, obs_maxchars=obs_maxchars, obs_mode=obs_mode)
        print("#" * 80)
        print(f"# {traj.instance_id}  exit={traj.exit_status}  steps={len(traj.steps)}  "
              f"prompt_chars={len(prompt)}")
        print("#" * 80)
        print("=== SYSTEM ===")
        print(SYSTEM_PROMPT)
        print("=== USER ===")
        print(prompt)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge process grader for mini-swe trajectories")
    parser.add_argument("--rollouts-dir", required=True)
    parser.add_argument("--out", default="examples/mini_swe/ideas/process_grader/grades.jsonl")
    parser.add_argument("--model", default=os.environ.get("PRM_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--endpoint",
        choices=["chat", "responses"],
        default=os.environ.get("PRM_ENDPOINT", "chat"),
        help="API endpoint: 'chat' (/chat/completions) or 'responses' (/responses, for gpt-5.x)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get("PRM_REASONING_EFFORT"),
        help="Reasoning effort for responses-API models (e.g. low/medium/high)",
    )
    parser.add_argument("--max-concurrency", type=int, default=int(os.environ.get("PRM_MAX_CONCURRENCY", "8")))
    parser.add_argument("--limit", type=int, default=None, help="Only grade the first N trajectories")
    parser.add_argument("--sample", type=int, default=None, help="Randomly sample N trajectories (overrides --limit)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --sample")
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("PRM_TIMEOUT", "60")))
    parser.add_argument("--obs-maxchars", type=int, default=int(os.environ.get("PRM_OBS_MAXCHARS", "1200")))
    parser.add_argument(
        "--obs-mode",
        choices=["full", "compact", "none"],
        default=os.environ.get("PRM_OBS_MODE", "full"),
        help="Observation rendering: full (head+tail trunc), compact (type-aware, "
             "recommended for long trajectories), none (drop bodies, keep returncode)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling the API")
    parser.add_argument("--list-models", action="store_true", help="List provider models and exit")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.list_models:
        asyncio.run(list_models())
        return

    if args.dry_run:
        _dry_run(args.rollouts_dir, args.limit or 1, args.obs_maxchars, args.obs_mode)
        return

    asyncio.run(
        grade_directory(
            args.rollouts_dir,
            args.out,
            model=args.model,
            endpoint=args.endpoint,
            reasoning_effort=args.reasoning_effort,
            max_concurrency=args.max_concurrency,
            limit=args.limit,
            sample=args.sample,
            seed=args.seed,
            timeout=args.timeout,
            obs_maxchars=args.obs_maxchars,
            obs_mode=args.obs_mode,
        )
    )


if __name__ == "__main__":
    main()
