#!/usr/bin/env python3
"""Full-run batch driver for TestPatchResolveAgent over SWE-bench-family datasets.

Fan-out version of smoke_resolve.py: ThreadPool over instances, each worker pinned to one of the
8 local sglang endpoints (round-robin by index). Per instance: a generator sandbox plus an
optional isolated verifier, resolve agent, then the OFFLINE gold-check (gold read only here, never
during generation). One JSONL row per instance; re-running skips instances already recorded.

Use ``--subset multilingual`` for SWE-bench Multilingual. Its Base/Gold checks automatically use
the language-specific build commands and log parsers instead of the Python-oriented classifier.

Run in a configured sandbox/model environment (SANDBOX_API_KEY + SANDBOX_BASE_URL sourced;
PYTHONPATH has external/azure-modal):
  python3 -u swe_harness/gentest_v2/batch_resolve.py \
    --output swe_harness/gentest_v2/runs/resolve_verified_full \
    --model "hosted_vllm//absolute/path/to/Qwen3.5-35B-A3B" \
    --model-class litellm_response \
    --ports 8000,8001,8002,8003,8004,8005,8006,8007 \
    --workers 32 --step-limit 60 --install-timeout 1200 \
    --overlay swe_harness/gentest_v2/versions/behavior_contract.yaml
"""
import os, sys, json, argparse, threading, traceback, time, copy, re, shlex
import concurrent.futures
from pathlib import Path

import requests

sys.path.insert(0, "swe_harness/src")

from minisweagent.config import get_config_from_spec
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swerebench import get_sb_environment, DATASET_MAPPING
from minisweagent.run.benchmarks.gentest import (
    SCALESWE_DATASET, GitHistoryGuardEnv, get_gentest_agent_environment, get_gentest_environment,
    run_shared_test_patch_official, run_test_patch_official, sanitize_git_history,
)
from minisweagent.agents import get_agent
import yaml

_print_lock = threading.Lock()
_write_lock = threading.Lock()
REBENCH_V2_DATASET = DATASET_MAPPING["rebench_v2"]
_SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.#-]{0,239}$")
_STRICT_HISTORY_SETTINGS = ("GENTEST_SANITIZE_GIT_HISTORY", "GENTEST_GIT_HISTORY_GUARD")


def log(msg: str):
    with _print_lock:
        print(msg, flush=True)


def _safe_path_component(value: object, *, field: str) -> str:
    text = str(value or "")
    if not _SAFE_PATH_COMPONENT.fullmatch(text):
        raise ValueError(f"unsafe {field} for trajectory path: {text!r}")
    return text


def _require_strict_history_isolation() -> None:
    disabled = [name for name in _STRICT_HISTORY_SETTINGS if os.environ.get(name, "1") != "1"]
    if disabled:
        settings = ", ".join(f"{name}=1" for name in disabled)
        raise ValueError(f"strict behavior-contract batch generation requires {settings}")


def _gold_diagnostics(result: dict) -> dict:
    """Persist enough offline-Gold detail to separate semantic failures from infra/empty runs."""
    keys = (
        "verdict", "returncode", "passed", "failed", "errors", "clean_fail",
        "infra_failure", "infra_reason", "failure_category", "next_action",
        "missing_module", "missing_symbol", "missing_fixture", "official_verifier_kind",
        "failing_nodes",
    )
    diagnostics = {key: result.get(key) for key in keys if result.get(key) is not None}
    diagnostics["output_tail"] = str(
        result.get("official_output_tail") or result.get("tail") or ""
    )[-2000:]
    return diagnostics


def load_instances(subset: str, split: str) -> list[dict]:
    from datasets import load_dataset
    path = DATASET_MAPPING.get(subset, subset)
    ds = load_dataset(path, split=split)
    rows = [dict(r) for r in ds]
    for row in rows:
        row.setdefault("dataset", path)
    return rows


def _adapt_rebench_v2_environment(env, instance: dict):
    """Expose ReBench V2's repo-root checkout through gentest's /testbed contract."""
    if instance.get("dataset") != REBENCH_V2_DATASET:
        return env

    repo = str(instance.get("repo") or "").strip()
    repo_name = repo.rsplit("/", 1)[-1]
    if not repo_name or repo_name in {".", ".."} or "/" in repo_name:
        raise ValueError(f"Invalid ReBench V2 repo for workdir resolution: {repo!r}")

    workdir = f"/{repo_name}"
    quoted_workdir = shlex.quote(workdir)
    try:
        linked = env.execute(
            {
                "command": (
                    f"test -d {quoted_workdir} && "
                    "if [ -e /testbed ] && [ ! -L /testbed ]; then "
                    "echo 'ReBench V2 adapter refuses to replace existing /testbed' >&2; exit 73; fi; "
                    f"ln -sfn {quoted_workdir} /testbed && "
                    f"commit=$(git -C {quoted_workdir} rev-parse HEAD) && "
                    "printf 'GENTEST_REBENCH_BASE_COMMIT=%s\\n' \"$commit\""
                )
            },
            cwd=workdir,
            timeout=120,
        )
        if linked.get("returncode") != 0:
            output = str(linked.get("output") or "") + str(linked.get("exception_info") or "")
            raise RuntimeError(f"ReBench V2 /testbed adapter failed: {output[-2000:]}")

        commit_prefix = "GENTEST_REBENCH_BASE_COMMIT="
        commit_lines = [
            line[len(commit_prefix) :].strip()
            for line in str(linked.get("output") or "").splitlines()
            if line.startswith(commit_prefix)
        ]
        if len(commit_lines) != 1 or not commit_lines[0]:
            raise RuntimeError(
                f"ReBench V2 adapter did not report exactly one base commit: {commit_lines!r}"
            )
        actual_commit = commit_lines[0]
        expected_commit = str(instance.get("base_commit") or "").strip()
        if expected_commit and actual_commit != expected_commit:
            raise RuntimeError(
                f"ReBench V2 base commit mismatch: expected {expected_commit}, got {actual_commit}"
            )
        return env
    except Exception:
        env.cleanup()
        raise


def get_resolve_environment(config: dict, instance: dict):
    """Use dataset-native workdir adapters while preserving legacy SWE-bench routing."""
    if instance.get("dataset") == SCALESWE_DATASET:
        return get_gentest_environment(config, instance)
    return _adapt_rebench_v2_environment(get_sb_environment(config, instance), instance)


def get_agent_environment(env, instance: dict):
    """Apply conda activation only to datasets whose images use the SWE-bench env contract."""
    return get_gentest_agent_environment(env, instance)


def _get_resolve_environment_with_retry(config: dict, instance: dict, *, phase: str):
    """Retry the transient create-then-disappear failure observed from the sandbox service."""
    for attempt in range(1, 4):
        try:
            return get_resolve_environment(config, instance)
        except requests.exceptions.HTTPError as exc:
            response = exc.response
            if response is None or response.status_code != 404 or attempt == 3:
                raise
            log(
                f"[sandbox-retry] {instance.get('instance_id')} phase={phase} "
                f"http_status=404 next_attempt={attempt + 1}/3"
            )
            time.sleep(2 ** (attempt - 1))


def build_model(args, api_base: str, model_overlay: dict):
    model_kwargs = {"max_output_tokens": args.max_output_tokens, "timeout": 600,
                    "tool_choice": args.tool_choice}
    is_openai_proxy = str(args.model).startswith("openai/")
    if api_base and is_openai_proxy:
        # openai-compatible proxy (e.g. gpt-5.x on local 8080): api_base + dummy key, NO
        # chat_template_kwargs (OpenAI models reject it). Reasoning effort still applies.
        model_kwargs["api_base"] = api_base
        model_kwargs["api_key"] = os.environ.get("OPENAI_API_KEY", "dummy")
        if args.reasoning_effort:
            model_kwargs["reasoning"] = {"effort": args.reasoning_effort, "summary": "auto"}
    elif api_base:
        # Local SGLang path. DeepSeek-V4 uses thinking_mode/reasoning_effort while Qwen3.5 uses
        # enable_thinking. Keep the explicit Qwen false value: absence does not disable thinking.
        thinking_mode = str(getattr(args, "thinking_mode", "") or "").strip()
        if thinking_mode:
            model_kwargs["chat_template_kwargs"] = {
                "thinking_mode": thinking_mode,
                # SGLang's DeepSeek-V4 reasoning parser currently keys forced extraction on the
                # legacy `thinking=True` toggle, while the checkpoint encoder keys generation on
                # `thinking_mode="thinking"`. Send both so reasoning lands in reasoning_content.
                "thinking": True,
                "reasoning_effort": args.reasoning_effort,
            }
        else:
            model_kwargs["chat_template_kwargs"] = {"enable_thinking": bool(args.enable_thinking)}
        model_kwargs["api_base"] = api_base
        model_kwargs["api_key"] = "EMPTY"
    else:
        # trapi/codex path: bearer token file (TRAPI_BEARER_TOKEN_FILE), reasoning effort
        if args.tool_choice == "auto":
            model_kwargs["tool_choice"] = "required"  # codex/trapi: required is clean (only base qwen leaks)
        if args.reasoning_effort:
            model_kwargs["reasoning"] = {"effort": args.reasoning_effort, "summary": "auto"}
    overlay_kwargs = model_overlay.get("model_kwargs") or {}
    merged = {**overlay_kwargs, **model_kwargs}  # driver kwargs win (api_base survives)
    return get_model(config={
        "model_class": args.model_class,
        "model_name": args.model,
        "model_kwargs": merged,
        **{k: v for k, v in model_overlay.items() if k not in ("model_class", "model_kwargs")},
    })


def _validate_resume_verify_mode(results_jsonl: Path, shared_verify: bool) -> None:
    """Prevent one resumable results file from mixing isolated and shared-verifier rows."""
    if not results_jsonl.exists():
        return
    mismatched = []
    for line_number, line in enumerate(results_jsonl.read_text().splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        recorded = row.get("shared_verify")
        if recorded is None:
            if shared_verify:
                mismatched.append(line_number)
        elif bool(recorded) != shared_verify:
            mismatched.append(line_number)
    if mismatched:
        mode = "shared" if shared_verify else "isolated"
        preview = ",".join(str(line_number) for line_number in mismatched[:5])
        raise ValueError(
            f"refusing to resume {mode}-verify into results containing another/legacy mode "
            f"(first mismatched lines: {preview}); use a new output directory"
        )


def run_one(inst: dict, args, cfg: dict, agent_cfg_base: dict, model_overlay: dict, api_base: str,
            run_key: str = None) -> dict:
    iid = _safe_path_component(inst["instance_id"], field="instance_id")
    run_key = _safe_path_component(run_key or iid, field="run_key")
    shared_verify = bool(getattr(args, "shared_verify", False))
    trajectory_path = Path(args.output) / "trajectories" / iid / f"{run_key}.traj.json"
    rec = {
        "instance_id": iid,
        "run_key": run_key,
        "repo": inst.get("repo"),
        "api_base": api_base,
        "trajectory_path": str(trajectory_path),
        "shared_verify": shared_verify,
        "isolated_verify": not shared_verify,
        "git_history_sanitize_requested": True,
        "git_history_sanitized": False,
        "git_history_guard_enabled": True,
    }
    env = verify_env = None
    agent = None
    try:
        try:
            _require_strict_history_isolation()
        except ValueError:
            rec["exit_status"] = "configuration_error"
            rec["error_category"] = "strict_behavior_contract_history_isolation_disabled"
            raise
        generation_inst = copy.deepcopy(inst)
        for hidden_field in ("patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"):
            generation_inst.pop(hidden_field, None)
        env = _get_resolve_environment_with_retry(cfg, generation_inst, phase="generation")
        verify_env = (
            env
            if shared_verify
            else _get_resolve_environment_with_retry(cfg, inst, phase="verification")
        )
        # Gold-free: physically prune git history so `git log --all`/`git show <fix_sha>` cannot
        # reach the fix. Applied to every owned sandbox (dedup when shared_verify aliases both).
        try:
            for _senv in ({id(env): env, id(verify_env): verify_env}).values():
                if _senv is not None:
                    sanitize_git_history(_senv)
        except Exception as sanitize_exc:  # noqa: BLE001
            rec.update(
                {
                    "exit_status": "sandbox_infra_error",
                    "infra_failure": True,
                    "error_category": "git_history_sanitization_failed",
                }
            )
            raise RuntimeError(
                f"strict behavior-contract git-history sanitization failed: {sanitize_exc}"
            ) from sanitize_exc
        rec["git_history_sanitized"] = True
        model = build_model(args, api_base, model_overlay)
        warmed = {"done": False}

        def verify_runner(test_patch: str, test_command: str) -> dict:
            skip = warmed["done"]
            if shared_verify:
                result = run_shared_test_patch_official(
                    env,
                    inst,
                    test_patch,
                    test_command,
                    int(args.install_timeout),
                    skip_install=skip,
                    restore_agent_workspace=True,
                )
            else:
                result = run_test_patch_official(
                    verify_env, inst, test_patch, test_command, int(args.install_timeout), skip_install=skip
                )
            warmed["done"] = True
            return result

        agent_cfg = dict(agent_cfg_base)
        agent_cfg.setdefault("agent_class", "testpatch_resolve")
        agent_cfg["step_limit"] = args.step_limit
        agent_cfg["time_limit"] = args.instance_timeout
        if getattr(args, "stop_at_first_reject", False):
            agent_cfg["stop_at_first_reject"] = True
        agent_cfg["output_path"] = trajectory_path
        agent_env = get_agent_environment(env, generation_inst)
        agent_env = GitHistoryGuardEnv(agent_env)
        agent = get_agent(
            model,
            agent_env,
            {**agent_cfg, "verify_runner": verify_runner},
            default_type="testpatch_resolve",
        )
        info = agent.run(task=inst.get("problem_statement", ""))
        gate = info.get("resolve_gate", {}) or {}
        submission = info.get("submission") or ""
        rec.update({
            "exit_status": info.get("exit_status"),
            "base_clean_fail": gate.get("base_clean_fail"),
            "test_command": gate.get("test_command"),
            "behavior_contract": gate.get("behavior_contract"),
            "official_verifier_kind": gate.get("official_verifier_kind"),
            "n_calls": getattr(agent, "n_calls", 0),
            "submitted": info.get("exit_status") == "Submitted" and bool(submission.strip()),
            "test_patch": submission,
        })
        if info.get("exit_status") == "SubmitRejected":
            # First-submit-failure state for recovery distillation: the rejected patch, the
            # gate feedback, and the sandbox is left with the policy's edits in place.
            rec["rejected_patch"] = info.get("rejected_patch") or ""
            rec["rejection_feedback"] = info.get("rejection_feedback") or ""
            rec["rejected_test_command"] = gate.get("test_command") or ""

        # OFFLINE gold-check (gold read only here): gold source fix + submitted test -> expect PASS.
        gold_pass = None
        if rec["submitted"] and not args.no_gold_check:
            gold_src = inst.get("patch") or ""
            if gold_src.strip():
                combined = gold_src.rstrip("\n") + "\n" + submission.rstrip("\n") + "\n"
                try:
                    if shared_verify:
                        gres = run_shared_test_patch_official(
                            env,
                            inst,
                            combined,
                            gate.get("test_command") or "",
                            int(args.install_timeout),
                            skip_install=True,
                            restore_agent_workspace=False,
                        )
                    else:
                        gres = run_test_patch_official(
                            verify_env, inst, combined, gate.get("test_command") or "",
                            int(args.install_timeout), skip_install=True
                        )
                    gold_pass = (gres.get("verdict") == "pass") or (
                        gres.get("passed") and not gres.get("failed") and not gres.get("errors"))
                    rec["gold_verdict"] = gres.get("verdict")
                    rec["gold_diagnostics"] = _gold_diagnostics(gres)
                except Exception as exc:  # noqa: BLE001
                    rec["gold_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        rec["gold_pass"] = bool(gold_pass) if gold_pass is not None else None
        rec["base_fail_gold_pass"] = bool(gate.get("base_clean_fail")) and bool(gold_pass)
        agent.save(trajectory_path, {"info": {"resolve_result": rec}})
        log(f"[done] {iid} submit={rec['submitted']} base_fail={rec['base_clean_fail']} "
            f"gold_pass={rec['gold_pass']} B2G={rec['base_fail_gold_pass']} ({api_base.rsplit(':',1)[-1]})")
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        rec["traceback"] = traceback.format_exc()[-1500:]
        if agent is not None:
            try:
                agent.save(trajectory_path, {"info": {"resolve_result": rec}})
            except Exception as save_exc:  # noqa: BLE001
                rec["trajectory_save_error"] = f"{type(save_exc).__name__}: {str(save_exc)[:300]}"
        log(f"[ERR] {iid}: {rec['error']}")
    finally:
        environments = (env,) if verify_env is env else (env, verify_env)
        for e in environments:
            try:
                if e is not None:
                    e.cleanup()
            except Exception:
                pass
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="output dir (results.jsonl + trajectories)")
    ap.add_argument(
        "--subset",
        default="verified",
        help="dataset alias/path; use 'multilingual' for SWE-bench Multilingual",
    )
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-class", default="litellm_response")
    ap.add_argument("--ports", default="8000,8001,8002,8003,8004,8005,8006,8007")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--tool-choice", default="auto")
    ap.add_argument("--enable-thinking", action="store_true", default=True)
    ap.add_argument("--no-thinking", dest="enable_thinking", action="store_false",
                    help="disable Qwen3.5 thinking (chat_template enable_thinking=False)")
    ap.add_argument("--reasoning-effort", default="medium", help="trapi/codex reasoning effort")
    ap.add_argument(
        "--thinking-mode",
        default="",
        help="local SGLang chat-template thinking_mode (for example DeepSeek-V4 'thinking')",
    )
    ap.add_argument("--max-output-tokens", type=int, default=8192)
    ap.add_argument("--step-limit", type=int, default=60)
    ap.add_argument(
        "--stop-at-first-reject", action="store_true",
        help="bash_testpatch_resolve: end at the first gate rejection (exit_status=SubmitRejected) "
             "instead of retrying. Harvests first-submit-failure states for recovery distillation.",
    )
    ap.add_argument(
        "--instance-timeout",
        type=int,
        default=600,
        help="wall-clock timeout per agent sample in seconds; 0 disables",
    )
    ap.add_argument("--install-timeout", type=int, default=1200)
    ap.add_argument(
        "--allow-network",
        action="store_true",
        help="allow sandbox egress; disabled by default for strict rollout isolation",
    )
    ap.add_argument(
        "--shared-verify",
        action="store_true",
        help="reuse the agent sandbox for transactional verification (one sandbox per worker)",
    )
    ap.add_argument("--no-gold-check", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1, help="runs per selected instance (A/B, multi-sample)")
    ap.add_argument("--ids", default="", help="optional comma list or file of instance_ids")
    ap.add_argument("--reverse", action="store_true", help="process selected instances in reverse dataset order")
    ap.add_argument("--overlay", default="swe_harness/gentest_v2/versions/behavior_contract.yaml")
    args = ap.parse_args()
    try:
        _require_strict_history_isolation()
    except ValueError as exc:
        ap.error(str(exc))

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = out_dir / "results.jsonl"
    try:
        _validate_resume_verify_mode(results_jsonl, args.shared_verify)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    ports = [p.strip() for p in args.ports.split(",") if p.strip()]
    if ports:
        api_bases = [f"http://{args.host}:{p}/v1" for p in ports]
    else:
        # trapi/codex: single shared remote endpoint, auth via TRAPI_BEARER_TOKEN_FILE
        api_bases = [""]

    cfg = get_config_from_spec("gentest")
    sb = get_config_from_spec("swebench_azure_modal")
    cfg = {**sb, **cfg}
    cfg.setdefault("environment", {}).update(sb.get("environment", {}))
    cfg["environment"]["base_url"] = os.environ["SANDBOX_BASE_URL"]
    cfg["environment"]["block_network"] = not args.allow_network

    overlay = yaml.safe_load(Path(args.overlay).read_text())
    agent_cfg_base = overlay.get("agent", {}) or {}
    model_overlay = overlay.get("model", {}) or {}

    instances = load_instances(args.subset, args.split)
    if args.ids:
        wanted = set()
        p = Path(args.ids)
        if p.exists():
            wanted = {x.strip() for x in p.read_text().replace(",", "\n").split() if x.strip()}
        else:
            wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        instances = [i for i in instances if i["instance_id"] in wanted]
    if args.reverse:
        instances.reverse()

    # Expand each selected instance into `repeat` runs. run_key = instance_id (+ #rep<k> when
    # repeat>1) is what we resume on, so A/B / multi-sample runs don't collapse to one row.
    expanded = []
    for inst in instances:
        for k in range(max(1, args.repeat)):
            run_key = inst["instance_id"] if args.repeat <= 1 else f"{inst['instance_id']}#rep{k}"
            expanded.append((run_key, inst))

    done = set()
    if results_jsonl.exists():
        for line in results_jsonl.read_text().splitlines():
            try:
                rec = json.loads(line)
                done.add(rec.get("run_key") or rec["instance_id"])
            except Exception:
                pass
    todo = [(rk, inst) for rk, inst in expanded if rk not in done]
    if args.limit:
        todo = todo[: args.limit]
    log(f"[batch] total={len(expanded)} done={len(done)} todo={len(todo)} "
        f"workers={args.workers} ports={len(ports)} repeat={args.repeat} "
        f"shared_verify={args.shared_verify} reverse={args.reverse}")

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(run_one, inst, args, cfg, agent_cfg_base, model_overlay,
                      api_bases[idx % len(api_bases)], rk): (rk, inst)
            for idx, (rk, inst) in enumerate(todo)
        }
        n = 0
        for fut in concurrent.futures.as_completed(futs):
            rec = fut.result()
            with _write_lock:
                with results_jsonl.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            n += 1
            if n % 10 == 0:
                log(f"[progress] {n}/{len(todo)} elapsed={int(time.time()-t0)}s")

    # summary
    rows = [json.loads(l) for l in results_jsonl.read_text().splitlines() if l.strip()]
    subm = sum(1 for r in rows if r.get("submitted"))
    b2g = sum(1 for r in rows if r.get("base_fail_gold_pass"))
    errs = sum(1 for r in rows if r.get("error"))
    log(f"\n===== SUMMARY ({len(rows)} instances) =====")
    log(f"submitted:            {subm}")
    log(f"BASE_FAIL_GOLD_PASS:  {b2g}")
    log(f"errors:               {errs}")


if __name__ == "__main__":
    main()
