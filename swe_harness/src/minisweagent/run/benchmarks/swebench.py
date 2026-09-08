#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-bench instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import random
import re
import threading
import time
import traceback
from pathlib import Path
from queue import Empty

import typer
from jinja2 import StrictUndefined, Template
from rich.live import Live

from minisweagent import Environment
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c swebench.yaml <other options>[/bold green]

Multiple configs will be recursively merged.

Examples:

[bold red]-c model.model_kwargs.temperature=0[/bold red] [red]You forgot to add the default config file! See above.[/red]

[bold green]-c swebench.yaml -c model.model_kwargs.temperature=0.5[/bold green]

[bold green]-c swebench.yaml -c agent.max_iterations=50[/bold green]
"""

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench.yaml"

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
    "rebench": "nebius/SWE-rebench",
    "rebench_v2": "nebius/SWE-rebench-V2",
    "pro": "ScaleAI/SWE-bench_Pro",
}

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
_OUTPUT_FILE_LOCK = threading.Lock()


class ProgressTrackingAgent(DefaultAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", deadline: float = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id
        self.deadline = deadline

    def step(self) -> dict:
        """Override step to provide progress updates."""
        if self.deadline and time.monotonic() > self.deadline:
            raise TimeoutError(f"Instance exceeded wall-clock deadline")
        self.progress_manager.update_instance_status(self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})")
        return super().step()


class _NoopProgressManager:
    def update_instance_status(self, instance_id: str, status: str) -> None:
        return


def _resolve_per_instance_api_base(config: dict, instance_id: str) -> None:
    """If model_kwargs.api_base contains a `{port}` or `{port-N}` placeholder (e.g.
    `http://localhost:{port-2}/v1`), hash the instance_id onto N ports starting at 8000
    (N defaults to 8) and substitute. If MSWEA_ROUTER_PORTS is set, use that explicit
    comma-separated port list instead. No-op otherwise."""
    model_kwargs = config.setdefault("model", {}).setdefault("model_kwargs", {})
    api_base = model_kwargs.get("api_base")
    if not isinstance(api_base, str):
        return
    match = re.search(r"\{port(?:-(\d+))?\}", api_base)
    if not match:
        return
    ports_env = os.getenv("MSWEA_ROUTER_PORTS")
    if ports_env:
        ports = [int(port.strip()) for port in ports_env.replace(" ", ",").split(",") if port.strip()]
    else:
        n_ports = int(match.group(1)) if match.group(1) else 8
        ports = list(range(8000, 8000 + n_ports))
    if not ports:
        raise ValueError("MSWEA_ROUTER_PORTS must contain at least one port")
    h = int(hashlib.md5(instance_id.encode()).hexdigest(), 16)
    chosen = ports[h % len(ports)]
    model_kwargs["api_base"] = api_base[: match.start()] + str(chosen) + api_base[match.end() :]


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None) or instance.get("docker_image", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"mirror.gcr.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    if image_name.startswith("docker.io/"):
        image_name = image_name.replace("docker.io/", "mirror.gcr.io/")  # redirect docker.io to mirror.gcr.io
    elif not image_name.startswith("mirror.gcr.io/") and not image_name.startswith("docker://"):
        image_name = "mirror.gcr.io/" + image_name  # ensure all images are pulled from mirror.gcr.io for better performance
    return image_name


def get_sb_environment(config: dict, instance: dict) -> Environment:
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    if instance.get("_swebench_pro"):
        from minisweagent.run.benchmarks.swebench_pro_verify_azure_modal import DOCKER_WORKDIR

        env_config["cwd"] = DOCKER_WORKDIR
        if env_config["environment_class"] == "docker":
            run_args = list(env_config.get("run_args", ["--rm"]))
            if "--entrypoint" not in run_args:
                run_args.extend(["--entrypoint", ""])
            env_config["run_args"] = run_args
    image_name = get_swebench_docker_image_name(instance)
    if env_config["environment_class"] in ["docker", "swerex_modal", "azure_modal"]:
        env_config["image"] = image_name
    elif env_config["environment_class"] in ["singularity", "contree"]:
        env_config["image"] = "docker://" + image_name

    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str, sample_idx: int = 0):
    """Update the output JSON file with results from a single instance sample."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        samples = output_data.get(instance_id, {}).get("samples", [])
        # Replace existing sample_idx or append
        samples = [s for s in samples if s.get("sample_idx") != sample_idx]
        samples.append({"sample_idx": sample_idx, "model_patch": result})
        samples.sort(key=lambda s: s["sample_idx"])
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": samples[0]["model_patch"],
            "samples": samples,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str, sample_idx: int | None = None):
    """Remove an instance (or a single sample) from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id not in output_data:
            return
        if sample_idx is None:
            del output_data[instance_id]
        else:
            samples = output_data[instance_id].get("samples", [])
            samples = [s for s in samples if s.get("sample_idx") != sample_idx]
            if not samples:
                del output_data[instance_id]
            else:
                output_data[instance_id]["samples"] = samples
                output_data[instance_id]["model_patch"] = samples[0]["model_patch"]
        output_path.write_text(json.dumps(output_data, indent=2))


def get_existing_samples(output_path: Path) -> set[tuple[str, int]]:
    """Return set of (instance_id, sample_idx) pairs that already exist in preds."""
    if not output_path.exists():
        return set()
    output_data = json.loads(output_path.read_text())
    existing = set()
    for instance_id, entry in output_data.items():
        for s in entry.get("samples", []):
            existing.add((instance_id, s["sample_idx"]))
        if "samples" not in entry:
            existing.add((instance_id, 0))
    return existing


_COMPLETED_EXIT_STATUSES = {"Submitted"}


def get_existing_samples_from_trajs(output_dir: Path) -> set[tuple[str, int]]:
    """Return set of (instance_id, sample_idx) pairs by scanning trajectory files.

    Only counts trajectories with a completed exit_status (Submitted, LimitsExceeded).
    Incomplete or errored trajectories are ignored so they get retried.
    """
    existing = set()
    for traj_file in output_dir.rglob("**/*.traj.json"):
        try:
            data = json.loads(traj_file.read_text())
            exit_status = data.get("info", {}).get("exit_status")
            if exit_status not in _COMPLETED_EXIT_STATUSES:
                continue
        except (json.JSONDecodeError, OSError):
            continue
        name = traj_file.stem.removesuffix(".traj")  # e.g. "id" or "id.sample_2"
        if ".sample_" in name:
            instance_id, _, idx = name.rpartition(".sample_")
            existing.add((instance_id, int(idx)))
        else:
            existing.add((name, 0))
    return existing


def _run_instance_subprocess(
    instance: dict,
    output_dir: Path,
    config: dict,
    sample_idx: int,
    instance_timeout: int,
    result_queue: multiprocessing.Queue,
) -> None:
    """Inner worker executed in a subprocess for hard wall-clock timeout enforcement."""
    import copy
    # --- L2_ARM hook (additive) ---
    import os as _os

    _arm = _os.environ.get("L2_ARM", "").strip()
    if _arm and _arm != "baseline":
        import sys as _sys

        _mods_path = _os.environ.get("L2_MODS_PATH", "").strip()
        if not _mods_path:
            raise RuntimeError("L2_MODS_PATH must be set when L2_ARM selects a non-baseline arm")
        if _mods_path not in _sys.path:
            _sys.path.insert(0, _mods_path)
        import run_l2 as _l2

        if _arm in _l2._MOD_MAP:
            globals()["ProgressTrackingAgent"] = _l2._build_combined(_l2._MOD_MAP[_arm])
    # --- end L2_ARM hook ---

    config = copy.deepcopy(config)  # isolate per-thread config to avoid race conditions
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    sample_suffix = f".sample_{sample_idx}" if sample_idx > 0 else ""
    traj_filename = f"{instance_id}{sample_suffix}.traj.json"
    task_label = f"{instance_id}#{sample_idx}" if sample_idx > 0 else instance_id
    # deadline for ProgressTrackingAgent step-level check (best-effort)
    deadline = time.monotonic() + instance_timeout if instance_timeout > 0 else 0

    _resolve_per_instance_api_base(config, instance_id)
    model = get_model(config=config.get("model", {}))
    task = instance["problem_statement"]
    progress_manager = _NoopProgressManager()

    agent = None
    exit_status = None
    result = None
    extra_info = {}

    try:
        env = get_sb_environment(config, instance)
        agent = ProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=task_label,
            deadline=deadline,
            **config.get("agent", {}),
        )
        info = agent.run(task)
        exit_status = info.get("exit_status")
        result = info.get("submission")
    except Exception as e:
        logger.error(f"Error processing instance {task_label}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, ""
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    finally:
        if agent is not None:
            traj_path = instance_dir / traj_filename
            agent.save(
                traj_path,
                {
                    "info": {
                        "exit_status": exit_status,
                        "submission": result,
                        "sample_idx": sample_idx,
                        **extra_info,
                    },
                    "instance_id": instance_id,
                },
            )
            logger.info(f"Saved trajectory to '{traj_path}'")
        result_queue.put(
            {
                "exit_status": exit_status,
                "result": result,
                "model_name": model.config.model_name,
                "extra_info": extra_info,
            }
        )


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    sample_idx: int = 0,
    no_preds: bool = False,
    instance_timeout: int = 0,
) -> None:
    """Process a single SWEBench instance (one sample)."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    sample_suffix = f".sample_{sample_idx}" if sample_idx > 0 else ""
    traj_filename = f"{instance_id}{sample_suffix}.traj.json"
    task_label = f"{instance_id}#{sample_idx}" if sample_idx > 0 else instance_id

    progress_manager.on_instance_start(task_label)
    progress_manager.update_instance_status(task_label, "Pulling/starting environment")

    if not no_preds:
        remove_from_preds_file(output_dir / "preds.json", instance_id, sample_idx)
    traj_path = instance_dir / traj_filename
    traj_path.unlink(missing_ok=True)

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_run_instance_subprocess,
        args=(instance, output_dir, config, sample_idx, instance_timeout, result_queue),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=instance_timeout if instance_timeout > 0 else None)

    if proc.is_alive():
        logger.warning(f"Instance {task_label} exceeded wall-clock timeout ({instance_timeout}s), skipping.")
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive() and hasattr(proc, "kill"):
            proc.kill()
            proc.join(timeout=5)
        exit_status = "WallClockTimeout"
        result = ""
        model_name = config.get("model", {}).get("model_name", "unknown")
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj_path.write_text(
            json.dumps(
                {
                    "info": {"exit_status": exit_status, "submission": result, "sample_idx": sample_idx},
                    "instance_id": instance_id,
                },
                indent=2,
            )
        )
    else:
        try:
            child_result = result_queue.get_nowait()
        except Empty:
            child_result = None

        if child_result is None:
            exit_status = "MissingResult"
            result = ""
            model_name = config.get("model", {}).get("model_name", "unknown")
            logger.warning(f"No result returned for {task_label}")
        else:
            exit_status = child_result.get("exit_status")
            result = child_result.get("result", "")
            model_name = child_result.get("model_name") or config.get("model", {}).get("model_name", "unknown")

    if not no_preds:
        update_preds_file(output_dir / "preds.json", instance_id, model_name, result, sample_idx)
    progress_manager.on_instance_end(task_label, exit_status)


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


def prepare_dataset_instances(instances: list[dict], dataset_path: str) -> list[dict]:
    """Apply dataset-specific execution and prompt metadata."""
    from minisweagent.run.benchmarks.swebench_pro_verify_azure_modal import DATASET_NAME, prepare_instance

    if dataset_path == DATASET_NAME:
        return [prepare_instance(instance) for instance in instances]
    return instances


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset (including 'pro') or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT, rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
    n_samples: int = typer.Option(1, "-n", "--n-samples", help="Number of samples (runs) per instance for pass@k evaluation", rich_help_panel="Basic"),
    no_preds: bool = typer.Option(False, "--no-preds", help="Disable writing to preds.json", rich_help_panel="Advanced"),
    instance_timeout: int = typer.Option(0, "--instance-timeout", help="Hard timeout in seconds per instance (0=no limit)", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    from datasets import load_dataset

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = prepare_dataset_instances(list(load_dataset(dataset_path, split=split)), dataset_path)

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)

    # Build work items: (instance, sample_idx) pairs
    work_items: list[tuple[dict, int]] = []
    if not redo_existing:
        existing_samples = get_existing_samples(output_path / "preds.json")
        existing_samples |= get_existing_samples_from_trajs(output_path)
    else:
        existing_samples = set()
    for instance in instances:
        for s_idx in range(n_samples):
            if (instance["instance_id"], s_idx) not in existing_samples:
                work_items.append((instance, s_idx))
    if len(work_items) < len(instances) * n_samples:
        logger.info(f"Skipping {len(instances) * n_samples - len(work_items)} existing samples")
    logger.info(f"Running {len(work_items)} jobs ({len(instances)} instances x {n_samples} samples)...")

    logger.info(f"Building agent config from specs: {config_spec}")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "environment": {"environment_class": environment_class or UNSET},
        "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
    })
    config = recursive_merge(*configs)

    progress_manager = RunBatchProgressManager(len(work_items), output_path / f"exit_statuses_{time.time()}.yaml")

    if workers <= 1:
        live_ctx = Live(progress_manager.render_group, refresh_per_second=4)
    else:
        progress_manager._main_progress_bar.live.redirect_stdout = False
        progress_manager._main_progress_bar.live.redirect_stderr = False
        live_ctx = progress_manager._main_progress_bar
    with live_ctx:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        futures = {}
        for instance, s_idx in work_items:
            iid = instance["instance_id"]
            task_label = f"{iid}#{s_idx}" if s_idx > 0 else iid
            future = executor.submit(process_instance, instance, output_path, config, progress_manager, s_idx, no_preds, instance_timeout)
            futures[future] = task_label
        try:
            done = set()
            while len(done) < len(futures):
                try:
                    for future in concurrent.futures.as_completed(futures, timeout=2.0):
                        done.add(future)
                        try:
                            future.result()
                        except concurrent.futures.CancelledError:
                            pass
                        except Exception as e:
                            task_label = futures[future]
                            logger.error(f"Error in future for {task_label}: {e}", exc_info=True)
                            progress_manager.on_uncaught_exception(task_label, e)
                except concurrent.futures.TimeoutError:
                    pass  # No futures completed in the last 2s, loop back to stay responsive
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Shutting down immediately...")
            executor.shutdown(wait=False, cancel_futures=True)
            os._exit(1)
        finally:
            executor.shutdown(wait=False)


if __name__ == "__main__":
    app()
