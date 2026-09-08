"""PrefixReplayAgent — teacher-prefix transplant probe.

Starts the student from the state reached by replaying the first K actions of a
teacher trajectory (e.g. Claude Opus 4.8) on the same instance, then hands
control to the student's normal agent loop.

Design (v1):
- Teacher actions are REPLAYED in the fresh sandbox (env state is exactly
  teacher-after-K). Observations shown to the student are the REPLAYED ones,
  not the teacher's recorded ones, so context always matches env state.
- Teacher turns are re-rendered in the student's own message format
  (tool-call assistant messages + tool observation messages), so the student
  perceives the prefix as its own history.
- `include_teacher_thought=True` (default): the teacher's visible THOUGHT text
  is kept as the assistant content — a full-fidelity transcript transplant.
  Set False for the actions+observations-only variant (no reasoning leak).
- Replayed turns do NOT count against step_limit (the student keeps its full
  budget; the probe measures what the student does WITH the head start).
- Instance id is derived from config.output_path (<outdir>/<iid>/<iid>.traj.json),
  matching how mini-extra swebench lays out outputs.
- If the teacher trajectory is missing or replay raises, the agent falls back
  to a vanilla run and records why in info.prefix_replay.

Config (extends AgentConfig):
    agent_class: minisweagent.agents.extra.prefix_replay.PrefixReplayAgent
    teacher_traj_dir: /path/to/teacher/trajectories
    prefix_fraction: 0.5       # fraction of teacher assistant-turns to replay
    prefix_turns: 0            # absolute override; 0 = use fraction
    include_teacher_thought: true
"""

import json
import re
from pathlib import Path

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import InterruptAgentFlow

_FENCE_RE = re.compile(r"```(?:mswea_bash_command|bash|sh)?\n(.*?)```", re.DOTALL)
_SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


class PrefixReplayAgentConfig(AgentConfig):
    teacher_traj_dir: str = ""
    """Directory containing <instance_id>/<instance_id>.traj.json teacher trajectories."""
    prefix_fraction: float = 0.5
    """Fraction of the teacher's assistant turns to replay (rounded down, >=1)."""
    prefix_turns: int = 0
    """Absolute number of teacher turns to replay; overrides prefix_fraction when > 0."""
    prefix_turns_file: str = ""
    """Optional JSON file mapping instance_id -> turns to replay (per-instance adaptive
    cut, e.g. 'just before the teacher's first edit'). Takes precedence over
    prefix_turns/prefix_fraction when the instance is present in the map."""
    include_teacher_thought: bool = True
    """Keep the teacher's THOUGHT text as the assistant content of replayed turns."""


class PrefixReplayAgent(DefaultAgent):
    def __init__(self, model, env, **kwargs):
        super().__init__(model, env, config_class=PrefixReplayAgentConfig, **kwargs)
        self._prefix_info: dict = {"applied": False}

    # -- helpers -------------------------------------------------------------

    def _instance_id(self) -> str:
        # Preferred: the swebench runner's progress wrapper sets self.instance_id.
        iid = getattr(self, "instance_id", "") or ""
        if iid:
            return iid.split("__sample_")[0]
        if self.config.output_path is not None:
            return Path(self.config.output_path).parent.name
        raise ValueError("PrefixReplayAgent needs instance_id or config.output_path to find the teacher traj")

    def _load_teacher_turns(self) -> list[dict]:
        iid = self._instance_id()
        path = Path(self.config.teacher_traj_dir) / iid / f"{iid}.traj.json"
        if not path.exists():
            raise FileNotFoundError(f"teacher traj not found: {path}")
        traj = json.loads(path.read_text())
        return [m for m in traj.get("messages", []) if m.get("role") == "assistant"]

    @staticmethod
    def _teacher_actions(turn: dict) -> list[str]:
        acts = turn.get("extra", {}).get("actions") or []
        commands = [a.get("command", "") for a in acts if a.get("command")]
        if not commands:
            content = turn.get("content") or ""
            commands = [m.strip() for m in _FENCE_RE.findall(content)]
        return [c for c in commands if c]

    @staticmethod
    def _teacher_thought(turn: dict) -> str:
        content = turn.get("content") or ""
        return _FENCE_RE.sub("", content).strip()

    # -- replay --------------------------------------------------------------

    def _replay_prefix(self) -> None:
        turns = self._load_teacher_turns()
        n = len(turns)
        if n < 2:
            raise ValueError(f"teacher traj too short to take a prefix ({n} turns)")
        k = self.config.prefix_turns or max(1, int(n * self.config.prefix_fraction))
        if self.config.prefix_turns_file:
            per_instance = json.loads(Path(self.config.prefix_turns_file).read_text())
            iid = self._instance_id()
            if iid in per_instance:
                k = max(1, int(per_instance[iid]))
        k = min(k, n - 1)  # never replay the teacher's final (submitting) turn

        replayed = 0
        for idx, turn in enumerate(turns[:k]):
            commands = self._teacher_actions(turn)
            # Hard guard: never replay a submission action mid-prefix.
            commands = [c for c in commands if _SUBMIT_MARKER not in c]
            if not commands:
                continue
            actions = [
                {"command": cmd, "tool_call_id": f"prefix_{idx}_{j}"} for j, cmd in enumerate(commands)
            ]
            tool_calls = [
                {
                    "index": j,
                    "id": a["tool_call_id"],
                    "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"command": a["command"]})},
                }
                for j, a in enumerate(actions)
            ]
            content = self._teacher_thought(turn) if self.config.include_teacher_thought else ""
            assistant_msg = {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
                "extra": {"actions": actions, "prefix_replay": True},
            }
            outputs = [self.env.execute(a) for a in actions]
            observation_msgs = self.model.format_observation_messages(
                assistant_msg, outputs, self.get_template_vars()
            )
            self.add_messages(assistant_msg, *observation_msgs)
            replayed += 1

        self._prefix_info = {
            "applied": True,
            "teacher_turns_total": n,
            "teacher_turns_requested": k,
            "teacher_turns_replayed": replayed,
            "include_teacher_thought": self.config.include_teacher_thought,
            "teacher_traj_dir": self.config.teacher_traj_dir,
        }
        self.logger.info(
            f"prefix replay: {replayed}/{k} teacher turns replayed (teacher total {n})"
        )

    # -- run loop (mirrors DefaultAgent.run with a replay phase inserted) -----

    def run(self, task: str = "", **kwargs) -> dict:
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )
        try:
            self._replay_prefix()
        except InterruptAgentFlow as e:
            # A replayed action triggered submission/limits — record and stop.
            self.add_messages(*e.messages)
            self._prefix_info = {"applied": False, "error": f"replay interrupted: {type(e).__name__}"}
        except Exception as e:
            # Fall back to a vanilla run; record why.
            self._prefix_info = {"applied": False, "error": f"{type(e).__name__}: {e}"}
            self.logger.warning(f"prefix replay failed, running vanilla: {e}")
        finally:
            self.save(self.config.output_path)

        while True:
            if self.messages[-1].get("role") == "exit":
                break
            try:
                self.step()
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
        return self.messages[-1].get("extra", {})

    def serialize(self, *extra_dicts) -> dict:
        return super().serialize({"info": {"prefix_replay": self._prefix_info}}, *extra_dicts)
