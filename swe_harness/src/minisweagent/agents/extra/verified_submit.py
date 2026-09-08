"""VerifiedSubmitAgent — execution-grounded submission gate (#105 rung B).

Blocks COMPLETE_TASK submission until the agent has demonstrated, within this
session, an assertion-bearing repro script that (a) FAILED (rc!=0) at least
once — bug reproduced — and (b) subsequently PASSED (rc==0) — fix verified.
This is hard control-flow, not advice: the gate swallows the Submitted
interrupt and injects a refusal message until satisfied (or max_rejections).

Config (extends AgentConfig):
    agent_class: minisweagent.agents.extra.verified_submit.VerifiedSubmitAgent
    repro_pattern: "repro"     # substring of the script path that counts
    max_rejections: 3          # after this many blocks, let submission through
"""

import re

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import Submitted

_EXEC_RE_TMPL = r"python3?\s+\S*{pat}\S*\.py"


class VerifiedSubmitAgentConfig(AgentConfig):
    repro_pattern: str = "repro"
    """Substring that identifies the repro script in an executed command."""
    max_rejections: int = 3
    """After this many blocked submissions, allow submission (avoid infinite loops)."""


class VerifiedSubmitAgent(DefaultAgent):
    def __init__(self, model, env, **kwargs):
        super().__init__(model, env, config_class=VerifiedSubmitAgentConfig, **kwargs)
        self._exec_re = re.compile(_EXEC_RE_TMPL.format(pat=re.escape(self.config.repro_pattern)), re.I)
        self._saw_fail = False
        self._saw_pass_after_fail = False
        self._rejections = 0

    def _scan_turn(self):
        """Update gate state from the most recent assistant action + observation."""
        msgs = self.messages
        for i in range(len(msgs) - 1, max(len(msgs) - 4, -1), -1):
            m = msgs[i]
            if m.get("role") != "assistant":
                continue
            cmds = " ; ".join(a.get("command", "") for a in (m.get("extra", {}).get("actions") or []))
            if not self._exec_re.search(cmds):
                return
            for o in msgs[i + 1:]:
                rc = o.get("extra", {}).get("returncode")
                if rc is None:
                    continue
                if rc != 0:
                    self._saw_fail = True
                elif self._saw_fail:
                    self._saw_pass_after_fail = True
                return
            return

    def _gate_ok(self) -> bool:
        return self._saw_pass_after_fail or self._rejections >= self.config.max_rejections

    def step(self) -> dict:
        try:
            out = super().step()
            self._scan_turn()
            return out
        except Submitted:
            self._scan_turn()
            if self._gate_ok():
                raise
            self._rejections += 1
            self.add_messages(
                self.model.format_message(
                    role="user",
                    content=(
                        "SUBMISSION BLOCKED by the verification gate "
                        f"({self._rejections}/{self.config.max_rejections}).\n\n"
                        "Before submitting you MUST demonstrate an execution-grounded check:\n"
                        f"1. Write a standalone script whose filename contains '{self.config.repro_pattern}' "
                        "(e.g. /tmp/repro.py) that reproduces the reported bug and exits NON-ZERO while "
                        "the bug is present (use assert / sys.exit(1) — printing is not verification).\n"
                        "2. Run it and observe it FAIL (this proves it reproduces the issue"
                        + (" — if your fix is already applied, temporarily revert with `git stash`, run it, then `git stash pop`)" if self._saw_fail is False else ")") + ".\n"
                        "3. Ensure your fix is applied, run the script again, and observe it PASS (exit 0).\n"
                        "4. Then submit again with the usual final command.\n\n"
                        "Status: repro-failure observed: "
                        f"{self._saw_fail}; pass-after-failure observed: {self._saw_pass_after_fail}."
                    ),
                )
            )
            return {}

    def serialize(self, *extra_dicts) -> dict:
        return super().serialize(
            {"info": {"verified_submit": {
                "saw_fail": self._saw_fail,
                "saw_pass_after_fail": self._saw_pass_after_fail,
                "rejections": self._rejections,
            }}},
            *extra_dicts,
        )
