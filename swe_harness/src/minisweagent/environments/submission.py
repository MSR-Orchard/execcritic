from minisweagent.exceptions import Submitted


SUBMISSION_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def raise_if_submitted(output: dict) -> None:
    """Raise Submitted when a successful command emits the submission marker."""
    if output.get("returncode") != 0:
        return

    lines = output.get("output", "").splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.strip() == SUBMISSION_MARKER:
            submission = "".join(lines[i + 1 :])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )
