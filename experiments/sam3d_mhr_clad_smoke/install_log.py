"""Structured install-command failure logging -- Task 03C.

Fixes a Task 03B bug: the notebook's `run_shell()` returned a plain bool,
and only ONE specific check (the post-install torch/torchvision pin
verification) ever called `state.add_failure(...)`. When the real Colab
run's `chumpy`/Detectron2 wheel builds failed, `dependencies_installed`
correctly became `False`, but nothing recorded *why* -- the notebook's
final decision-gate cell printed "Recorded failure categories: none"
despite two clear, observed build failures.

Every command that can fail during environment setup now gets its own
record via `InstallLog.record()`: the exact command, its return code, the
tail of its stderr, and (when known) which `FailureCategory` it maps to.
Nothing about this changes what the notebook actually runs -- it only
makes failures visible, which is the whole point of a diagnostic
notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CommandResult:
    command: str
    ok: bool
    returncode: int | None = None
    stderr_tail: str | None = None
    category: str | None = None  # a decision_gate.FailureCategory value, if not ok


@dataclass
class InstallLog:
    records: list[CommandResult] = field(default_factory=list)

    def record(
        self,
        command: str,
        ok: bool,
        returncode: int | None = None,
        stderr_tail: str | None = None,
        category: str | None = None,
    ) -> CommandResult:
        """Append one command's outcome. `category` should only be set when
        `ok` is False -- a successful command has no failure category."""
        result = CommandResult(
            command=command,
            ok=ok,
            returncode=returncode,
            stderr_tail=(stderr_tail[-1500:] if stderr_tail else None) if not ok else None,
            category=category if not ok else None,
        )
        self.records.append(result)
        return result

    @property
    def failures(self) -> list[CommandResult]:
        return [r for r in self.records if not r.ok]

    def categories(self) -> list[str]:
        """Distinct failure category values recorded so far, in first-seen
        order -- ready to feed into PipelineState.add_failure() in a loop."""
        seen: list[str] = []
        for r in self.failures:
            if r.category and r.category not in seen:
                seen.append(r.category)
        return seen

    def summary(self) -> list[dict]:
        """JSON-serializable list of every failed command's full record --
        this is what the notebook prints/saves so an install failure is
        never silently swallowed into just a boolean."""
        return [
            {
                "command": r.command,
                "returncode": r.returncode,
                "stderr_tail": r.stderr_tail,
                "category": r.category,
            }
            for r in self.failures
        ]

    def has_failures(self) -> bool:
        return len(self.failures) > 0
