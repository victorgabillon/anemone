"""Types used by backup policies."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BackupResult:
    """Result flags produced by a backup policy execution."""

    value_changed: bool
    pv_changed: bool
    over_changed: bool

