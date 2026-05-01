from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import inspect
import traceback

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


@dataclass(frozen=True)
class WorkerMessage:
    level: str
    text: str


class WorkerSignals(QObject):
    started = Signal(str)
    message = Signal(object)
    progress = Signal(str, int, int, str)
    finished = Signal(str, object)
    failed = Signal(str, str)


class FunctionWorker(QRunnable):
    def __init__(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.name = name
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        self.signals.started.emit(self.name)
        try:
            kwargs = dict(self.kwargs)
            if _accepts_progress_callback(self.fn):
                kwargs["progress_callback"] = self._emit_progress
            result = self.fn(*self.args, **kwargs)
        except Exception:
            self.signals.failed.emit(self.name, traceback.format_exc())
            return
        self.signals.finished.emit(self.name, result)

    def _emit_progress(self, done: int, total: int, text: str = "") -> None:
        self.signals.progress.emit(self.name, int(done), int(total), str(text))


def _accepts_progress_callback(fn: Callable[..., Any]) -> bool:
    try:
        return "progress_callback" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
