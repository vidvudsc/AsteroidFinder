from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import traceback

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


@dataclass(frozen=True)
class WorkerMessage:
    level: str
    text: str


class WorkerSignals(QObject):
    started = Signal(str)
    message = Signal(object)
    finished = Signal(str, object)
    failed = Signal(str, str)
    cancelled = Signal(str)


class FunctionWorker(QRunnable):
    def __init__(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.name = name
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @Slot()
    def run(self) -> None:
        if self._cancelled:
            self.signals.cancelled.emit(self.name)
            return
        self.signals.started.emit(self.name)
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            self.signals.failed.emit(self.name, traceback.format_exc())
            return
        if self._cancelled:
            self.signals.cancelled.emit(self.name)
            return
        self.signals.finished.emit(self.name, result)
