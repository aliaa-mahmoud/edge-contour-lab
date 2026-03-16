"""
CannyController — bridges the UI and the core Canny pipeline.

Responsibilities:
  • Accept a file path or a numpy image array.
  • Call run_canny() with user-supplied parameters.
  • Cache intermediate results so the UI can display each stage.
  • Emit Qt signals so the UI updates reactively.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui  import QImage, QPixmap

from core.canny import run_canny
from core.image_manager import load_image, to_qpixmap


class _CannyWorker(QThread):
    """Runs the (slow) Canny pipeline in a background thread."""

    finished  = pyqtSignal(dict)      # emits the full result dict
    error     = pyqtSignal(str)

    def __init__(self, image: np.ndarray, params: dict, parent=None):
        super().__init__(parent)
        self._image  = image
        self._params = params

    def run(self):
        try:
            result = run_canny(
                self._image,
                kernel_size=self._params.get("kernel_size", 5),
                sigma      =self._params.get("sigma",       1.4),
                low_ratio  =self._params.get("low_ratio",   0.05),
                high_ratio =self._params.get("high_ratio",  0.15),
            )
            self.finished.emit(result)
        except Exception as exc:                        # noqa: BLE001
            self.error.emit(str(exc))


class CannyController(QObject):
    """
    Public API consumed by the UI layer.

    Signals
    -------
    result_ready(dict)  — emitted when Canny finishes; dict contains all stages
    status_message(str) — short text for a status-bar label
    error_occurred(str) — human-readable error string
    """

    result_ready   = pyqtSignal(dict)
    status_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_result: dict | None = None
        self._worker: _CannyWorker | None = None

    # ── public slots (called by UI) ─────────────────────────────────────────

    @pyqtSlot(str, dict)
    def process_image_path(self, path: str, params: dict):
        """Load *path* from disk then run the pipeline with *params*."""
        try:
            image = load_image(path)
        except Exception as exc:                        # noqa: BLE001
            self.error_occurred.emit(f"Failed to load image: {exc}")
            return
        self._run(image, params)

    @pyqtSlot(object, dict)
    def process_image_array(self, image: np.ndarray, params: dict):
        """Run the pipeline on an already-loaded numpy array."""
        self._run(image, params)

    # ── internal helpers ────────────────────────────────────────────────────

    def _run(self, image: np.ndarray, params: dict):
        if self._worker and self._worker.isRunning():
            self.status_message.emit("Already processing — please wait…")
            return

        self.status_message.emit("Running Canny edge detection…")
        self._worker = _CannyWorker(image, params, parent=self)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    @pyqtSlot(dict)
    def _on_finished(self, result: dict):
        self._last_result = result
        self.status_message.emit("Canny complete.")
        self.result_ready.emit(result)

    @pyqtSlot(str)
    def _on_error(self, msg: str):
        self.error_occurred.emit(msg)

    # ── convenience getters ─────────────────────────────────────────────────

    def get_stage_pixmap(self, stage: str) -> QPixmap | None:
        """
        Return a QPixmap for *stage* (e.g. 'edges', 'magnitude', 'blurred').
        Returns None if no result is cached yet.
        """
        if self._last_result is None:
            return None
        arr = self._last_result.get(stage)
        if arr is None:
            return None
        return to_qpixmap(arr)

    @property
    def last_result(self) -> dict | None:
        return self._last_result

    @property
    def edge_map(self) -> np.ndarray | None:
        """Final binary edge map (0/255 uint8), or None."""
        if self._last_result is None:
            return None
        return self._last_result.get("edges")