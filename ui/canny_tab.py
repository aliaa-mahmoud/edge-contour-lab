"""
CannyTab — PyQt5 widget for Member 1 (Canny edge detection).

Layout
------
  Left panel  : parameter controls + run button
  Right panel : image viewer with tabs for each pipeline stage
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QPushButton, QSlider, QDoubleSpinBox, QSpinBox,
    QGroupBox, QTabWidget, QScrollArea, QFileDialog,
    QSizePolicy, QSplitter, QFrame, QComboBox
)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui  import QPixmap, QFont, QPainter

from controllers.canny_controller import CannyController
from core.image_manager import load_image, to_qpixmap, save_image


class _ImageLabel(QLabel):
    """A QLabel that scales its pixmap to fill available space."""

    def __init__(self, placeholder="No image loaded", parent=None):
        super().__init__(placeholder, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 150)
        self.setStyleSheet(
            "background:#1e1e2e; color:#6c6c8a; border-radius:6px;"
        )
        self._pixmap_orig: QPixmap | None = None

    def set_pixmap(self, pix: QPixmap):
        self._pixmap_orig = pix
        self.setText("")
        self._refresh()

    def resizeEvent(self, event):
        self._refresh()
        super().resizeEvent(event)

    def _refresh(self):
        if self._pixmap_orig is None:
            return
        scaled = self._pixmap_orig.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        super().setPixmap(scaled)


class CannyTab(QWidget):

    STAGES = [
        ("Original",     "original"),
        ("Grayscale",    "gray"),
        ("Blurred",      "blurred"),
        ("Magnitude",    "magnitude"),
        ("Suppressed",   "suppressed"),
        ("Thresholded",  "thresholded"),
        ("Edges",        "edges"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)

        self.controller = CannyController(self)
        self._current_image: np.ndarray | None = None
        self._results: dict | None = None

        self._build_ui()
        self._connect_signals()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([280, 960])

    # ── Left: controls ─────────────────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(280)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Image load
        load_grp = QGroupBox("Image")
        load_lay  = QVBoxLayout(load_grp)

        self._load_btn = QPushButton("📂  Open Image…")
        self._load_btn.setCursor(Qt.PointingHandCursor)
        self._img_name_lbl = QLabel("No image selected")
        self._img_name_lbl.setWordWrap(True)
        self._img_name_lbl.setStyleSheet("color:#888; font-size:11px;")

        load_lay.addWidget(self._load_btn)
        load_lay.addWidget(self._img_name_lbl)
        layout.addWidget(load_grp)

        # Gaussian parameters
        gauss_grp = QGroupBox("Gaussian Blur")
        gauss_lay  = QGridLayout(gauss_grp)

        gauss_lay.addWidget(QLabel("Kernel size:"), 0, 0)
        self._kernel_spin = QSpinBox()
        self._kernel_spin.setRange(3, 21)
        self._kernel_spin.setSingleStep(2)
        self._kernel_spin.setValue(5)
        gauss_lay.addWidget(self._kernel_spin, 0, 1)

        gauss_lay.addWidget(QLabel("Sigma:"), 1, 0)
        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setRange(0.1, 10.0)
        self._sigma_spin.setSingleStep(0.1)
        self._sigma_spin.setValue(1.4)
        gauss_lay.addWidget(self._sigma_spin, 1, 1)

        layout.addWidget(gauss_grp)

        # Threshold parameters
        thresh_grp = QGroupBox("Double Threshold")
        thresh_lay  = QGridLayout(thresh_grp)

        thresh_lay.addWidget(QLabel("Low ratio:"), 0, 0)
        self._low_spin = QDoubleSpinBox()
        self._low_spin.setRange(0.01, 0.49)
        self._low_spin.setSingleStep(0.01)
        self._low_spin.setDecimals(3)
        self._low_spin.setValue(0.05)
        thresh_lay.addWidget(self._low_spin, 0, 1)

        thresh_lay.addWidget(QLabel("High ratio:"), 1, 0)
        self._high_spin = QDoubleSpinBox()
        self._high_spin.setRange(0.02, 0.99)
        self._high_spin.setSingleStep(0.01)
        self._high_spin.setDecimals(3)
        self._high_spin.setValue(0.15)
        thresh_lay.addWidget(self._high_spin, 1, 1)

        layout.addWidget(thresh_grp)

        # Run button
        self._run_btn = QPushButton("▶  Run Canny")
        self._run_btn.setEnabled(False)
        self._run_btn.setMinimumHeight(40)
        font = self._run_btn.font()
        font.setBold(True)
        self._run_btn.setFont(font)
        self._run_btn.setStyleSheet(
            "QPushButton { background:#5c6bc0; color:white; border-radius:6px; }"
            "QPushButton:hover { background:#7986cb; }"
            "QPushButton:disabled { background:#333; color:#666; }"
        )
        layout.addWidget(self._run_btn)

        # Save button
        self._save_btn = QPushButton("💾  Save Edge Map")
        self._save_btn.setEnabled(False)
        self._save_btn.setStyleSheet(
            "QPushButton { background:#2e7d32; color:white; border-radius:6px; }"
            "QPushButton:hover { background:#388e3c; }"
            "QPushButton:disabled { background:#333; color:#666; }"
        )
        layout.addWidget(self._save_btn)

        # Stats box
        stats_grp = QGroupBox("Statistics")
        stats_lay  = QVBoxLayout(stats_grp)
        self._stats_lbl = QLabel("—")
        self._stats_lbl.setWordWrap(True)
        self._stats_lbl.setStyleSheet("font-size:11px; color:#aaa;")
        stats_lay.addWidget(self._stats_lbl)
        layout.addWidget(stats_grp)

        layout.addStretch()
        return panel

    # ── Right: image viewer ────────────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Stage selector
        header = QHBoxLayout()
        header.addWidget(QLabel("Pipeline stage:"))
        self._stage_combo = QComboBox()
        for label, _ in self.STAGES:
            self._stage_combo.addItem(label)
        self._stage_combo.setCurrentIndex(0)
        header.addWidget(self._stage_combo)
        header.addStretch()
        layout.addLayout(header)

        # Image display
        self._img_label = _ImageLabel()
        layout.addWidget(self._img_label, stretch=1)

        return panel

    # ── Signal connections ────────────────────────────────────────────────────

    def _connect_signals(self):
        self._load_btn.clicked.connect(self._on_load)
        self._run_btn.clicked.connect(self._on_run)
        self._save_btn.clicked.connect(self._on_save)
        self._stage_combo.currentIndexChanged.connect(self._on_stage_changed)

        self.controller.result_ready.connect(self._on_result)

    # ── Slots ─────────────────────────────────────────────────────────────────

    @pyqtSlot()
    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if path:
            self.load_image(path)

    def load_image(self, path: str):
        """Public: called by MainWindow's menu action too."""
        try:
            self._current_image = load_image(path)
        except Exception as exc:
            self.controller.error_occurred.emit(str(exc))
            return

        self._img_name_lbl.setText(Path(path).name)
        self._results = None
        self._stage_combo.setCurrentIndex(0)           # show original
        self._show_array(self._current_image, "original")
        self._run_btn.setEnabled(True)
        self._save_btn.setEnabled(False)
        self._stats_lbl.setText(
            f"Size: {self._current_image.shape[1]} × {self._current_image.shape[0]}\n"
            f"Channels: {self._current_image.ndim}"
        )

    @pyqtSlot()
    def _on_run(self):
        if self._current_image is None:
            return
        params = {
            "kernel_size": self._kernel_spin.value(),
            "sigma":       self._sigma_spin.value(),
            "low_ratio":   self._low_spin.value(),
            "high_ratio":  self._high_spin.value(),
        }
        self._run_btn.setEnabled(False)
        self.controller.process_image_array(self._current_image, params)

    @pyqtSlot(dict)
    def _on_result(self, result: dict):
        self._results = result

        # Add grayscale / original into result for display
        if self._current_image is not None:
            self._results["original"] = self._current_image
            if self._current_image.ndim == 3:
                from core.image_manager import to_grayscale
                self._results["gray"] = to_grayscale(self._current_image)
            else:
                self._results["gray"] = self._current_image

        self._run_btn.setEnabled(True)
        self._save_btn.setEnabled(True)

        # Show current stage
        self._on_stage_changed(self._stage_combo.currentIndex())

        # Update stats
        edges = result.get("edges")
        if edges is not None:
            edge_pixels = int((edges > 0).sum())
            total_pixels = edges.size
            self._stats_lbl.setText(
                f"Size: {edges.shape[1]} × {edges.shape[0]}\n"
                f"Edge pixels: {edge_pixels:,}\n"
                f"Edge density: {edge_pixels/total_pixels*100:.2f}%\n"
                f"Low threshold:  {result['low_thresh']:.1f}\n"
                f"High threshold: {result['high_thresh']:.1f}"
            )

    @pyqtSlot(int)
    def _on_stage_changed(self, idx: int):
        if self._results is None and idx == 0 and self._current_image is not None:
            self._show_array(self._current_image, "original")
            return
        if self._results is None:
            return
        _, key = self.STAGES[idx]
        arr = self._results.get(key)
        if arr is not None:
            self._show_array(arr, key)

    @pyqtSlot()
    def _on_save(self):
        if self._results is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Edge Map", "edge_map.png",
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)"
        )
        if path:
            save_image(self._results["edges"], path)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _show_array(self, arr: np.ndarray, _key: str):
        pix = to_qpixmap(arr)
        self._img_label.set_pixmap(pix)