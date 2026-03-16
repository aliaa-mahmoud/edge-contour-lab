"""
MainWindow — top-level application window.

Hosts a QTabWidget with one tab per member's algorithm.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QWidget,
    QStatusBar, QAction, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui  import QIcon

from .canny_tab import CannyTab


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edge & Contour Lab — Group Project")
        self.resize(1280, 800)

        # ── Central tab widget ───────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self.setCentralWidget(self._tabs)

        # Member 1 — Canny (implemented)
        self._canny_tab = CannyTab(self)
        self._tabs.addTab(self._canny_tab, "🔍  Canny Edges")

        # Placeholder tabs for other members
        self._tabs.addTab(QWidget(), "📐  Shapes (M2)")
        self._tabs.addTab(QWidget(), "🐍  Snake (M3)")
        self._tabs.addTab(QWidget(), "📏  Contour (M4)")

        # ── Status bar ───────────────────────────────────────────────────────
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._canny_tab.controller.status_message.connect(self._status.showMessage)
        self._canny_tab.controller.error_occurred.connect(self._show_error)

        # ── Menu bar ─────────────────────────────────────────────────────────
        self._build_menu()

    # ── Menu ──────────────────────────────────────────────────────────────────

    def _build_menu(self):
        menu = self.menuBar()

        file_menu = menu.addMenu("&File")

        open_act = QAction("&Open Image…", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._open_image)
        file_menu.addAction(open_act)

        file_menu.addSeparator()

        quit_act = QAction("&Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        help_menu = menu.addMenu("&Help")
        about_act = QAction("&About", self)
        about_act.triggered.connect(self._about)
        help_menu.addAction(about_act)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if path:
            # Route to whichever tab is active
            idx = self._tabs.currentIndex()
            if idx == 0:
                self._canny_tab.load_image(path)
            else:
                self._status.showMessage(
                    "Switch to the Canny tab to open an image here.")

    def _show_error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)

    def _about(self):
        QMessageBox.about(
            self, "About",
            "<b>Edge &amp; Contour Lab</b><br>"
            "Computer-Vision Assignment 2<br><br>"
            "<b>Member 1:</b> Canny Edge Detector (from scratch)<br>"
            "<b>Member 2:</b> Shape Detection (Hough)<br>"
            "<b>Member 3:</b> Active Contour (Snake)<br>"
            "<b>Member 4:</b> Chain Code &amp; Measurements"
        )