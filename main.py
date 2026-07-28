#!/usr/bin/env python3
"""BRAHMS — Three-Wave Mixing Simulator entry point."""

import sys
import os
from pathlib import Path

# Ensure high-DPI scaling
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")


def main():
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QFont, QFontDatabase, QIcon
    from PyQt6.QtCore import Qt, QLocale

    app = QApplication(sys.argv)

    # Force C locale so QDoubleSpinBox always uses '.' as decimal separator
    # regardless of the system locale (which may use ',' on many Linux setups).
    QLocale.setDefault(QLocale(QLocale.Language.C))
    app.setApplicationName("BRAHMS")
    app.setApplicationDisplayName("")
    app.setOrganizationName("NLO Lab")

    # ── Font ────────────────────────────────────────────────────────────
    # Try Libre Baskerville (installed via setup_linux.sh),
    # fall back to Georgia or any available serif.
    font = QFont("Libre Baskerville")
    if not font.exactMatch():
        for fallback in ["EB Garamond", "Georgia", "DejaVu Serif", "serif"]:
            font = QFont(fallback)
            if font.exactMatch():
                break
    font.setPointSize(10)
    app.setFont(font)

    # ── Stylesheet ──────────────────────────────────────────────────────
    assets = Path(__file__).parent / "gui" / "assets"
    qss    = assets / "style.qss"
    if qss.exists():
        app.setStyleSheet(qss.read_text())

    # ── Window icon ─────────────────────────────────────────────────────
    # On Wayland/GNOME the taskbar icon comes from the .desktop file.
    # setDesktopFileName links this process to brahms.desktop so the
    # compositor can match window → app-id → icon.
    app.setDesktopFileName("brahms")

    root = Path(__file__).parent
    icon_candidates = [
        root / "icon" / "icon.png",
        assets / "icon.svg",
        assets / "icon.png",
    ]
    _app_icon = QIcon()
    for icon_path in icon_candidates:
        if icon_path.exists():
            _app_icon = QIcon(str(icon_path))
            break
    app.setWindowIcon(_app_icon)

    # ── Main window ─────────────────────────────────────────────────────
    from gui.mainwindow import MainWindow
    win = MainWindow()
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
