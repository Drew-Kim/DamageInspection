#!/usr/bin/env python3
"""
Raspberry Pi Box Damage Inspection
179M AI Senior Design Project
This is PyQt5-based GUI application that captures video from the Raspberry Pi AI Camera (IMX-500 format) 
"""

import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)


class CameraAppV1(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Damaged Box Inspection - V1")
        self.resize(1200, 760)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        layout.addLayout(left_col, 4)
        layout.addLayout(right_col, 2)

        # Placeholder for future video feed.
        self.video_label = QLabel("Video feed will be added in next versions")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background:#111; color:#ddd; border:1px solid #444; font-size:16px;"
        )
        self.video_label.setMinimumSize(860, 520)
        left_col.addWidget(self.video_label)

        # Buttons exist, but are not functional yet.
        button_row = QHBoxLayout()
        self.btn_take = QPushButton("Take Scan")
        self.btn_retake = QPushButton("Retake")
        button_row.addWidget(self.btn_take)
        button_row.addWidget(self.btn_retake)
        left_col.addLayout(button_row)

        self.status_label = QLabel("Status: Format only")
        self.scan_time_label = QLabel("Last Scan: --")
        left_col.addWidget(self.status_label)
        left_col.addWidget(self.scan_time_label)

        right_col.addWidget(QLabel("Detection Results"))

        # Empty table for now.
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Class", "Confidence", "Location (TL,BR)"])
        self.table.setMinimumHeight(520)
        right_col.addWidget(self.table)

        action_row = QHBoxLayout()
        self.btn_confirm = QPushButton("Confirm")
        self.btn_database = QPushButton("Database")
        action_row.addWidget(self.btn_confirm)
        action_row.addWidget(self.btn_database)
        right_col.addLayout(action_row)


def main() -> int:
    app = QApplication(sys.argv)
    window = CameraAppV1()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
