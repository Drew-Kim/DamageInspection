"""
Raspberry Pi Box Damage Inspection
179M AI Senior Design Project
This is PyQt5-based GUI application that captures video from the Raspberry Pi AI Camera (IMX-500 format)
and displays the live feed with detected bounding boxes. The application allows the user to take scans of the current frame.
"""

import sys
from datetime import datetime
from typing import List, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


DemoRow = Tuple[str, float, Tuple[int, int, int, int]]


def centered_item(value: str) -> QTableWidgetItem:
    item = QTableWidgetItem(value)
    item.setTextAlignment(Qt.AlignCenter)
    return item


class CameraAppV2(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Damaged Box Inspection - Version 2")
        self.resize(1200, 760)

        self.last_demo_rows: List[DemoRow] = [
            ("dent", 0.91, (120, 90, 280, 210)),
            ("tear", 0.83, (360, 140, 520, 300)),
        ]

        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        layout.addLayout(left_col, 4)
        layout.addLayout(right_col, 2)

        self.video_label = QLabel("Video feed will be added in Version 3")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background:#111; color:#ddd; border:1px solid #444; font-size:16px;"
        )
        self.video_label.setMinimumSize(860, 520)
        left_col.addWidget(self.video_label)

        button_row = QHBoxLayout()
        self.btn_take = QPushButton("Take Scan")
        self.btn_retake = QPushButton("Retake")
        self.btn_take.clicked.connect(self.take_scan)
        self.btn_retake.clicked.connect(self.retake)
        button_row.addWidget(self.btn_take)
        button_row.addWidget(self.btn_retake)
        left_col.addLayout(button_row)

        self.status_label = QLabel("Status: Ready")
        self.scan_time_label = QLabel("Last Scan: --")
        left_col.addWidget(self.status_label)
        left_col.addWidget(self.scan_time_label)

        right_col.addWidget(QLabel("Detection Results"))

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Class", "Confidence", "Location (TL,BR)"])
        self.table.setColumnWidth(0, 160)
        self.table.setColumnWidth(1, 100)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(520)
        right_col.addWidget(self.table)

        action_row = QHBoxLayout()
        self.btn_confirm = QPushButton("Confirm")
        self.btn_database = QPushButton("Database")
        self.btn_confirm.setEnabled(False)
        self.btn_database.setEnabled(False)
        action_row.addWidget(self.btn_confirm)
        action_row.addWidget(self.btn_database)
        right_col.addLayout(action_row)

    def _populate_demo_rows(self, rows: List[DemoRow]) -> None:
        self.table.setRowCount(len(rows))
        for row_i, row in enumerate(rows):
            class_name, confidence, bbox = row
            self.table.setItem(row_i, 0, centered_item(class_name))
            self.table.setItem(row_i, 1, centered_item(f"{confidence:.3f}"))
            self.table.setItem(row_i, 2, centered_item(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"))

    def take_scan(self) -> None:
        self.status_label.setText("Status: Demo snapshot captured")
        self.scan_time_label.setText(f"Last Scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._populate_demo_rows(self.last_demo_rows)

    def retake(self) -> None:
        self.status_label.setText("Status: Ready")
        self.table.setRowCount(0)


def main() -> int:
    app = QApplication(sys.argv)
    window = CameraAppV2()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
