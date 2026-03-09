"""
Raspberry Pi Box Damage Inspection
179M AI Senior Design Project
This is PyQt5-based GUI application that captures video from the Raspberry Pi AI Camera (IMX-500 format)
and displays the live feed with detected bounding boxes. The application allows the user to take scans of the current frame.
"""


import argparse
import os
import sys
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from picamera2 import Picamera2
from picamera2.devices import IMX500


class CameraAppV4(QMainWindow):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args

        self.setWindowTitle("Damaged Box Inspection - Version 4")
        self.resize(1200, 760)

        self.imx500 = None
        self.picam2 = None

        self.live_mode = True
        self.last_frame: Optional[np.ndarray] = None
        self.frozen_frame: Optional[np.ndarray] = None

        self._build_ui()

        if not self._start_camera():
            QMessageBox.critical(self, "Camera Error", "Could not start IMX500 camera.")
            raise RuntimeError("Camera start failed")

        self.status_label.setText("Status: Live feed running")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / max(1, self.args.fps)))

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        layout.addLayout(left_col, 4)
        layout.addLayout(right_col, 2)

        self.video_label = QLabel("Waiting for camera...")
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

        self.status_label = QLabel("Status: Starting camera")
        self.scan_time_label = QLabel("Last Scan: --")
        left_col.addWidget(self.status_label)
        left_col.addWidget(self.scan_time_label)

        right_col.addWidget(QLabel("Detection Results"))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Class", "Confidence", "Location (TL,BR)"])
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

    def _start_camera(self) -> bool:
        if not os.path.isfile(self.args.model):
            return False

        try:
            self.imx500 = IMX500(self.args.model)
            self.picam2 = Picamera2(self.imx500.camera_num)
            config = self.picam2.create_preview_configuration(
                main={"size": (self.args.width, self.args.height), "format": "RGB888"}
            )
            self.picam2.start(config, show_preview=False)
            return True
        except Exception:
            self.imx500 = None
            self.picam2 = None
            return False

    def update_frame(self) -> None:
        if not self.live_mode:
            if self.frozen_frame is not None:
                self._show_frame(self.frozen_frame)
            return

        request = self.picam2.capture_request()
        if request is None:
            self.status_label.setText("Status: Camera read failed")
            return

        try:
            frame = request.make_array("main")
        finally:
            request.release()

        if frame is None:
            self.status_label.setText("Status: Camera read failed")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.last_frame = frame_rgb
        self._show_frame(frame_rgb)

    def _show_frame(self, frame_rgb: np.ndarray) -> None:
        h, w, c = frame_rgb.shape
        image = QImage(frame_rgb.data, w, h, c * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    def take_scan(self) -> None:
        if self.last_frame is None:
            self.status_label.setText("Status: No frame yet")
            return

        self.frozen_frame = self.last_frame.copy()
        self.live_mode = False
        self.scan_time_label.setText(f"Last Scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.status_label.setText("Status: Scan captured")

        # Placeholder table row for now (real detections come in Version 5).
        self.table.setRowCount(1)
        self.table.setItem(0, 0, QTableWidgetItem("pending"))
        self.table.setItem(0, 1, QTableWidgetItem("--"))
        self.table.setItem(0, 2, QTableWidgetItem("--"))

    def retake(self) -> None:
        self.live_mode = True
        self.frozen_frame = None
        self.status_label.setText("Status: Live feed running")
        self.table.setRowCount(0)

    def closeEvent(self, event) -> None:
        self.timer.stop()
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
        super().closeEvent(event)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI Version 4")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--model", type=str, default="/home/pi/network.rpk")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    window = CameraAppV4(args)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
