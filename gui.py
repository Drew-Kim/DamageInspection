"""
Raspberry Pi Box Damage Inspection
179M AI Senior Design Project
This is PyQt5-based GUI application that captures video from the Raspberry Pi AI Camera (IMX-500 format)
and displays the live feed with detected bounding boxes. The application allows the user to take scans of the current frame.
"""

import argparse
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

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
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection
from picamera2.devices.imx500.postprocess import scale_boxes


MAX_DETECTIONS = 10
NANODET_IOU = 0.65
LIVE_STATUS = "Status: Live Feed"


@dataclass
class DetectionRow:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]


def centered_item(value: str) -> QTableWidgetItem:
    item = QTableWidgetItem(value)
    item.setTextAlignment(Qt.AlignCenter)
    return item


class LocalDatabaseV6:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._create_tables()

    def _create_tables(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_time TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    x1 INTEGER NOT NULL,
                    y1 INTEGER NOT NULL,
                    x2 INTEGER NOT NULL,
                    y2 INTEGER NOT NULL
                )
                """
            )

    def insert_entries(self, scan_time: str, rows: List[DetectionRow]) -> int:
        if not rows:
            return 0

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO entries (scan_time, class_name, confidence, x1, y1, x2, y2)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        scan_time,
                        row.class_name,
                        float(row.confidence),
                        int(row.bbox[0]),
                        int(row.bbox[1]),
                        int(row.bbox[2]),
                        int(row.bbox[3]),
                    )
                    for row in rows
                ],
            )
        return len(rows)


class CameraAppV6(QMainWindow):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args

        self.setWindowTitle("Damaged Box Inspection - Version 6")
        self.resize(1360, 860)

        self.imx500 = None
        self.picam2 = None
        self.intrinsics = None
        self.labels: List[str] = []

        self.live_mode = True
        self.last_frame: Optional[np.ndarray] = None
        self.frozen_frame: Optional[np.ndarray] = None
        self.last_detections: List[DetectionRow] = []
        self.last_scan_time: Optional[str] = None

        self.previous_detections: List[DetectionRow] = []
        self.frames_since_detection = 0

        self.db = LocalDatabaseV6(self.args.db_path)

        self._build_ui()

        if not self._start_imx_camera():
            QMessageBox.critical(self, "Camera Error", "Could not start IMX500 camera.")
            raise RuntimeError("Camera start failed")

        self.status_label.setText(LIVE_STATUS)

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

        self.video_label = QLabel("Waiting for camera frames...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background:#111; color:#ddd; border:1px solid #444; font-size:16px;"
        )
        self.video_label.setMinimumSize(900, 560)
        left_col.addWidget(self.video_label)

        controls = QHBoxLayout()
        self.btn_take = QPushButton("Take Scan")
        self.btn_retake = QPushButton("Retake")
        self.btn_take.clicked.connect(self.take_snapshot)
        self.btn_retake.clicked.connect(self.retake)
        controls.addWidget(self.btn_take)
        controls.addWidget(self.btn_retake)
        left_col.addLayout(controls)

        self.status_label = QLabel(LIVE_STATUS)
        self.scan_time_label = QLabel("Last Scan: --")
        left_col.addWidget(self.status_label)
        left_col.addWidget(self.scan_time_label)

        info = QLabel("Detection Results")
        info.setStyleSheet("font-size:16px; font-weight:600;")
        right_col.addWidget(info)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Class", "Confidence", "Location (TL,BR)"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setColumnWidth(0, 150)
        self.table.setColumnWidth(1, 90)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(520)
        right_col.addWidget(self.table)

        action_row = QHBoxLayout()
        self.btn_confirm = QPushButton("Confirm")
        self.btn_database = QPushButton("Database")
        self.btn_confirm.clicked.connect(self.confirm_scan)
        self.btn_database.setEnabled(False)
        action_row.addWidget(self.btn_confirm)
        action_row.addWidget(self.btn_database)
        right_col.addLayout(action_row)

    def _start_imx_camera(self) -> bool:
        if not os.path.isfile(self.args.model):
            return False

        try:
            self.imx500 = IMX500(self.args.model)
            self.intrinsics = self.imx500.network_intrinsics

            if self.intrinsics is None:
                self.intrinsics = NetworkIntrinsics()
                self.intrinsics.task = "object detection"
            elif self.intrinsics.task != "object detection":
                return False

            if os.path.isfile(self.args.labels):
                with open(self.args.labels, "r", encoding="utf-8") as f:
                    self.intrinsics.labels = f.read().splitlines()

            self.intrinsics.ignore_dash_labels = True
            self.intrinsics.update_with_defaults()
            self.labels = [label for label in (self.intrinsics.labels or []) if label and label != "-"]

            self.picam2 = Picamera2(self.imx500.camera_num)
            config = self.picam2.create_preview_configuration(
                main={"size": (self.args.width, self.args.height), "format": "RGB888"},
                buffer_count=12,
            )
            self.imx500.show_network_fw_progress_bar()
            self.picam2.start(config, show_preview=False)

            if self.intrinsics.preserve_aspect_ratio:
                self.imx500.set_auto_aspect_ratio()

            return True
        except Exception:
            self.imx500 = None
            self.intrinsics = None
            self.picam2 = None
            return False

    def _read_frame_and_detections(self) -> Tuple[Optional[np.ndarray], List[DetectionRow]]:
        request = self.picam2.capture_request()
        if request is None:
            return None, []

        try:
            frame = request.make_array("main")
            metadata = request.get_metadata()
        finally:
            request.release()

        if frame is None:
            return None, []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self._parse_imx_detections(metadata) if metadata else []
        return frame_rgb, detections

    def _parse_imx_detections(self, metadata: dict) -> List[DetectionRow]:
        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            return []

        input_w, input_h = self.imx500.get_input_size()

        if self.intrinsics.postprocess == "nanodet":
            boxes, scores, classes = postprocess_nanodet_detection(
                outputs=np_outputs[0],
                conf=self.args.threshold,
                iou_thres=NANODET_IOU,
                max_out_dets=MAX_DETECTIONS,
            )[0]
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
            return self._build_rows(boxes, scores, classes, metadata)

        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]

        candidates: List[List[DetectionRow]] = []
        for use_norm in (bool(self.intrinsics.bbox_normalization), not bool(self.intrinsics.bbox_normalization)):
            test_boxes = boxes / input_h if use_norm else boxes
            candidates.append(self._build_rows(test_boxes[:, [1, 0, 3, 2]], scores, classes, metadata))
            candidates.append(self._build_rows(test_boxes, scores, classes, metadata))

        return max(candidates, key=len, default=[])

    def _build_rows(self, boxes, scores, classes, metadata: dict) -> List[DetectionRow]:
        rows: List[DetectionRow] = []
        for box, score, category in zip(boxes, scores, classes):
            confidence = float(score)
            if confidence < self.args.threshold:
                continue

            x, y, w, h = self.imx500.convert_inference_coords(box, metadata, self.picam2)
            x, y, w, h = int(x), int(y), int(w), int(h)
            if w <= 0 or h <= 0:
                continue

            class_idx = int(category)
            class_name = self.labels[class_idx] if 0 <= class_idx < len(self.labels) else f"class_{class_idx}"
            rows.append(DetectionRow(class_name, confidence, (x, y, x + w, y + h)))
            if len(rows) >= MAX_DETECTIONS:
                break

        return rows

    def _stabilize(self, rows: List[DetectionRow]) -> List[DetectionRow]:
        if rows:
            self.previous_detections = rows
            self.frames_since_detection = 0
            return rows

        if self.previous_detections and self.frames_since_detection < self.args.hold_frames:
            self.frames_since_detection += 1
            return self.previous_detections

        self.previous_detections = []
        return []

    def update_frame(self) -> None:
        if not self.live_mode:
            if self.frozen_frame is not None:
                self._show_frame(self.frozen_frame)
            return

        frame, detections = self._read_frame_and_detections()
        if frame is None:
            self.status_label.setText("Status: Camera read failed")
            return

        stable = self._stabilize(detections)
        frame_draw = frame.copy()
        self._draw_overlay(frame_draw, stable)

        self.last_frame = frame_draw
        self.last_detections = stable
        self.status_label.setText(f"{LIVE_STATUS} | Detections: {len(stable)}")
        self._show_frame(frame_draw)

    def _draw_overlay(self, frame_rgb: np.ndarray, rows: List[DetectionRow]) -> None:
        for row in rows:
            x1, y1, x2, y2 = row.bbox
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{row.class_name} ({row.confidence:.2f})"
            cv2.putText(
                frame_rgb,
                label,
                (x1 + 5, max(y1 - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

    def _show_frame(self, frame_rgb: np.ndarray) -> None:
        h, w, c = frame_rgb.shape
        image = QImage(frame_rgb.data, w, h, c * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    def take_snapshot(self) -> None:
        if self.last_frame is None:
            self.status_label.setText("Status: No feed yet")
            return

        self.frozen_frame = self.last_frame.copy()
        self.live_mode = False
        self.last_scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.scan_time_label.setText(f"Last Scan: {self.last_scan_time}")
        self.status_label.setText(f"Status: Scan captured ({len(self.last_detections)} detections)")
        self._populate_table(self.last_detections)

    def confirm_scan(self) -> None:
        if self.last_scan_time is None:
            QMessageBox.warning(self, "Confirm", "Take a scan first before confirming.")
            return

        saved_count = self.db.insert_entries(self.last_scan_time, self.last_detections)
        self.status_label.setText(
            f"Status: Saved {saved_count} entr{'y' if saved_count == 1 else 'ies'}"
        )

    def retake(self) -> None:
        self.live_mode = True
        self.frozen_frame = None
        self.status_label.setText(LIVE_STATUS)
        self._populate_table([])

    def _populate_table(self, rows: List[DetectionRow]) -> None:
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self.table.setItem(i, 0, centered_item(row.class_name))
            self.table.setItem(i, 1, centered_item(f"{row.confidence:.3f}"))
            self.table.setItem(i, 2, centered_item(f"{row.bbox[0]},{row.bbox[1]},{row.bbox[2]},{row.bbox[3]}"))

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
    parser = argparse.ArgumentParser(description="GUI Version 6")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--model", type=str, default="/home/pi/network.rpk")
    parser.add_argument("--labels", type=str, default="/home/pi/labels.txt")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--hold-frames", type=int, default=5)
    parser.add_argument("--db-path", type=str, default="/home/pi/DamageInspection/inspection.db")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    window = CameraAppV6(args)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
