"""
Raspberry Pi Box Damage Inspection
179M AI Senior Design Project
This is PyQt5-based GUI application that captures video from the Raspberry Pi AI Camera (IMX-500 format) 
and displays the live feed with detected bounding boxes. The application allows the user to take scans of the current frame. 
The user can confirm and save detections to a local SQLite database, and also view and delete past entries in the database.
""
Dependencies:
- PyQt5 for the GUI
- OpenCV for drawing and color conversion
- NumPy for array handling
- Picamera2 and IMX500 API for camera access
"""
import argparse
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
# OpenCV for drawing boxes and color conversion
import cv2
# NumPy for model outputs
import numpy as np
# PyQt5 for GUI components
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import ( QAbstractItemView, QApplication, QDialog, QHeaderView, QHBoxLayout, QLabel, 
    QMainWindow, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)
# Picamera2 and IMX500 API for camera access
from picamera2 import Picamera2
from picamera2.devices import IMX500 
from picamera2.devices.imx500 import ( NetworkIntrinsics, postprocess_nanodet_detection )
from picamera2.devices.imx500.postprocess import scale_boxes
# Data structure for one detection row, used for both GUI table and database storage.
@dataclass
class DetectionRow:
    class_name: str
    confidence: float
    # Bounding box is stored as top-left and bottom-right pixel corners: (x1, y1, x2, y2)
    bbox: Tuple[int, int, int, int]

# Global Constants
MAX_DETECTIONS = 10 # Limit to at most 10 boxes shown per frame 
NANODET_IOU = 0.65 # IOU threshold for NanoDet postprocess
LIVE_STATUS = "Status: Live Feed"

# Helper Function: Center texts for table display
def centered_item(value: str) -> QTableWidgetItem:
    item = QTableWidgetItem(value)
    item.setTextAlignment(Qt.AlignCenter)
    return item

# LocalDatabase class using SQLite for storing detections 
class LocalDatabase:
    # If one scan has 3 detections, we store 3 rows with same scan_time
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._create_tables()

    def _create_tables(self):
        # Create the table. If it already exists, this does nothing
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_time TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    x1 INTEGER NOT NULL, y1 INTEGER NOT NULL,
                    x2 INTEGER NOT NULL, y2 INTEGER NOT NULL )
                """
            )
    # Insert multiple rows for one scan. Returns how many rows were inserted.
    def insert_entries(self, scan_time: str, rows: List[DetectionRow]):
        # Nothing to insert if scan had no detections
        if not rows:
            return 0

        # Insert rows in one call with the same scan_time and the id primary key auto-increments 
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO entries (scan_time, class_name, confidence, x1, y1, x2, y2)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (scan_time, row.class_name, float(row.confidence), int(row.bbox[0]),
                        int(row.bbox[1]), int(row.bbox[2]), int(row.bbox[3]) )
                    for row in rows
                ],
            )
        return len(rows)
    # Newest rows first so recent scans show on top
    def fetch_entries(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, scan_time, class_name, confidence, x1, y1, x2, y2
                FROM entries
                ORDER BY id DESC
                """
            )
            return cursor.fetchall()
    # Delete exactly one row by primary key.
    def delete_entry(self, entry_id: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
# Window to show database entries in a table and allows deletion
class DatabaseOverviewDialog(QDialog):
    def __init__(self, db: LocalDatabase, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("Database Overview")
        self.resize(1000, 600) # Window size
        # Main Vertical Layout
        layout = QVBoxLayout(self)
        # Create a table with 5 columns: ID, Scan Time, Class, Confidence, Location
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Entry ID", "Scan Time", "Class", "Confidence", "Location (TL,BR)"] )
        # Prevent typing in to the table 
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # Select one full row at a time for deletion
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        # Configures how columns resize
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)

        layout.addWidget(self.table)
        # Create buttons for deleting selected entry
        buttons = QHBoxLayout()
        self.btn_delete = QPushButton("Delete Selected")
        self.btn_delete.clicked.connect(self.delete_selected)
        buttons.addWidget(self.btn_delete)
        layout.addLayout(buttons)

        # Load all rows and display them in the table
        rows = self.db.fetch_entries()
        self.table.setRowCount(len(rows))
        for row_i, row in enumerate(rows):
            entry_id, scan_time, class_name, confidence, x1, y1, x2, y2 = row
            values = [
                str(entry_id),
                scan_time,
                class_name,
                f"{confidence:.3f}",
                f"{x1},{y1},{x2},{y2}",
            ]
            for col_i, value in enumerate(values):
                self.table.setItem(row_i, col_i, centered_item(value))
    # Delete the currently selected row after confirming
    def delete_selected(self):
        row = self.table.currentRow()
        if row < 0: # No row selected
            QMessageBox.information(self, "Delete", "Select a row first.")
            return

        # Column 0 stores the Entry ID
        entry_id = int(self.table.item(row, 0).text())
        answer = QMessageBox.question(
            self,
            "Delete Entry",
            f"Delete entry ID {entry_id}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        # Delete from database and remove from table
        self.db.delete_entry(entry_id)
        self.table.removeRow(row)

# Main GUI Camera Application Window
# Left side: video feed, scan controls, status labels
# Right side: detection table, confirm button, database overview button
class CameraApp(QMainWindow):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        # Main Windows settings
        self.setWindowTitle("Damaged Box Inspection - GUI")
        self.resize(1360, 860)
        # Camera/model objects from Picamera2 + IMX500 APIs. Initialized in _start_imx_camera()
        self.picam2 = None
        self.imx500 = None
        self.intrinsics = None
        self.labels: List[str] = []

        # GUI state variables 
        self.live_mode = True
        self.last_frame: Optional[np.ndarray] = None # Last frame with detections drawn
        self.frozen_frame: Optional[np.ndarray] = None # Saved snapshot frame when not in live mode
        self.last_detections: List[DetectionRow] = [] # Most recent detection
        self.last_scan_time: Optional[str] = None # Time of the last scan

        # Anti-flicker stabilization for detections: if one frame has detections but the next few frames don't
        self.previous_detections: List[DetectionRow] = []
        self.frames_since_detection = 0
        # Database for storing confirmed scans using SQLite
        self.db = LocalDatabase(self.args.db_path)
        # Build the GUI layout and widgets
        self._build_ui()
        # Start the camera and model pipeline
        if not self._start_imx_camera(): # Check if camera started successfully
            QMessageBox.critical(
                self,
                "Camera Error",
                "Could not start IMX500 camera. Check model path, labels path, and camera connection.",
            )
            raise RuntimeError("Camera start failed")

        # Timer drives the live update loop.
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / max(1, self.args.fps)))
    # Build the GUI layout and connect buttons to their functions.
    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        # Main window split into left and right columns
        layout = QHBoxLayout(root)
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        left_col.setSpacing(4)
        left_col.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(left_col, 4) # left side more space
        layout.addLayout(right_col, 2)
        # Left side
        # Video display area
        self.video_label = QLabel("Waiting for camera frames...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background:#111; color:#ddd; border:1px solid #444; font-size:16px;"
        )
        self.video_label.setMinimumSize(900, 560)
        left_col.addWidget(self.video_label)
        # Buttons below the video feed
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        self.btn_take = QPushButton("Take Scan")
        self.btn_retake = QPushButton("Retake")
        # Remove keyboard focus border outline for a cleaner look 
        self.btn_take.setFocusPolicy(Qt.NoFocus)
        self.btn_retake.setFocusPolicy(Qt.NoFocus)
        # Connect buttons to their functions
        self.btn_take.clicked.connect(self.take_snapshot)
        self.btn_retake.clicked.connect(self.retake)
        controls.addWidget(self.btn_take)
        controls.addWidget(self.btn_retake)
        left_col.addLayout(controls)
        # Status label and last scan time below the buttons
        self.status_label = QLabel(LIVE_STATUS)
        self.status_label.setMaximumHeight(22)
        left_col.addWidget(self.status_label)

        self.scan_time_label = QLabel("Last Scan: --")
        self.scan_time_label.setMaximumHeight(22)
        left_col.addWidget(self.scan_time_label)
        # Right side
        info = QLabel("Detection Results")
        info.setStyleSheet("font-size:16px; font-weight:600;")
        right_col.addWidget(info)
        # Small table that shows detections from the last captured scan.
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Class", "Confidence", "Location (TL,BR)"])
        # Prevent typing in to the table
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setColumnWidth(0, 150)
        self.table.setColumnWidth(1, 90)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(520)
        right_col.addWidget(self.table)
        # Database action buttons
        bottom_buttons = QHBoxLayout()
        self.btn_confirm = QPushButton("Confirm")
        self.btn_database = QPushButton("Database")
        self.btn_confirm.setFocusPolicy(Qt.NoFocus)
        self.btn_database.setFocusPolicy(Qt.NoFocus)
        self.btn_confirm.clicked.connect(self.confirm_scan)
        self.btn_database.clicked.connect(self.open_database_overview)
        bottom_buttons.addWidget(self.btn_confirm)
        bottom_buttons.addWidget(self.btn_database)
        right_col.addLayout(bottom_buttons)
    # Change the Take Scan button text and color
    def _set_take_button_state(self, captured: bool):
        if captured:
            self.btn_take.setText("Scan Captured")
            self.btn_take.setStyleSheet("color: green;")
        else:
            self.btn_take.setText("Take Scan")
            self.btn_take.setStyleSheet("")
    # Start the IMX500 camera pipeline and load the model
    def _start_imx_camera(self) -> bool:
        if not os.path.isfile(self.args.model):
            return False
        try:
            # Load model package (.rpk) and read model metadata
            self.imx500 = IMX500(self.args.model)
            self.intrinsics = self.imx500.network_intrinsics

            if self.intrinsics is None:
                self.intrinsics = NetworkIntrinsics()
                self.intrinsics.task = "object detection"
            elif self.intrinsics.task != "object detection":
                return False

            # Load labels file 
            if os.path.isfile(self.args.labels):
                with open(self.args.labels, "r", encoding="utf-8") as f:
                    self.intrinsics.labels = f.read().splitlines()

            self.intrinsics.ignore_dash_labels = True
            self.intrinsics.update_with_defaults()

            # Drop empty labels and "-" placeholder labels.
            self.labels = [label for label in (self.intrinsics.labels or []) if label and label != "-"]

            self.picam2 = Picamera2(self.imx500.camera_num)
            controls = {}
            if getattr(self.intrinsics, "inference_rate", None):
                controls["FrameRate"] = self.intrinsics.inference_rate

            config_kwargs = {
                "main": {"size": (self.args.width, self.args.height), "format": "RGB888"},
                "buffer_count": 12,
            }
            if controls:
                config_kwargs["controls"] = controls
            config = self.picam2.create_preview_configuration(**config_kwargs)

            # Start camera stream.
            self.imx500.show_network_fw_progress_bar()
            self.picam2.start(config, show_preview=False)

            if self.intrinsics.preserve_aspect_ratio:
                self.imx500.set_auto_aspect_ratio()

            return True
        except Exception:
            # Is anything fails, clean up
            self.imx500 = None
            self.intrinsics = None
            self.picam2 = None
            return False
        
    # Read one camera frame and parse detections for that frame
    def _read_frame_and_detections(self):
        request = self.picam2.capture_request()
        if request is None:
            return None, []
        try:
            # Read the main image frame and metadata containing the model outputs
            frame = request.make_array("main")
            metadata = request.get_metadata()
        finally:
            request.release() # Release the request back to t

        if frame is None:
            return None, []

        # Qt displays RGB images, so convert camera frame for correct display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self._parse_imx_detections(metadata) if metadata else []
        return frame_rgb, detections
    # Parse the model outputs from the metadata and convert
    def _parse_imx_detections(self, metadata: dict) -> List[DetectionRow]:
        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            return []
        # Size expected by the model 
        input_w, input_h = self.imx500.get_input_size()

        # Special handling for NanoDet models
        if self.intrinsics.postprocess == "nanodet":
            boxes, scores, classes = postprocess_nanodet_detection(
                outputs=np_outputs[0],
                conf=self.args.threshold,
                iou_thres=NANODET_IOU,
                max_out_dets=MAX_DETECTIONS,
            )[0]
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
            return self._build_rows(boxes, scores, classes, metadata)

        # Format: boxes, scores, classes
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]

        # Try a few common formats
        candidates: List[List[DetectionRow]] = []
        for use_norm in (bool(self.intrinsics.bbox_normalization), not bool(self.intrinsics.bbox_normalization)):
            test_boxes = boxes / input_h if use_norm else boxes
            candidates.append(self._build_rows(test_boxes[:, [1, 0, 3, 2]], scores, classes, metadata))
            candidates.append(self._build_rows(test_boxes, scores, classes, metadata))
        # Choose the candidate with the most valid detections, as long its not empty
        return max(candidates, key=len, default=[])

    # Convert raw model outputs into DetectionRow items, applying confidence threshold and coordinate conversion.
    def _build_rows(self, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, metadata: dict ):
        rows: List[DetectionRow] = []
        for box, score, category in zip(boxes, scores, classes):
            confidence = float(score)
            # Skip low-confidence detections to reduce clutter
            if confidence < self.args.threshold:
                continue
            # Convert model-space box values into preview pixel coordinates.
            x, y, w, h = self.imx500.convert_inference_coords(box, metadata, self.picam2)
            x, y, w, h = int(x), int(y), int(w), int(h)
            # Invalid box check
            if w <= 0 or h <= 0:
                continue
            # Map class index to class name
            class_idx = int(category)
            class_name = self.labels[class_idx] if 0 <= class_idx < len(self.labels) else f"class_{class_idx}"
            # Store as top left and bottom right corner coordinates
            rows.append(DetectionRow(class_name, confidence, (x, y, x + w, y + h)))
            # Stop if max detections reached to avoid overwhelming the display
            if len(rows) >= MAX_DETECTIONS:
                break
        return rows

    # Anti flicker stabilization helper function
    def _stabilize(self, rows: List[DetectionRow]):
        if rows:
            self.previous_detections = rows
            self.frames_since_detection = 0
            return rows
        # If no detections now, but some recently, keep showing those frames
        if self.previous_detections and self.frames_since_detection < self.args.hold_frames:
            self.frames_since_detection += 1
            return self.previous_detections
        # Too long without detections, clear history
        self.previous_detections = []
        return []

    # Main timer loop
    def update_frame(self):
        if not self.live_mode:
            # Freeze the frame
            if self.frozen_frame is not None:
                self._show_frame(self.frozen_frame)
            return
        # Read current frame and detections from the camera
        frame, detections = self._read_frame_and_detections()
        if frame is None:
            self.status_label.setText("Status: Camera read failed")
            return
        # Apply anti-flicker stabilization to the detections and draw them on the frame
        stable = self._stabilize(detections)
        frame_draw = frame.copy()
        self._draw_overlay(frame_draw, stable)
        # Save the most recent frame with detections drawn
        self.last_frame = frame_draw
        self.last_detections = stable
        # Update status and display
        self.status_label.setText(f"{LIVE_STATUS} | Detections: {len(stable)}")
        self._show_frame(frame_draw)

    # Draw bounding boxes and labels on the frame for each detection
    def _draw_overlay(self, frame_rgb: np.ndarray, rows: List[DetectionRow]):
        for row in rows:
            x1, y1, x2, y2 = row.bbox
            # Draw green box and label with class name and confidence score
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

    # Convert the RGB frame with OpenCV boxes into a Qt image and display it in the video label.
    def _show_frame(self, frame_rgb: np.ndarray):
        h, w, c = frame_rgb.shape
        # Create QImage from raw frame data
        image = QImage(frame_rgb.data, w, h, c * w, QImage.Format_RGB888)
        # Convert QImage to QPixmap for display
        pixmap = QPixmap.fromImage(image)
        # Scale image to fit video label while keeping aspect ratio
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    # Take a snapshot of the current frame and detections and show detections in the table
    def take_snapshot(self):
        if self.last_frame is None:
            self.status_label.setText("Status: No feed yet")
            return
        # Save the current frame and detections
        self.frozen_frame = self.last_frame.copy()
        # Stop live feed
        self.live_mode = False
        # Update the Take Scan button to indicate frame is captured
        self._set_take_button_state(captured=True)
        # Save timestamp
        self.last_scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.scan_time_label.setText(f"Last Scan: {self.last_scan_time}")
        # Show detection info and populate the table with the most recent detections from the scan
        self.status_label.setText(f"Status: Scan Captured ({len(self.last_detections)} detections)")
        self._populate_table(self.last_detections)

    # Save the current scan's detections to the local database
    def confirm_scan(self):
        if self.last_scan_time is None:
            QMessageBox.warning(self, "Confirm", "Take a scan first before confirming.")
            return
        # Save detections
        saved_count = self.db.insert_entries(self.last_scan_time, self.last_detections)
        # Update status message
        self.status_label.setText( f"Status: Scanned {saved_count} entr{'y' if saved_count == 1 else 'ies'}" )

    # Open the database overview window to show saved entries and allow deletion
    def open_database_overview(self):
        dialog = DatabaseOverviewDialog(self.db, self)
        dialog.exec_()

    # Return back to live camera feed mode, clear the frozen frame and table info
    def retake(self):
        self.live_mode = True
        self.frozen_frame = None
        self._set_take_button_state(captured=False)
        self.status_label.setText(LIVE_STATUS)
        self._populate_table([]) # Clear the table when retaking

    # Populate the right side table
    def _populate_table(self, rows: List[DetectionRow]) -> None:
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self.table.setItem(i, 0, centered_item(row.class_name))
            self.table.setItem(i, 1, centered_item(f"{row.confidence:.3f}"))
            self.table.setItem(i, 2, centered_item(f"{row.bbox[0]},{row.bbox[1]},{row.bbox[2]},{row.bbox[3]}"))

    # Stop the camera and close the application safetly
    def _stop_camera(self):
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
            self.picam2 = None

        self.imx500 = None
        self.intrinsics = None

    # Called when the window is closing to stop timer and camera safely
    def closeEvent(self, event):
        self.timer.stop()
        self._stop_camera()
        super().closeEvent(event)

# Command line argument parsing for runtime 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI for IMX500 damaged box inspection")

    # Display settings.
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)

    # Model and detection settings.
    parser.add_argument("--model", type=str, default="/home/pi/network.rpk")
    parser.add_argument("--labels", type=str, default="/home/pi/labels.txt")
    parser.add_argument("--threshold", type=float, default=0.40)

    # Detection stabilization setting
    parser.add_argument("--hold-frames", type=int, default=5, help="How many empty frames to keep old boxes (anti-flicker)")
    # Database path setting
    parser.add_argument("--db-path", type=str, default="/home/pi/DamageInspection/inspection.db", help="SQLite database file path")
    return parser.parse_args()

# Main for the GUI application
def main() -> int:
    # PyQt startup
    args = parse_args()
    app = QApplication(sys.argv)
    window = CameraApp(args)
    window.show()
    return app.exec_()

# Run main() and exit when application is closed
if __name__ == "__main__":
    raise SystemExit(main())
