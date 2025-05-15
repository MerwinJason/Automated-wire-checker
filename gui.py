"""
Main GUI application for the Breadboard Connection Checker.
Uses PyQt5 for the interface and integrates other modules for
camera capture, processing, detection, and validation.
"""
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox,
                             QPushButton, QLabel, QTextEdit, QSlider, QFileDialog, QComboBox,
                             QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, QSize
from typing import Optional, Tuple # Added for type hinting

import json
import os
from logger_setup import logger
try:
    from PyQt5.QtWidgets import QInputDialog
except ImportError:
    QInputDialog = None # Fallback if PyQt5 is not available (e.g. running standalone without GUI integration)


logger.info("calibrate_homography.py loaded") # This seems like a leftover log from context, but keeping as is.
from capture import Camera
from preprocess import preprocess_frame
from wire_detector import WireDetector
from terminal_locator import TerminalLocator
from connection_validator import ConnectionValidator
from utils import COLOR_RANGES, Wire # For drawing

logger.info("gui.py loaded")

class BreadboardApp(QMainWindow):
    """Main application window."""
    def __init__(self):
        super().__init__()
        logger.info("Initializing BreadboardApp GUI.")
        self.setWindowTitle("Breadboard Connection Checker")
        self.setGeometry(100, 100, 1200, 800)

        # Default values for new detector parameters
        self.default_sv_tolerance = 10
        self.default_min_contour_area = 100
        self.default_morph_kernel_size = 5
        
        self.camera = None
        self.wire_detector = WireDetector(base_color_ranges_map=COLOR_RANGES,
                                          initial_sv_tolerance=self.default_sv_tolerance,
                                          initial_min_contour_area=self.default_min_contour_area,
                                          initial_morph_kernel_size=self.default_morph_kernel_size)
        self.terminal_locator = TerminalLocator("calibration/points.json") # Load default
        self.connection_validator: Optional[ConnectionValidator] = None
        self.current_frame: Optional[np.ndarray] = None # Store the latest raw frame for capture
        self.validation_mode_active = False
        self.processed_frame_for_display: Optional[np.ndarray] = None # Store the frame with overlays

        # ROI (x, y, w, h) - This should ideally be configurable or part of calibration
        self.roi_rect: Optional[Tuple[int, int, int, int]] = None # No cropping by default

        # Timer for main frame updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # New state for delayed "all correct" message
        self.CORRECT_MESSAGE_DELAY_MS = 1500  # Delay in milliseconds (e.g., 1.5 seconds)
        self.correct_message_delay_timer = QTimer(self)
        self.correct_message_delay_timer.setSingleShot(True) # Timer fires once
        self.correct_message_delay_timer.timeout.connect(self.show_connections_correct_message_after_delay)
        
        self.is_awaiting_correct_message_display = False # True if delay timer is running
        self.connections_confirmed_correct_and_message_shown = False # True if msg shown and state is still correct

        # Latch for the overall connection status from the last validation cycle
        self.current_validation_overall_ok = False

        self.initUI()
        logger.info("GUI Initialized.")

    def initUI(self):
        """Sets up the UI elements."""
        logger.debug("Setting up UI elements.")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel: Video display
        video_layout = QVBoxLayout()
        self.video_label = QLabel("Video feed will appear here.")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 1px solid black;")
        video_layout.addWidget(self.video_label)
        main_layout.addLayout(video_layout, 2) # Video takes more space

        # Right panel: Controls and Log
        controls_layout = QVBoxLayout()

        # Camera Controls
        self.fps_slider = QSlider(Qt.Horizontal)
        self.fps_slider.setMinimum(1)
        self.fps_slider.setMaximum(60)
        self.fps_slider.setValue(15) # Default FPS
        self.fps_slider.setTickInterval(5)
        self.fps_slider.setTickPosition(QSlider.TicksBelow)
        controls_layout.addWidget(QLabel("Target FPS:"))
        controls_layout.addWidget(self.fps_slider)

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["Default", "640x480", "800x600", "1280x720"])
        controls_layout.addWidget(QLabel("Resolution Cap:"))
        controls_layout.addWidget(self.resolution_combo)

        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_capture)
        controls_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.clicked.connect(self.stop_capture)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.stop_button)

        # Validation Button
        self.validate_button = QPushButton("Validate")
        self.validate_button.clicked.connect(self.toggle_validation_mode)
        self.validate_button.setEnabled(False) # Enable after camera start
        controls_layout.addWidget(self.validate_button)

        self.load_spec_button = QPushButton("Load Wiring Spec (JSON)")
        self.load_spec_button.clicked.connect(self.load_spec)
        controls_layout.addWidget(self.load_spec_button)
        
        # HSV S/V Tolerance Slider
        self.sv_tolerance_slider = QSlider(Qt.Horizontal)
        self.sv_tolerance_slider.setMinimum(0)
        self.sv_tolerance_slider.setMaximum(100) # Max tolerance value
        self.sv_tolerance_slider.setValue(self.default_sv_tolerance)
        self.sv_tolerance_slider.setTickInterval(10)
        self.sv_tolerance_slider.setTickPosition(QSlider.TicksBelow)
        self.sv_tolerance_slider.valueChanged.connect(self.update_sv_tolerance)
        controls_layout.addWidget(QLabel("Color S/V Tolerance:"))
        controls_layout.addWidget(self.sv_tolerance_slider)

        # Min Wire Area Slider
        self.min_area_slider = QSlider(Qt.Horizontal)
        self.min_area_slider.setMinimum(10)
        self.min_area_slider.setMaximum(1000)
        self.min_area_slider.setValue(self.default_min_contour_area)
        self.min_area_slider.setTickInterval(100)
        self.min_area_slider.setTickPosition(QSlider.TicksBelow)
        self.min_area_slider.valueChanged.connect(self.update_min_wire_area)
        controls_layout.addWidget(QLabel("Min Wire Area:"))
        controls_layout.addWidget(self.min_area_slider)

        # Morph Kernel Size Slider
        self.morph_kernel_slider = QSlider(Qt.Horizontal)
        self.morph_kernel_slider.setMinimum(1)
        self.morph_kernel_slider.setMaximum(15)
        self.morph_kernel_slider.setValue(self.default_morph_kernel_size)
        self.morph_kernel_slider.setTickInterval(2)
        self.morph_kernel_slider.setTickPosition(QSlider.TicksBelow)
        self.morph_kernel_slider.valueChanged.connect(self.update_morph_kernel_size)
        controls_layout.addWidget(QLabel("Morphological Kernel Size:"))
        controls_layout.addWidget(self.morph_kernel_slider)
        
        self.calibrate_button = QPushButton("Run Homography Calibration")
        self.calibrate_button.clicked.connect(self.run_homography_calibration_gui)
        controls_layout.addWidget(self.calibrate_button)

        # Status/Log Area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        controls_layout.addWidget(QLabel("Log/Status:"))
        controls_layout.addWidget(self.log_area)

        main_layout.addLayout(controls_layout, 1)
        self.log_message("Application started. Please load calibration and spec files.")

    def log_message(self, message):
        logger.info(message)
        self.log_area.append(message)

    def start_capture(self):
        logger.info("Start capture button clicked.")
        target_fps = self.fps_slider.value()
        resolution_str = self.resolution_combo.currentText()
        resolution_cap = None
        if resolution_str != "Default":
            w, h = map(int, resolution_str.split('x'))
            resolution_cap = (w, h)

        try:
            self.camera = Camera(device_index=0, target_fps=target_fps, resolution_cap=resolution_cap)
            self.timer.start(int(1000 / target_fps))
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.validate_button.setEnabled(True)
            self.log_message(f"Camera started with Target FPS: {target_fps}, Resolution: {resolution_str}")
            if not self.terminal_locator.is_ready():
                self.log_message("WARNING: Terminal locator not ready. Homography calibration might be missing or invalid.")
        except Exception as e:
            self.log_message(f"Error starting camera: {e}")
            logger.error(f"Error starting camera: {e}", exc_info=True)

    def stop_capture(self):
        logger.info("Stop capture button clicked.")
        self.timer.stop()
        if self.camera:
            self.camera.release()
            self.camera = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.validate_button.setEnabled(False)
        if self.validation_mode_active:
            self.validation_mode_active = False
            self.validate_button.setText("Validate")
            self._reset_correct_message_state()
        self.video_label.setText("Camera stopped.")
        self.log_message("Camera stopped.")

    def update_frame(self):
        if not self.camera:
            return

        frame = self.camera.get_frame()
        if frame is None:
            self.log_message("Failed to get frame from camera.")
            return

        self.current_frame = frame.copy()
        processed_display_frame = preprocess_frame(frame, self.roi_rect)
        output_frame_canvas = processed_display_frame.copy()
        frame_hsv = cv2.cvtColor(output_frame_canvas, cv2.COLOR_BGR2HSV)
        detected_wires_objects = self.wire_detector.detect_wires(frame_hsv)

        if self.validation_mode_active:
            if not self.terminal_locator.is_ready():
                self.log_message("Validation Mode: Terminal locator not ready. Please calibrate.")
                for wire_obj in detected_wires_objects:
                    cv2.polylines(output_frame_canvas, [wire_obj.polyline_px], isClosed=False, color=(0,255,255), thickness=1)
                self.display_image(output_frame_canvas)
                return
            if not self.connection_validator:
                self.log_message("Validation Mode: No wiring specification loaded.")
                for wire_obj in detected_wires_objects:
                    cv2.polylines(output_frame_canvas, [wire_obj.polyline_px], isClosed=False, color=(0,255,255), thickness=1)
                self.display_image(output_frame_canvas)
                return

            wires_for_validation = []
            for wire_obj in detected_wires_objects:
                term_A = self.terminal_locator.pixel_to_terminal(wire_obj.endpoints_px[0])
                term_B = self.terminal_locator.pixel_to_terminal(wire_obj.endpoints_px[1])
                wires_for_validation.append(wire_obj._replace(terminal_A=term_A, terminal_B=term_B))

            for wire in wires_for_validation:
                cv2.polylines(output_frame_canvas, [wire.polyline_px], isClosed=False, color=(0, 255, 0), thickness=2)
                cv2.circle(output_frame_canvas, wire.endpoints_px[0], 5, (255, 0, 0), -1)
                cv2.circle(output_frame_canvas, wire.endpoints_px[1], 5, (255, 0, 0), -1)
                term_A_str = wire.terminal_A if wire.terminal_A and wire.terminal_A != "N/A" else "?"
                term_B_str = wire.terminal_B if wire.terminal_B and wire.terminal_B != "N/A" else "?"
                cv2.putText(output_frame_canvas, f"{wire.color[:3]}:{term_A_str}", (wire.endpoints_px[0][0] + 7, wire.endpoints_px[0][1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                cv2.putText(output_frame_canvas, f"{wire.color[:3]}:{term_B_str}", (wire.endpoints_px[1][0] + 7, wire.endpoints_px[1][1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            validation_results = self.connection_validator.validate_connections(wires_for_validation)
            
            frame_is_currently_all_ok = True if validation_results else False
            for _, status_val in validation_results:
                if status_val != "OK":
                    frame_is_currently_all_ok = False
                    break
            self.current_validation_overall_ok = frame_is_currently_all_ok

            for spec_entry_draw, status_draw in validation_results:
                try:
                    px_from = self.terminal_locator.board_to_pixel(tuple(map(int, spec_entry_draw['from'].split(','))))
                    px_to = self.terminal_locator.board_to_pixel(tuple(map(int, spec_entry_draw['to'].split(','))))

                    if px_from and px_to:
                        mid_x_spec = (px_from[0] + px_to[0]) // 2
                        mid_y_spec = (px_from[1] + px_to[1]) // 2
                        text_offset_y = -10 if mid_y_spec > output_frame_canvas.shape[0] / 2 else 20
                        status_text_color = (0,0,255)
                        (text_width, text_height), _ = cv2.getTextSize(status_draw, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                        text_pos_x = mid_x_spec - text_width // 2

                        if status_draw == "OK":
                            cv2.drawMarker(output_frame_canvas, px_from, (0, 255, 0), cv2.MARKER_STAR, 12, 2)
                            cv2.drawMarker(output_frame_canvas, px_to, (0, 255, 0), cv2.MARKER_STAR, 12, 2)
                        else:
                            cv2.drawMarker(output_frame_canvas, px_from, (0,0,255), cv2.MARKER_TILTED_CROSS, 15, 2)
                            cv2.drawMarker(output_frame_canvas, px_to, (0,0,255), cv2.MARKER_TILTED_CROSS, 15, 2)
                            if "Missing" in status_draw:
                                cv2.line(output_frame_canvas, px_from, px_to, (0, 0, 255), 2)
                            cv2.rectangle(output_frame_canvas, (text_pos_x -2, mid_y_spec + text_offset_y - text_height - 2), (text_pos_x + text_width + 2, mid_y_spec + text_offset_y + 2), (200,200,200), -1)
                            cv2.putText(output_frame_canvas, status_draw, (text_pos_x, mid_y_spec + text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_text_color, 1)
                except Exception as e:
                    logger.error(f"Error drawing validation feedback for spec_entry {spec_entry_draw}: {e}", exc_info=True)
            
            if self.current_validation_overall_ok and validation_results:
                if not self.is_awaiting_correct_message_display and not self.connections_confirmed_correct_and_message_shown:
                    logger.info("All connections correct. Starting delay for confirmation message.")
                    self.correct_message_delay_timer.start(self.CORRECT_MESSAGE_DELAY_MS)
                    self.is_awaiting_correct_message_display = True
            else:
                if self.is_awaiting_correct_message_display:
                    logger.info("Connections became incorrect during delay. Cancelling confirmation message timer.")
                    self.correct_message_delay_timer.stop()
                    self.is_awaiting_correct_message_display = False
                if self.connections_confirmed_correct_and_message_shown:
                    logger.info("Connections no longer correct. Resetting 'all correct' message shown flag.")
                    self.connections_confirmed_correct_and_message_shown = False
            
            self.display_image(output_frame_canvas)

        else: # Normal Live View Drawing (not validation mode)
            self.processed_frame_for_display = output_frame_canvas
            connected_terminal_pixels = set()
            if self.terminal_locator.is_ready():
                for wire_obj in detected_wires_objects:
                    cv2.polylines(self.processed_frame_for_display, [wire_obj.polyline_px], isClosed=False, color=(0,255,0), thickness=2)
                    term_A = self.terminal_locator.pixel_to_terminal(wire_obj.endpoints_px[0])
                    term_B = self.terminal_locator.pixel_to_terminal(wire_obj.endpoints_px[1])
                    connected_terminal_pixels.add(wire_obj.endpoints_px[0])
                    connected_terminal_pixels.add(wire_obj.endpoints_px[1])
                    cv2.circle(self.processed_frame_for_display, wire_obj.endpoints_px[0], 5, (255,0,0), -1)
                    cv2.circle(self.processed_frame_for_display, wire_obj.endpoints_px[1], 5, (255,0,0), -1)
                    cv2.putText(self.processed_frame_for_display, f"{wire_obj.color[:3]}:{term_A}", (wire_obj.endpoints_px[0][0] + 7, wire_obj.endpoints_px[0][1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
                    cv2.putText(self.processed_frame_for_display, f"{wire_obj.color[:3]}:{term_B}", (wire_obj.endpoints_px[1][0] + 7, wire_obj.endpoints_px[1][1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
                    ep1_px, ep2_px = wire_obj.endpoints_px
                    mid_x, mid_y = int((ep1_px[0] + ep2_px[0]) / 2), int((ep1_px[1] + ep2_px[1]) / 2)
                    term_A_str, term_B_str = (term_A if term_A != "N/A" else "?"), (term_B if term_B != "N/A" else "?")
                    wire_label_text = f"{wire_obj.color} ({term_A_str} to {term_B_str})"
                    (text_width, text_height), _ = cv2.getTextSize(wire_label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.putText(self.processed_frame_for_display, wire_label_text, (mid_x - text_width // 2, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            else:
                for wire_obj in detected_wires_objects:
                     cv2.polylines(self.processed_frame_for_display, [wire_obj.polyline_px], isClosed=False, color=(0,255,255), thickness=1)
                     ep1_px, ep2_px = wire_obj.endpoints_px
                     mid_x, mid_y = int((ep1_px[0] + ep2_px[0]) / 2), int((ep1_px[1] + ep2_px[1]) / 2)
                     cv2.putText(self.processed_frame_for_display, f"{wire_obj.color}", (mid_x - 10, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            if self.terminal_locator.is_ready() and self.terminal_locator.board_dimensions:
                all_terminals_info = self.terminal_locator.get_all_terminal_pixel_coordinates()
                for logical_coord, pixel_coord in all_terminals_info:
                    is_connected = any(np.linalg.norm(np.array(pixel_coord) - np.array(cp)) < 10 for cp in connected_terminal_pixels)
                    if not is_connected:
                        cv2.circle(self.processed_frame_for_display, pixel_coord, 2, (128, 128, 128), -1)
            self.display_image(self.processed_frame_for_display)

    def toggle_validation_mode(self):
        logger.info("Toggle validation mode button clicked.")
        if not self.camera or not self.timer.isActive():
            self.log_message("Camera is not running. Please start the camera first.")
            return
        if not self.connection_validator:
            self.log_message("No wiring specification loaded. Please load a spec file.")
            return
        if not self.terminal_locator.is_ready():
            self.log_message("Terminal locator not ready. Please run calibration.")
            return

        self.validation_mode_active = not self.validation_mode_active
        if self.validation_mode_active:
            self.validate_button.setText("Stop Validation")
            self._reset_correct_message_state()
            self.log_message("Real-time validation mode STARTED.")
        else:
            self.validate_button.setText("Validate")
            self._reset_correct_message_state()
            self.log_message("Real-time validation mode STOPPED.")

    def _reset_correct_message_state(self):
        """Helper to reset flags and timer related to the 'all correct' message."""
        if self.is_awaiting_correct_message_display:
            self.correct_message_delay_timer.stop()
            self.is_awaiting_correct_message_display = False
        self.connections_confirmed_correct_and_message_shown = False
        logger.debug("Reset 'all correct' message state.")

    def show_connections_correct_message_after_delay(self):
        logger.debug("Correct message delay timer timed out.")
        self.is_awaiting_correct_message_display = False

        if self.current_validation_overall_ok and not self.connections_confirmed_correct_and_message_shown:
            logger.info("Delay ended. Connections confirmed correct. Showing success message.")
            
            main_timer_was_active = self.timer.isActive()
            if main_timer_was_active: self.timer.stop()

            QMessageBox.information(self, "Validation Complete", "All connections are correct!")
            self.connections_confirmed_correct_and_message_shown = True

            if main_timer_was_active and self.validation_mode_active and self.camera and not self.start_button.isEnabled():
                self.timer.start(int(1000 / self.fps_slider.value()))
        elif not self.current_validation_overall_ok:
            logger.info("Delay ended, but connections are no longer correct. 'All correct' message suppressed.")
            self.connections_confirmed_correct_and_message_shown = False

    def load_spec(self):
        logger.info("Load spec button clicked.")
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Wiring Specification JSON", "specs/",
                                                   "JSON Files (*.json);;All Files (*)", options=options)
        if file_path:
            try:
                self.connection_validator = ConnectionValidator(file_path)
                self.log_message(f"Wiring specification loaded from: {file_path}")
                if not self.connection_validator.required_connections_spec:
                    self.log_message(f"Warning: Specification file {file_path} loaded but was empty or invalid.")
                self._reset_correct_message_state()
            except Exception as e:
                self.log_message(f"Error loading specification file: {e}")
                logger.error(f"Error loading spec: {e}", exc_info=True)

    def run_homography_calibration_gui(self):
        self.log_message("Homography calibration requested. Stopping camera if running.")
        was_running = self.timer.isActive()
        if was_running:
            self.stop_capture()
        
        self.log_message("Starting calibration tool. Follow instructions in the console and new window(s).")
        try:
            import calibrate_homography 
            calibrate_homography.run_calibration(camera_index=0, qt_parent=self)
            self.log_message("Calibration tool finished. Reloading terminal locator.")
            self.terminal_locator = TerminalLocator("calibration/points.json")
            if self.terminal_locator.is_ready():
                self.log_message("Terminal locator reloaded successfully with new calibration.")
            else:
                self.log_message("Terminal locator failed to load after calibration. Check points.json.")
        except Exception as e:
            self.log_message(f"Error during calibration process: {e}")
            logger.error(f"Calibration GUI call error: {e}", exc_info=True)
        
        if was_running:
            self.start_capture()
            
    def update_sv_tolerance(self, value):
        if self.wire_detector:
            self.wire_detector.set_hsv_sv_tolerance(value)
            self.log_message(f"Color S/V tolerance set to: {value}")

    def update_min_wire_area(self, value):
        if self.wire_detector:
            self.wire_detector.set_min_contour_area(value)
            self.log_message(f"Min wire area set to: {value}")

    def update_morph_kernel_size(self, value):
        if self.wire_detector:
            actual_value = value
            if actual_value % 2 == 0:
                actual_value = value + 1 if value < self.morph_kernel_slider.maximum() else value -1
                if actual_value < self.morph_kernel_slider.minimum(): actual_value = self.morph_kernel_slider.minimum()
            self.wire_detector.set_morph_kernel_size(actual_value)
            self.log_message(f"Morph kernel size set to: {actual_value} (input: {value})")

    def display_image(self, cv_img):
        try:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            logger.error(f"Error displaying image: {e}", exc_info=True)
            self.log_message(f"Error displaying image: {e}")

    def closeEvent(self, event):
        logger.info("Close event triggered. Cleaning up.")
        self.stop_capture()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = BreadboardApp()
    main_window.show()
    sys.exit(app.exec_())
