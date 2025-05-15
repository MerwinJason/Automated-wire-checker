"""
Calibration tool for generating homography points.
Allows users to click on points in an image of the breadboard and
input their corresponding logical board coordinates.
Saves these points to 'calibration/points.json'.
"""
import cv2
import numpy as np
import json
import os
from logger_setup import logger

try:
    from PyQt5.QtWidgets import QInputDialog
except ImportError:
    QInputDialog = None # Fallback if PyQt5 is not available (e.g. running standalone without GUI integration)



image_points = []
board_points_input = [] # Store as (row, col)

def mouse_callback(event, x, y, flags, param):
    """Handles mouse clicks to select points on the image."""
    global image_points, board_points_input
    qt_parent = param.get('qt_parent')
    image_display = param['image_display']
    if event == cv2.EVENT_LBUTTONDOWN:
        image_points.append([x, y])
        logger.info(f"Image point selected: ({x}, {y})")
        
        # Prompt for board coordinates
        # For simplicity, using console input. A GUI input would be nicer.
        while True:
            try:
                row_str, col_str = None, None
                if qt_parent and QInputDialog:
                    row_str_dialog, ok1 = QInputDialog.getText(qt_parent, "Calibration Input", f"Enter BOARD ROW index for pixel ({x},{y}):")
                    if not ok1: # User cancelled
                        image_points.pop() # Remove the last added image point
                        logger.info("Calibration input cancelled by user for row.")
                        return
                    col_str_dialog, ok2 = QInputDialog.getText(qt_parent, "Calibration Input", f"Enter BOARD COLUMN index for pixel ({x},{y}):")
                    if not ok2: # User cancelled
                        image_points.pop() # Remove the last added image point
                        logger.info("Calibration input cancelled by user for column.")
                        return
                    row_str, col_str = row_str_dialog, col_str_dialog
                else:
                    # Fallback to console input if not run from GUI or QInputDialog is unavailable
                    print(f"Selected pixel: ({x},{y})")
                    row_str = input(f"Enter BOARD ROW index: ")
                    col_str = input(f"Enter BOARD COLUMN index: ")
                row = int(row_str)
                col = int(col_str)
                board_points_input.append([row, col]) # Store as [row, col]
                logger.info(f"Board point entered: (row={row}, col={col})")
                break
            except ValueError:
                err_msg = "Invalid input. Please enter integer values for row and column."
                logger.error(err_msg)
                if qt_parent and QInputDialog: # Should ideally show a Qt message box here
                    print(f"Error: {err_msg}") # Print to console as fallback
                else:
                    print(err_msg)
        
        # Draw on image
        cv2.circle(image_display, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(image_display, f"P{len(image_points)}", (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.imshow("Calibration Image - Click Points (Press 's' to save, 'q' to quit)", image_display)
        
def run_calibration(image_path=None, camera_index=0, qt_parent=None):
    """
    Runs the calibration process.

    Args:
        image_path (str, optional): Path to a static image for calibration.
                                    If None, uses the camera.
        camera_index (int): Camera device index if image_path is None.
        qt_parent (QWidget, optional): Parent QWidget for Qt dialogs
    """
    logger.info(f"Starting calibration. Image path: {image_path}, Camera index: {camera_index}")
    
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            print(f"Error: Could not load image {image_path}")
            return
    else:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error(f"Cannot open camera {camera_index}")
            print(f"Error: Cannot open camera {camera_index}")
            return
        print("Press 'c' to capture a frame for calibration, then 'q' to close camera view.")
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame from camera.")
                print("Error: Failed to capture frame.")
                cap.release()
                return
            cv2.imshow("Camera Feed - Press 'c' to capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                img = frame.copy()
                logger.info("Frame captured for calibration.")
                break
            elif key == ord('q'):
                logger.info("Calibration aborted by user from camera view.")
                cap.release()
                cv2.destroyAllWindows()
                return
        cap.release()
        cv2.destroyWindow("Camera Feed - Press 'c' to capture")

    img_display = img.copy()
    cv2.namedWindow("Calibration Image - Click Points (Press 's' to save, 'q' to quit)")
    cv2.setMouseCallback(
        "Calibration Image - Click Points (Press 's' to save, 'q' to quit)",
        mouse_callback,
        param={'image_display': img_display, 'qt_parent': qt_parent})
    cv2.imshow("Calibration Image - Click Points (Press 's' to save, 'q' to quit)", img_display)

    print("Click at least 4 corresponding points on the image. Press 's' to save, 'q' to quit without saving.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if len(image_points) >= 4 and len(image_points) == len(board_points_input):
                calibration_dir = "calibration"
                os.makedirs(calibration_dir, exist_ok=True)
                output_path = os.path.join(calibration_dir, "points.json")
                 # Get board dimensions
                board_rows_str, board_cols_str = None, None
                try:
                    if qt_parent and QInputDialog:
                        board_rows_str, ok_rows = QInputDialog.getText(qt_parent, "Board Dimensions", "Enter total number of logical BOARD ROWS:")
                        if not ok_rows: raise ValueError("User cancelled row input")
                        board_cols_str, ok_cols = QInputDialog.getText(qt_parent, "Board Dimensions", "Enter total number of logical BOARD COLUMNS:")
                        if not ok_cols: raise ValueError("User cancelled col input")
                    else:
                        print("Enter the dimensions of your breadboard's terminal grid.")
                        board_rows_str = input("Total number of logical BOARD ROWS: ")
                        board_cols_str = input("Total number of logical BOARD COLUMNS: ")
                    
                    num_rows = int(board_rows_str)
                    num_cols = int(board_cols_str)

                    data = {
                        "image_points": image_points, 
                        "board_points": board_points_input,
                        "board_dimensions": {"rows": num_rows, "cols": num_cols}
                    }
                    with open(output_path, 'w') as f:
                        json.dump(data, f, indent=4)
                    logger.info(f"Calibration data saved to {output_path}: {data}")
                    print(f"Calibration data (including dimensions) saved to {output_path}")

                except ValueError as e:
                    logger.error(f"Invalid input for board dimensions: {e}. Data not saved with dimensions.")
                    print(f"Error getting board dimensions: {e}. Points saved without dimensions if you proceed or re-save.")
                    # Optionally, save without dimensions or prompt again
                    # For now, we'll just log and print the error.
                    
            else:
                logger.warning(f"Not enough points to save. Need at least 4. Got {len(image_points)} image points and {len(board_points_input)} board points.")
                print("Not enough points (need at least 4 pairs). Data not saved.")
            break
        elif key == ord('q'):
            logger.info("Calibration quit by user. Data not saved.")
            print("Calibration quit. Data not saved.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage:
    # To use a static image:
    # print("Ensure you have an image of your breadboard, e.g., 'my_breadboard.jpg'")
    # run_calibration(image_path="path_to_your_breadboard_image.jpg")

    # To use the default camera:
    print("Using live camera feed for calibration.")
    run_calibration(camera_index=0)