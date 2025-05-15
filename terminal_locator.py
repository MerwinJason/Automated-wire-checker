"""
Maps pixel coordinates to breadboard terminal IDs using homography.
"""
import cv2
import numpy as np
import json
from logger_setup import logger

logger.info("terminal_locator.py loaded")

class TerminalLocator:
    """
    Manages the mapping from image pixel coordinates to logical breadboard terminals.
    """
    def __init__(self, calibration_file_path="calibration/points.json"):
        """
        Initializes the TerminalLocator.

        Args:
            calibration_file_path (str): Path to the JSON file containing
                                         image points and corresponding board points.
        """
        logger.info(f"Initializing TerminalLocator with calibration file: {calibration_file_path}")
        self.homography_matrix = None
        self.board_dimensions = None # e.g., {"rows": 30, "cols": 10}
        self.homography_matrix_inv = None # type: ignore
        try:
            with open(calibration_file_path, 'r') as f:
                self.calibration_data = json.load(f)
            logger.info(f"Loaded calibration data: {self.calibration_data}")

            image_points_list = self.calibration_data["image_points"]
            board_points_list = self.calibration_data["board_points"] # These are [row, col]

            logger.debug(f"Raw image_points_list from JSON: {image_points_list}")
            logger.debug(f"Raw board_points_list from JSON: {board_points_list}")

            # Ensure they are lists of lists (or at least lists of iterables of length 2)
            if not (isinstance(image_points_list, list) and all(isinstance(pt, list) and len(pt) == 2 for pt in image_points_list)):
                logger.error(f"image_points is not a list of [x,y] pairs: {image_points_list}")
                raise ValueError("Invalid format for image_points in calibration file.")
            
            if not (isinstance(board_points_list, list) and all(isinstance(pt, list) and len(pt) == 2 for pt in board_points_list)):
                logger.error(f"board_points is not a list of [row,col] pairs: {board_points_list}")
                raise ValueError("Invalid format for board_points in calibration file.")

            image_points_np = np.array(image_points_list, dtype=np.float32)
            # For board_points, cv2.findHomography expects (x,y) destination points.
            # Our board_points_list contains [row, col]. We map col to x_board and row to y_board.
            board_points_xy_np = np.array([[pt[1], pt[0]] for pt in board_points_list], dtype=np.float32)

            logger.debug(f"image_points_np shape: {image_points_np.shape}, dtype: {image_points_np.dtype}, ndim: {image_points_np.ndim}")
            logger.debug(f"board_points_xy_np shape: {board_points_xy_np.shape}, dtype: {board_points_xy_np.dtype}, ndim: {board_points_xy_np.ndim}")
            
            if image_points_np.ndim != 2 or image_points_np.shape[1] != 2 or \
               board_points_xy_np.ndim != 2 or board_points_xy_np.shape[1] != 2:
                logger.error("Input points are not correctly shaped Nx2 arrays.")
                raise ValueError("Points must be Nx2 arrays for homography.")
            
            if image_points_np.shape[0] < 4 or board_points_xy_np.shape[0] < 4:
                logger.error("Not enough points for homography calculation. Need at least 4.")
                raise ValueError("Insufficient points for homography. Need at least 4 pairs.")

            self.homography_matrix, mask = cv2.findHomography(image_points_np, board_points_xy_np, cv2.RANSAC, 5.0)
            if self.homography_matrix is None:
                logger.error("Homography calculation failed.")
                raise ValueError("Homography calculation failed.")
            self.homography_matrix_inv = np.linalg.inv(self.homography_matrix) # type: ignore
            logger.info(f"Homography matrix computed: \n{self.homography_matrix}")

            if "board_dimensions" in self.calibration_data:
                self.board_dimensions = self.calibration_data["board_dimensions"]
                logger.info(f"Loaded board dimensions: {self.board_dimensions}")
                
            
        except FileNotFoundError:
            logger.error(f"Calibration file not found: {calibration_file_path}")
            # Allow initialization without calibration for GUI to start
        except Exception as e:
            logger.error(f"Error initializing TerminalLocator: {e}")
            # Allow initialization without calibration

    def is_ready(self):
        """Checks if the homography matrix is available."""
        return self.homography_matrix is not None

    def pixel_to_board(self, pixel_pt):
        """
        Projects a pixel coordinate into board-space coordinates (e.g., row/column indices).

        Args:
            pixel_pt (tuple): An (x, y) pixel coordinate.

        Returns:
            tuple: (row, col) board coordinate, or None if homography is not set.
        """
        if not self.is_ready():
            logger.warning("Homography matrix not available for pixel_to_board.")
            return None
        
        pixel_pt_np = np.array([[pixel_pt]], dtype=np.float32) # Needs to be in shape (1, 1, 2)
        transformed_pt = cv2.perspectiveTransform(pixel_pt_np, self.homography_matrix)
        board_coords = (transformed_pt[0][0][1], transformed_pt[0][0][0]) # (y,x) -> (row, col)
        logger.debug(f"Pixel pt {pixel_pt} transformed to board pt {board_coords}")
        return board_coords
    
    def board_to_pixel(self, board_pt_logical):
        """
        Projects a logical board coordinate (row, col) into pixel-space.

        Args:
            board_pt_logical (tuple): A (row, col) logical board coordinate.

        Returns:
            tuple: (x, y) pixel coordinate, or None if inverse homography is not set.
        """
        if self.homography_matrix_inv is None:
            logger.warning("Inverse homography matrix not available for board_to_pixel.")
            return None
        
        # Board points are (row, col), but perspectiveTransform expects (x,y) order.
        # Our board_points in calibration are [row, col], so we map col to x, row to y for transform.
        board_pt_xy_order = np.array([[(board_pt_logical[1], board_pt_logical[0])]], dtype=np.float32)
        transformed_pt = cv2.perspectiveTransform(board_pt_xy_order, self.homography_matrix_inv)
        pixel_coords = (int(round(transformed_pt[0][0][0])), int(round(transformed_pt[0][0][1])))
        logger.debug(f"Logical board pt {board_pt_logical} transformed to pixel_pt {pixel_coords}")
        return pixel_coords

    def board_to_terminal(self, board_pt):
        """
        Converts board-space coordinates (row/column) to a "row,col" terminal ID string.
        Example: (5.2, 0.9) -> "5,1" (row 5, column 1, after rounding)

        Args:
            board_pt (tuple): A (row, col) board coordinate.

        Returns:
            str: The terminal ID string (e.g., "T5A").
        """
        if board_pt is None:
            return "N/A"
        row, col = board_pt
        # Round to nearest integer for row and column indices
        # This assumes terminals are on a grid.
        # The exact logic might need adjustment based on breadboard layout.
        terminal_row_num = int(round(row))
        terminal_col_num = int(round(col))
        terminal_id = f"{terminal_row_num},{terminal_col_num}"
        logger.debug(f"Board pt {board_pt} converted to terminal ID {terminal_id}")
        return terminal_id

    def pixel_to_terminal(self, pixel_pt):
        """
        Combines pixel_to_board and board_to_terminal.
        """
        logger.debug(f"Converting pixel_pt {pixel_pt} to terminal ID.")
        board_pt = self.pixel_to_board(pixel_pt)
        if board_pt is None:
            logger.warning(f"Could not convert pixel {pixel_pt} to board coordinates.")
            return "N/A"
        return self.board_to_terminal(board_pt)
    
    def get_all_terminal_pixel_coordinates(self):
        """
        Generates pixel coordinates for all terminals on the breadboard based on
        loaded board_dimensions and the inverse homography.

        Returns:
            list[tuple]: A list of ((row, col), (pixel_x, pixel_y)) for each terminal,
                         or an empty list if dimensions or homography are not available.
        """
        if not self.board_dimensions or self.homography_matrix_inv is None:
            logger.warning("Board dimensions or inverse homography not available. Cannot get all terminal pixels.")
            return []

        all_terminals_px = []
        rows = self.board_dimensions.get("rows", 0)
        cols = self.board_dimensions.get("cols", 0)

        for r in range(rows):
            for c in range(cols):
                pixel_coord = self.board_to_pixel((r, c))
                if pixel_coord:
                    all_terminals_px.append(((r, c), pixel_coord))
        logger.info(f"Generated {len(all_terminals_px)} total terminal pixel coordinates.")
        return all_terminals_px