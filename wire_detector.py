"""
Detects wires in an image based on color, and identifies their
endpoints and approximate shape.
"""
import cv2
import numpy as np
from logger_setup import logger
from utils import COLOR_RANGES, Wire # Assuming COLOR_RANGES is imported from utils

logger.info("wire_detector.py loaded")

class WireDetector:
    """
    Detects wires of specified colors in an HSV image.
    """
    def __init__(self, base_color_ranges_map=None,
                 initial_sv_tolerance=0,
                 initial_min_contour_area=100,
                 initial_morph_kernel_size=5):
        """
        Initializes the WireDetector.

        Args:
            base_color_ranges_map (dict, optional): Base HSV ranges. Defaults to COLOR_RANGES.
            initial_sv_tolerance (int): Initial tolerance for S and V values.
            initial_min_contour_area (int): Initial minimum area for a contour to be a wire.
            initial_morph_kernel_size (int): Initial size for the morphological kernel (must be odd).
        """
        self.base_color_ranges = base_color_ranges_map if base_color_ranges_map is not None else COLOR_RANGES
        self.current_color_ranges = {} # Will be populated by set_hsv_sv_tolerance
        self.min_contour_area = initial_min_contour_area
        self.morph_kernel_size = initial_morph_kernel_size if initial_morph_kernel_size % 2 != 0 else initial_morph_kernel_size + 1

        self.set_hsv_sv_tolerance(initial_sv_tolerance) # Initialize current_color_ranges
        logger.info(f"WireDetector initialized. Base colors: {len(self.base_color_ranges)}. Min Area: {self.min_contour_area}. Morph Kernel: {self.morph_kernel_size}")

    def _find_furthest_points(self, contour):
        """
        Finds the two furthest apart points in a contour.
        This is a simple O(N^2) approach, can be optimized for very large contours.
        For typical wire contours, this should be acceptable.
        """
        max_dist = -1
        pt1, pt2 = None, None
        if len(contour) < 2:
            return None, None

        # A more robust approach for wire-like shapes: use minAreaRect
        rect = cv2.minAreaRect(contour) # ((center_x, center_y), (width, height), angle)
        box_float = cv2.boxPoints(rect) # Get 4 corners of the rotated rectangle, returns float
        box = box_float.astype(np.int32) # Convert to integer coordinates


        # The endpoints are likely the midpoints of the two shorter sides of the bounding rectangle
        # Or, for simpler straight wires, two opposing corners of the minAreaRect.
        # Let's calculate distances between all pairs of box points.
        # The longest distance between corners of the minAreaRect often corresponds to the wire's length.
        # The two points forming this longest diagonal are good candidates for endpoints.
        
        dists = []
        for i in range(4):
            for j in range(i + 1, 4):
                dists.append((np.linalg.norm(box[i] - box[j]), box[i], box[j]))
        
        dists.sort(key=lambda x: x[0], reverse=True)

        # The two longest diagonals of the minAreaRect should be similar in length.
        # The points forming these are the corners.
        # For a wire, we want the two points that define its "length".
        # A simpler heuristic: take the two points from the contour that are furthest apart.
        # This can be slow. Let's use the minAreaRect corners.
        # The two points that form the longest side of the minAreaRect are good candidates.
        # The width and height from minAreaRect are (rect[1][0], rect[1][1])
        # If width > height, endpoints are along the width.

        # For simplicity and robustness for now, let's use the two most distant points from the contour itself.
        # This is computationally more expensive but direct.
        # Consider simplifying if performance is an issue.
        for i in range(len(contour)):
            for j in range(i + 1, len(contour)):
                d = np.linalg.norm(contour[i][0] - contour[j][0])
                if d > max_dist:
                    max_dist = d
                    pt1 = tuple(contour[i][0])
                    pt2 = tuple(contour[j][0])
        return pt1, pt2

    def set_hsv_sv_tolerance(self, sv_tolerance):
        """
        Adjusts the Saturation and Value tolerance for color detection.
        Hue ranges remain fixed from the base.

        Args:
            sv_tolerance (int): Value to add/subtract from S and V range limits.
        """
        logger.info(f"Setting HSV S/V tolerance to: {sv_tolerance}")
        self.current_color_ranges = {}
        for color_name, base_range in self.base_color_ranges.items():
            new_lower_s = np.clip(base_range.lower[1] - sv_tolerance, 0, 255)
            new_upper_s = np.clip(base_range.upper[1] + sv_tolerance, 0, 255)
            new_lower_v = np.clip(base_range.lower[2] - sv_tolerance, 0, 255)
            new_upper_v = np.clip(base_range.upper[2] + sv_tolerance, 0, 255)

            new_lower = np.array([base_range.lower[0], new_lower_s, new_lower_v], dtype=np.uint8)
            new_upper = np.array([base_range.upper[0], new_upper_s, new_upper_v], dtype=np.uint8)
            
            # self.current_color_ranges[color_name] = Wire._fields['color']._replace(lower=new_lower, upper=new_upper) # Incorrect: Wire is for detected wire instances
            # Correct way: base_range is an HSVRange object, so type(base_range) is the HSVRange class
            # If Wire.color is just a string, and HSVRange is separate:
            self.current_color_ranges[color_name] = type(base_range)(lower=new_lower, upper=new_upper)
        logger.debug(f"Updated current_color_ranges: {self.current_color_ranges}")

    def set_min_contour_area(self, area):
        logger.info(f"Setting min contour area to: {area}")
        self.min_contour_area = area

    def set_morph_kernel_size(self, size):
        self.morph_kernel_size = size if size % 2 != 0 else size + 1 # Ensure odd
        logger.info(f"Setting morph kernel size to: {self.morph_kernel_size}")

    def detect_wires(self, frame_hsv):
        """
        Detects wires in the provided HSV frame.

        Args:
            frame_hsv (numpy.ndarray): The input image in HSV color space.

        Returns:
            list[Wire]: A list of detected Wire objects.
        """
        logger.info("Starting wire detection.")
        detected_wires = []

        if not self.current_color_ranges:
            logger.warning("Current color ranges not set in WireDetector. Using base ranges.")
            active_ranges = self.base_color_ranges
        else:
            active_ranges = self.current_color_ranges

        for color_name, hsv_range in active_ranges.items():
            logger.debug(f"Detecting wires for color: {color_name}")
            mask = cv2.inRange(frame_hsv, hsv_range.lower, hsv_range.upper)
            
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            logger.debug(f"Found {len(contours)} contours for color {color_name} after morphology.")

            for contour in contours:
                if cv2.contourArea(contour) < self.min_contour_area: # Skip tiny blobs (tune this threshold)
                    continue

                # Approximate contour to a simpler polyline
                epsilon = 0.01 * cv2.arcLength(contour, True) # Adjust epsilon for more/less simplification
                polyline_px = cv2.approxPolyDP(contour, epsilon, True)

                # Find endpoints (two furthest apart points in the contour)
                # This is a simplified approach. More advanced methods might be needed for complex wire shapes.
                endpoint1_px, endpoint2_px = self._find_furthest_points(contour)

                if endpoint1_px and endpoint2_px:
                    wire = Wire(color=color_name, endpoints_px=(endpoint1_px, endpoint2_px), polyline_px=polyline_px, terminal_A=None, terminal_B=None)
                    detected_wires.append(wire)
                    logger.debug(f"Detected wire: Color={color_name}, Endpoints={endpoint1_px}, {endpoint2_px}, Polyline points={len(polyline_px)}")
        
        logger.info(f"Wire detection complete. Found {len(detected_wires)} wires.")
        return detected_wires

    def detect_wires_ml(self, frame):
        """
        Placeholder for a future ML-based wire segmentation method.
        """
        logger.warning("detect_wires_ml() is a placeholder and not implemented.")
        # This would involve running a pre-trained model (e.g., U-Net, Mask R-CNN)
        # on the frame to get segmentation masks for wires, then processing those masks.
        return []