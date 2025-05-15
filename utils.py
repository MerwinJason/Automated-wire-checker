"""
Utility definitions for the breadboard checker application, including
color definitions and data structures.
"""
from collections import namedtuple
import numpy as np
from logger_setup import logger

logger.info("utils.py loaded")

# Define a simple structure for HSV color ranges
HSVRange = namedtuple("HSVRange", ["lower", "upper"])

# Define a structure for detected wires
Wire = namedtuple("Wire", ["color", "endpoints_px", "polyline_px", "terminal_A", "terminal_B"])


COLOR_RANGES = {
    # Note: These HSV ranges are examples and WILL need tuning for your specific camera and lighting.
    # OpenCV HSV ranges: H: 0-179, S: 0-255, V: 0-255
    "red": HSVRange(
        lower=np.array([0, 120, 70]), upper=np.array([10, 255, 255])
    ), # Also consider second range for red: np.array([170, 120, 70]), np.array([180, 255, 255])
    "green": HSVRange(
        lower=np.array([35, 100, 50]), upper=np.array([85, 255, 255])
    ),
    "blue": HSVRange(
        lower=np.array([90, 100, 50]), upper=np.array([130, 255, 255])
    ),
    "yellow": HSVRange(
        lower=np.array([20, 100, 100]), upper=np.array([30, 255, 255])
    ),
    "orange": HSVRange(
        lower=np.array([10, 100, 100]), upper=np.array([20, 255, 255])
    ),
    "black": HSVRange( # Black can be tricky, depends on lighting.
        lower=np.array([0, 0, 0]), upper=np.array([180, 255, 50]) # Low V value
    ),
    "white": HSVRange( # White also tricky, often high V, low S.
        lower=np.array([0, 0, 180]), upper=np.array([180, 30, 255])
    )
}
logger.info(f"Defined COLOR_RANGES: {COLOR_RANGES.keys()}")