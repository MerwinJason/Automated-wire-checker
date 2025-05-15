"""
Image preprocessing routines for normalizing camera frames before analysis.
Includes cropping and image enhancement techniques.
"""
import cv2
import numpy as np
from logger_setup import logger

logger.info("preprocess.py loaded")

def crop_frame(frame, roi_rect):
    """
    Crops the frame to a specified region of interest (ROI).

    Args:
        frame (numpy.ndarray): The input BGR image.
        roi_rect (tuple): A tuple (x, y, w, h) defining the ROI.
                          If None or invalid, returns the original frame.

    Returns:
        numpy.ndarray: The cropped BGR image.
    """
    logger.debug(f"Cropping frame with ROI: {roi_rect}")
    if roi_rect and len(roi_rect) == 4:
        x, y, w, h = roi_rect
        if x >= 0 and y >= 0 and w > 0 and h > 0 and (x + w) <= frame.shape[1] and (y + h) <= frame.shape[0]:
            cropped = frame[y:y+h, x:x+w]
            logger.debug(f"Frame cropped to shape: {cropped.shape}")
            return cropped
        else:
            logger.warning(f"Invalid ROI {roi_rect} for frame shape {frame.shape}. Returning original frame.")
            return frame
    logger.debug("No ROI specified or invalid ROI, returning original frame.")
    return frame

def normalize_frame(frame):
    """
    Normalizes the frame using histogram equalization (CLAHE on L-channel of LAB).

    Args:
        frame (numpy.ndarray): The input BGR image.

    Returns:
        numpy.ndarray: The normalized BGR image.
    """
    logger.debug("Normalizing frame.")
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a_channel, b_channel))
    normalized = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    logger.debug("Frame normalized.")
    return normalized

def preprocess_frame(frame, roi_rect=None):
    """
    Chains preprocessing steps: cropping and normalization.
    """
    logger.info(f"Preprocessing frame with ROI: {roi_rect}")
    processed_frame = crop_frame(frame, roi_rect)
    processed_frame = normalize_frame(processed_frame)
    logger.info("Frame preprocessing complete.")
    return processed_frame