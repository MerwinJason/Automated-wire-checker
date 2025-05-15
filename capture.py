"""
Handles camera capture operations.
Provides a class to interface with a video device, apply frame rate
and resolution limits, and retrieve frames.
"""
import cv2
import time
from logger_setup import logger

logger.info("capture.py loaded")

class Camera:
    """
    Manages video capture from a specified camera device.
    """
    def __init__(self, device_index=0, target_fps=30, resolution_cap=None):
        """
        Initializes the camera.

        Args:
            device_index (int): The index of the video device (e.g., 0 for default camera).
            target_fps (int): The desired frames per second.
            resolution_cap (tuple, optional): A tuple (width, height) for resolution.
                                              Defaults to None (camera default).
        """
        logger.info(f"Initializing Camera with device_index={device_index}, target_fps={target_fps}, resolution_cap={resolution_cap}")
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera device at index {device_index}")
            raise IOError(f"Cannot open video device {device_index}")

        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps if target_fps > 0 else 0
        self.last_frame_time = 0

        if resolution_cap and isinstance(resolution_cap, tuple) and len(resolution_cap) == 2:
            logger.info(f"Attempting to set resolution to {resolution_cap[0]}x{resolution_cap[1]}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_cap[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_cap[1])
        
        # Some cameras might not respect FPS settings directly, this is a software cap.
        # self.cap.set(cv2.CAP_PROP_FPS, target_fps) # This often doesn't work reliably

        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS) # May return 0 or incorrect value
        logger.info(f"Camera opened. Actual resolution: {actual_width}x{actual_height}, Reported FPS: {actual_fps}")


    def get_frame(self):
        """
        Retrieves the latest frame from the camera, enforcing the target FPS.

        Returns:
            numpy.ndarray: The captured BGR image frame, or None if an error occurs.
        """
        logger.debug("Attempting to get frame.")
        if self.frame_time > 0:
            current_time = time.time()
            time_to_wait = self.frame_time - (current_time - self.last_frame_time)
            if time_to_wait > 0:
                time.sleep(time_to_wait)
            self.last_frame_time = time.time()

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to retrieve frame from camera.")
            return None
        logger.debug(f"Frame retrieved successfully, shape: {frame.shape}")
        return frame

    def release(self):
        """
        Releases the video capture device.
        """
        logger.info("Releasing camera.")
        if self.cap.isOpened():
            self.cap.release()
        logger.info("Camera released.")