"""
Marker Detector — Computer Vision Recruitment Task

Implement the MarkerDetector class and the utility functions below.
See README.md for full task description.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple


class MarkerDetector:
    """
    Detects colored markers in images using classical computer vision techniques.

    Each detection is a dictionary with the following fields:
        - 'color':  str           — one of 'red', 'green', 'blue', 'yellow'
        - 'bbox':   (x, y, w, h) — bounding rectangle of the detected contour
        - 'center': (cx, cy)     — center coordinates of the bounding box
        - 'area':   float        — area of the detected contour

    Optionally, if you attempt the bonus task:
        - 'shape':  str          — one of 'circle', 'triangle', 'rectangle'
    """

    # Define HSV color ranges for each target color.
    # Each entry maps a color name to a list of (lower_bound, upper_bound) tuples.
    # Use np.array([H, S, V]) for bounds. OpenCV uses H: 0-179, S: 0-255, V: 0-255.
    #
    # Hint: Red wraps around the hue spectrum (both ~0-10 and ~170-179 are red),
    # so you will likely need TWO ranges for red.
    COLOR_RANGES = {
    "red": [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([170, 100, 100]), np.array([179, 255, 255])),
    ],
    "green": [
        (np.array([35, 80, 80]), np.array([85, 255, 255]))
    ],
    "blue": [
        (np.array([90, 80, 80]), np.array([130, 255, 255]))
    ],
    "yellow": [
        (np.array([20, 100, 100]), np.array([35, 255, 255]))
    ],
        # Example format:
        # "green": [(np.array([35, 80, 80]), np.array([85, 255, 255]))],
    }
    # Minimum contour area to consider (filters noise)
    min_area = 500

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect colored markers in the given BGR image.

        Args:
            image: Input image in BGR format (as loaded by cv2.imread).

        Returns:
            A list of detection dictionaries, each containing:
            'color', 'bbox', 'center', and 'area' keys.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        detections = []

        for color, ranges in self.COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

            for lower, upper in ranges:
                mask = cv2.bitwise_or(
                    mask,
                    cv2.inRange(hsv, lower, upper)
                )

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)

                if area < self.min_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)

                detections.append({
                    "color": color,
                    "bbox": (x, y, w, h),
                    "center": center,
                    "area": float(area),
                })

        return detections
def compute_iou(box_a: Tuple, box_b: Tuple) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Each box is represented as (x, y, w, h) where:
        - (x, y) is the top-left corner
        - (w, h) is the width and height

    Args:
        box_a: First bounding box as (x, y, w, h).
        box_b: Second bounding box as (x, y, w, h).

    Returns:
        IoU value as a float between 0.0 (no overlap) and 1.0 (perfect overlap).
    """
    x_a, y_a, w_a, h_a = box_a
    x_b, y_b, w_b, h_b = box_b

    x_left = max(x_a, x_b)
    y_top = max(y_a, y_b)
    x_right = min(x_a + w_a, x_b + w_b)
    y_bottom = min(y_a + h_a, y_b + h_b)

    intersection_w = max(0, x_right - x_left)
    intersection_h = max(0, y_bottom - y_top)
    intersection_area = intersection_w * intersection_h

    area_a = w_a * h_a
    area_b = w_b * h_b
    union_area = area_a + area_b - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def filter_detections(
    detections: List[Dict], iou_threshold: float = 0.5
) -> List[Dict]:
    """
    Filter overlapping detections using Non-Maximum Suppression (NMS).

    When two detections overlap (IoU > iou_threshold), keep the one with the
    larger area and discard the other.

    Args:
        detections: List of detection dictionaries (each must have 'bbox' and 'area').
        iou_threshold: IoU threshold above which two detections are considered overlapping.

    Returns:
        Filtered list of detections with overlapping duplicates removed.
    """
    if not detections:
        return []

    sorted_detections = sorted(
        detections,
        key=lambda detection: detection["area"],
        reverse=True
    )

    filtered = []

    for detection in sorted_detections:
        should_keep = True

        for kept_detection in filtered:
            iou = compute_iou(
                detection["bbox"],
                kept_detection["bbox"]
            )

            if iou > iou_threshold:
                should_keep = False
                break

        if should_keep:
            filtered.append(detection)

    return filtered
