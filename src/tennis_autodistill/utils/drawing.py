from typing import Tuple
import cv2
import supervision as sv

def draw_keypoints(annotated_frame, detected_keypoints: sv.KeyPoints, color:Tuple[int, int, int]=None, radius=5):
    """
    Draw keypoints on the annotated frame.

    Args:
        annotated_frame (np.ndarray): The frame on which keypoints will be drawn.
        detected_keypoints (sv.KeyPoints): The detected keypoints to draw.
        color (tuple): The color of the keypoints. Default is (255, 0, 0).
        radius (int): The radius of the keypoints. Default is 5.
    """
    DEFAULT_COLOR = (255, 0, 0)
    for idx, (x, y) in enumerate(detected_keypoints.xy.squeeze()):
        if color is None:
            color_from_pallete = sv.ColorPalette.DEFAULT.colors[idx]
            _color = color_from_pallete.as_bgr()
        else:
            _color = DEFAULT_COLOR
        # Draw a circle at each keypoint location
        cv2.circle(annotated_frame, (int(x), int(y)), radius=radius, color=_color, thickness=-1)
    return annotated_frame


def draw_minimap(frame, minimap, alpha=0.95):
    """
    Draw the minimap into the right corner of the frame.
    """
    _frame = frame.copy()
    height_minimap = _frame.shape[0]
    width_minimap = round(height_minimap * minimap.shape[1] / minimap.shape[0])
    minimap = cv2.resize(minimap, (width_minimap, height_minimap))
    offset_x, offset_y = -minimap.shape[1], minimap.shape[0]
    roi = _frame[0:offset_y, offset_x:]
    # Blend the images with alpha transparency for the minimap
    blended = cv2.addWeighted(minimap, alpha, roi, 1 - alpha, 0)
    # Place the blended result back on the frame
    _frame[0:offset_y, offset_x:] = blended
    return _frame