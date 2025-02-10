from pathlib import Path
import cv2
import numpy as np
import supervision as sv
import fire

import torch
from tqdm import tqdm
from ultralytics import YOLO

from tennis_autodistill.utils.drawing import draw_minimap
from tennis_autodistill.utils.transformations import transform_point
from tennis_autodistill.utils.coords_court_map import COORDS_COURT_MAP
from tennis_autodistill.utils.drawing import draw_keypoints

class Pipeline:
    def __init__(self, path_classification_model, path_player_detector, path_keypoints_detector):
        # Check if paths exist for classification, player and keypoints models
        path_classification_model = Path(path_classification_model)
        if not path_classification_model.exists():
            raise FileNotFoundError(f"The path for path_classification_model does not exist: {path_classification_model}")
        path_player_detector = Path(path_player_detector)
        if not path_player_detector.exists():
            raise FileNotFoundError(f"The path for path_player_detector does not exist: {path_player_detector}")
        path_keypoints_detector = Path(path_keypoints_detector)
        if not path_keypoints_detector.exists():
            raise FileNotFoundError(f"The path for path_keypoints_detector does not exist: {path_keypoints_detector}")

        # Load models into device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🛠️ Loading models into {self.device}...")
        self.model_classification = YOLO(path_classification_model).to(self.device)
        print("✅ Model classification loaded")
        self.model_player_detector = YOLO(path_player_detector).to(self.device)
        print("✅ Model player detector loaded")
        self.model_keypoints_detector = YOLO(path_keypoints_detector).to(self.device)
        print("✅ Model keypoints detector loaded")

        # Create annotators
        self.bounding_box_annotator = sv.BoxAnnotator(thickness=4)

        # Define destination points for homography (court map)
        self.pts_minimap = np.array(list(COORDS_COURT_MAP.values()))
        self.minimap = cv2.imread("tennis_court_full_map.png")

        # Load minimap image
        assert self.minimap is not None, "Minimap image not found"

        # Define progress bar to process frames
        self.pbar = None

    def process_frame(self, frame: np.ndarray, index:int) -> np.ndarray:
        self.pbar.update(1)
        self.pbar.set_description(f"Processing frame {index}")

        # Classify frame into (0: close_up, 1: game_play, 2: ignore1, 3: ignore2, 4: partial_view)
        result_classification = self.model_classification(frame, verbose=False)
        # Keep frame in the pipeline if the classification is 1 "game play" with a min confidence of 75%
        if result_classification[0].probs.top1 != 1 or result_classification[0].probs.top1 < 0.75:
            return frame

        annotated_frame = frame.copy()
        minimap = self.minimap.copy()

        # Run player detector
        result_player_detection = self.model_player_detector(frame, verbose=False, conf=.5)[0]
        detected_players = sv.Detections.from_ultralytics(result_player_detection)
        # Get the 2 players that are closest to the bottom of the frame
        ids_players_closest_to_bottom = np.argsort(detected_players.xyxy[:,3])[::-1][:2]
        detected_players = detected_players[ids_players_closest_to_bottom]

        # Detect keypoints
        result_keypoints = self.model_keypoints_detector(frame, verbose=False)[0]
        detected_keypoints = sv.KeyPoints.from_ultralytics(result_keypoints)

        # Compute the homography matrix
        source_pts = np.array(detected_keypoints.xy.squeeze())
        H_court2map, _ = cv2.findHomography(source_pts, self.pts_minimap)
        # Transform the center of the bottom part of the players into the court map
        for player_xyxy in detected_players.xyxy:
            x1,_, x2, y2 = player_xyxy
            xy_bottom_center_player = ((x1 + x2) / 2).item(), y2.item()
            center_player_minimap = transform_point(H_court2map, xy_bottom_center_player)
            center_player_minimap = tuple(map(round, center_player_minimap))
            # Draw the minimap
            cv2.circle(minimap, center_player_minimap, 50, (0, 0, 255), -1)

        # Annotate
        annotated_frame = self.bounding_box_annotator.annotate(annotated_frame, detections=detected_players)
        annotated_frame = draw_keypoints(annotated_frame, detected_keypoints)
        # Include the minimap into the right coner of the frame
        annotated_frame = draw_minimap(annotated_frame, minimap)
        return annotated_frame

    def run(self, input_video, output_folder):
        # Check if input_video exists
        if not Path(input_video).exists():
            raise FileNotFoundError(f"The path for input_video does not exist: {input_video}")

        # Define output video path
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        output_video_path = output_folder / f"output-{Path(input_video).stem}.mp4"

        video_info = sv.VideoInfo.from_video_path(input_video)
        self.pbar = tqdm(total=video_info.total_frames, desc="Processing frames")

        # Process video
        sv.process_video(
            source_path = input_video,
            target_path = output_video_path,
            callback=self.process_frame
        )

def main(input_video, path_classification_model, path_player_detector, path_keypoints_detector, output_folder):
    """
    Run the pipeline with the specified parameters.

    Args:
        input_video (str): Path to the input video file.
        path_classification_model (str): Path to the classification model.
        path_player_detector (str): Path to the player detector model.
        path_keypoints_detector (str): Path to the keypoints detector model.
        output_folder (str): Path to the output folder.
    """
    pipeline = Pipeline(path_classification_model, path_player_detector, path_keypoints_detector)
    pipeline.run(input_video, output_folder)

if __name__ == "__main__":
    fire.Fire(main)

