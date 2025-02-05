from pathlib import Path
from typing import List, Tuple
import supervision as sv
from supervision.dataset.utils import save_dataset_images
from supervision.utils.file import save_text_file
from ruamel.yaml import YAML


def create_yolo_yaml_for_landmarks(dict_classes: dict, dataset_path: str, output_yaml_path: str):
    """
    Creates a YOLO .yaml configuration file based on the provided frame annotations for landmarks (Pose Estimation).

    Args:
        dict_classes: Dictionary of class names and their corresponding indices.
        dataset_path: The root directory of the dataset.
        output_yaml_path: The path where the .yaml file will be saved.
    """
    max_keypoints = len(dict_classes)

    # Define the dataset structure
    yolo_config = {
        'path': str(dataset_path),
        'train': 'images',
        'val': 'images',
        'names': {0: 'court'},
        'kpt_shape': [max_keypoints, 3]
    }

    # Use ruamel.yaml to write the configuration
    yaml = YAML()
    yaml.default_flow_style = False

    with open(output_yaml_path, 'w') as yaml_file:
        yaml.dump(yolo_config, yaml_file)

    print(f"YOLO configuration file created at: {output_yaml_path}")



def convert_to_yolo_format(bbox, img_width, img_height):
    """Convert bounding box to YOLO format."""
    x_center = (bbox[0] + bbox[2]) / 2.0 / img_width
    y_center = (bbox[1] + bbox[3]) / 2.0 / img_height
    width = (bbox[2] - bbox[0]) / img_width
    height = (bbox[3] - bbox[1]) / img_height
    return x_center, y_center, width, height


def keypoints_to_yolo_annotations(keypoints: sv.KeyPoints, max_keypoints: int, image_shape: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> str:
    """Create content for the YOLO annotation file."""
    # Reference: https://youtu.be/gA5N54IO1ko?t=824

    img_h, img_w = image_shape[:2]
    classes_and_points = {}
    for xy, _, class_id, _ in keypoints:
        x, y = map(int, xy[0])
        # visibility is 2 (visible)
        classes_and_points[class_id.item()] = (x/img_w, y/img_h, 2)

    # Convert the bbox to YOLO format
    bbox_yolo = convert_to_yolo_format(bbox, img_w, img_h)
    # Start with class id 0
    ret = f"0 {bbox_yolo[0]} {bbox_yolo[1]} {bbox_yolo[2]} {bbox_yolo[3]} "

    # Add the keypoints
    for idx in range(max_keypoints):
        x_relative, y_relative, visibility = classes_and_points.get(idx, "0 0 0")
        ret += f"{x_relative} {y_relative} {visibility} "

    return ret

def dataset_keypoints_to_yolo(dataset_keypoints, dict_keypoints_classes, dataset_bbxes, dir_output: str):
    """
    Prepare training data for YOLOv8 from a dataset of keypoints.

    Args:
        dataset_keypoints: The dataset containing keypoints.
        dict_keypoints_classes: The dictionary containing the classes of the keypoints.
        dataset_bbxes: The dataset containing the bounding boxes.
        dir_output: The directory where the output will be saved.
    """
    dir_output = Path(dir_output).resolve()
    images_directory_path = dir_output / "images"
    annotations_directory_path = dir_output / "labels"
    data_yaml_path = dir_output / "data.yaml"

    save_dataset_images(dataset=dataset_keypoints, images_directory_path=images_directory_path)

    # Create the annotation files
    annot_directory = Path(annotations_directory_path)
    annot_directory.mkdir(parents=True, exist_ok=True)
    # Loop through the dataset and create the annotation files
    for image_path, image, keypoints in dataset_keypoints:
        image_name = Path(image_path).name
        yolo_annotations_name = Path(image_name).with_suffix(".txt")
        yolo_annotations_path = annot_directory / yolo_annotations_name
        lines = keypoints_to_yolo_annotations(
            keypoints=keypoints,
            max_keypoints=len(dict_keypoints_classes),
            image_shape=image.shape,
            bbox=dataset_bbxes.get(image_path),
        )
        save_text_file(lines=[lines.strip()], file_path=yolo_annotations_path)

    create_yolo_yaml_for_landmarks(dict_classes=dict_keypoints_classes, dataset_path=dir_output, output_yaml_path=data_yaml_path)