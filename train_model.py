
import cv2
import numpy as np
import supervision as sv
import pickle
import shutil
from tqdm import tqdm
import os
import random
from itertools import chain
from pathlib import Path
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8
from autodistill_gpt_4v import GPT4V
from dotenv import load_dotenv
from utils.coords_court_map import COORDS_COURT_MAP
from utils.transformations import transform_image, transform_point
from utils.video import split_video_to_frames
from utils.yolo import dataset_keypoints_to_yolo
from supervision.dataset.formats.yolo import save_yaml_file

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EPSILON = 1e-9
# For visualization
SAMPLE_GRID_SIZE = (4, 4)
SAMPLE_PLOT_SIZE = (16, 10)
SHOW_COUNT = 16

dir_dataset = Path("dataset/videos")
dir_extracted_frames = Path("dataset/frames")
dir_output_annotations_gpt4v = Path("dataset/frames_labeled_by_gpt4v")
dir_classified_frames = Path("dataset/frames_classified_as_playable_area_by_yolov8")

# Proportions of training and testing sets
prop_train = 0.8
prop_test = 0.2

################################################################################
# Creating the dataset
################################################################################
lst_video_paths = list(dir_dataset.glob("*.mp4"))
# for video_path in lst_video_paths:
#     split_video_to_frames(
#         video_path=video_path,
#         output_folder = dir_extracted_frames,
#         frames_per_second=1,
#         overwrite=True
# )

# Separate videos by court type
# dict_videos_court_type = {}
# for video_path in dataset_dir.glob("*.mp4"):
#     court_type = video_path.stem.split("-")[0]
#     dict_videos_court_type.setdefault(court_type, []).append(video_path.resolve())

# videos_split = {"train": [], "test": []}
# # For each court type, separate videos into training and testing sets
# for court_type, videos in dict_videos_court_type.items():
#     # Shuffle videos
#     random.shuffle(videos)
#     # Get the number of videos to use for training and testing
#     num_train = int(len(videos) * prop_train)
#     num_test = int(len(videos) * prop_test)
#     # Separate videos into training and testing sets
#     videos_split["train"].extend(videos[:num_train])
#     videos_split["test"].extend(videos[num_train:])

# Print the number of frames in each split
# for split_name in videos_split.keys():
#     output_folder = output_frames_dir / split_name
#     print(f"Number of frames in {split_name}: {len(list(output_folder.glob('*.jpg')))}")

dict_ontology = {
            "an image showing the full playable area of a tennis court with camera positioned higher and centered, providing a full view of the entire court": "game_play",
            "an image showing partially the playable area of a tennis court, providing a partial view of the court": "partial_view",
            "a close-up shot of a tennis player": "close_up",
            "anything else": "ignore1",
            "none": "ignore2"
        }
classes_to_return = [f"* {v}" for v in dict_ontology]

prompt = f"""What is in the best description of the image?
Return one of these options that best describes the image:
{'\n'.join(classes_to_return)}
You MUST return only one of the options from that list.
Nothing besides that.
"""

def post_process(result):
    result = result.lower()
    result = result.replace("*", "")
    result = result.replace("'", "")
    result = result.strip()
    return result

base_model_gpt4v = GPT4V(
    ontology=CaptionOntology(
        dict_ontology
    ),
    api_key=OPENAI_API_KEY,
    resize_min=360,
    prompt=prompt
)
# Use the base model to label the dataset
# base_model_gpt4v.label(dir_extracted_frames, extension=".jpg", output_folder=dir_output_annotations_gpt4v, fn_post_process=post_process, shelve_file="frames_prediction.db")

# Let's train the yolo nano classification model using the samples labeled by gpt4v
# yolo_classification_model = YOLOv8("yolo11n-cls.pt")
# yolo_classification_model.train(dir_output_annotations_gpt4v, epochs=200, device=0, erasing=0, mosaic=0, crop_fraction=1.)

# run inference on the new trained model
# from ultralytics import YOLO
# yolo_classification_model = YOLO("/home/rafael_shotquality_com/code/tennis-autodistill/runs/classify/train/weights/best.pt").to("cuda")

# Let's run the model on all samples
# all_predictions = {}
# all_frames = list(dir_extracted_frames.glob("*.jpg"))
# print("🏃‍♂️ Running inference on all samples...")
# pbar = tqdm(all_frames)
# for sample in pbar:
#     pbar.set_description(f"🔥 Running model on {sample.stem}")
#     pred = yolo_classification_model.predict(sample, verbose=False)
#     all_predictions[sample] = pred[0].probs.top1
# # Distribution of our predictions
# total_predictions = len(all_predictions)
# print("\n🔍 Distribution of our predictions:")
# for i in yolo_classification_model.names.keys():
#     total_predictions_in_class = len([v for v in all_predictions.values() if v == i])
#     percentage_in_class = 100*total_predictions_in_class / total_predictions
#     print(f"* {yolo_classification_model.names[i]}: {total_predictions_in_class} ({percentage_in_class:.2f}%)")
# pickle.dump(all_predictions, open("all_predictions.pkl", "wb"))

# # Saving inference re the frames that are classified as `playable area (id=1)`
# dir_classified_frames.mkdir(parents=True, exist_ok=True)
# all_predictions = pickle.load(open("all_predictions.pkl", "rb"))
# for img_path, pred_class in all_predictions.items():
#     if pred_class != 1:
#         continue
#     # Copy the image to the output folder
#     shutil.copy(img_path, dir_classified_frames / img_path.name)



# Evaluate the model
# results = {}
# for split in ["train", "valid", "test"]:
#     results.setdefault(split, {})
#     print(f"\n🔍 Results for {split}:")
#     dir_split = dir_output_annotations_gpt4v / split
#     for target_class_dir in dir_split.glob("*"):
#         target_class_name = target_class_dir.stem
#         results[split].setdefault(target_class_name, [])

#         for img_path in target_class_dir.glob("*.jpg"):
#             pred = yolo_classification_model.predict(img_path, verbose=False)
#             class_name = yolo_classification_model.names[pred[0].probs.top1]
#             results[split][target_class_name].append(class_name == target_class_name)

#         total_correct = sum(results[split][target_class_name])
#         total_samples = len(results[split][target_class_name])
#         if total_samples == 0:
#             print(f"⚠️ No samples for {split} '{target_class_name}'")
#             continue
#         split_class_accuracy = total_correct / (total_samples + EPSILON)
#         print(f"📊 Accuracy for '{target_class_name}': {100 * split_class_accuracy:.2f}%")

#     # Accuracy for the entire split
#     flat_results = list(chain.from_iterable(results[split].values()))
#     total_correct = sum(flat_results)
#     total_samples = len(flat_results)
#     total_accuracy = total_correct / (total_samples + EPSILON)
#     print(f"✅ Final accuracy for split '{split}': {100 * total_accuracy:.2f}%")

# Show randomly SHOW_COUNT images classified as `playable area`
# lst_playable_area_paths = list(dir_classified_frames.glob("*.jpg"))
# # randomly get SHOW_COUNT images
# lst_playable_area_paths = random.sample(lst_playable_area_paths, SHOW_COUNT)
# lst_playable_area_images = [cv2.imread(path) for path in lst_playable_area_paths]
# sv.plot_images_grid(images=lst_playable_area_images, titles=lst_playable_area_paths, grid_size=SAMPLE_GRID_SIZE, size=SAMPLE_PLOT_SIZE)

##############################################
# Choosing another base model (SAM)
##############################################
dir_output_annotations_sam = Path("dataset/sam_labeled_data")
base_model_grounded_sam = GroundedSAM(
    ontology=CaptionOntology(
        {"playable area of a tennis court": "playable_area",
         "tennis player": "tennis_player"}
    ),
)
id2class = {i:class_info[1] for i, class_info in enumerate(base_model_grounded_sam.ontology.promptMap)}
# labeled_mask_dataset = base_model_gpt4v2.label(str(dir_classified_frames), extension=".jpg", output_folder=str(dir_output_annotations_sam), record_confidence=True)

# TODO REMOVE - Ja geramos antes com o .label(...)
labeled_mask_dataset = pickle.load(open("labeled_dataset.pkl", "rb"))
################

# Using supervision to visualize the labeled dataset annotated with SAM
mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Create a dictionary to map the points to their ids
points_ids2classes = {point_name:idx for idx, point_name in enumerate(COORDS_COURT_MAP)}

THRESH_PLAYABLE_AREA = 0.97
images = []
image_names = []
dict_imgs2kpts = {}
dict_xyxy_playable_area = {} # Dictionary to store the xyxy of the playable area for each image
dict_imgs2players = {}

for i, (image_path, image, annotation) in enumerate(labeled_mask_dataset):
    # if i == SHOW_COUNT:
    #     break
    if not Path(image_path).exists():
        print(f"⚠️ Image {image_path} does not exist")
        continue

    annotated_image = image.copy()
    annotated_image = mask_annotator.annotate(
        scene=annotated_image, detections=annotation)
    annotated_image = box_annotator.annotate(
        scene=annotated_image, detections=annotation)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=annotation)

    # Get image dimensions
    img_h, img_w = image.shape[:2]

    # Get all predicted classes
    predicted_classes = [id2class[idx] for idx in annotation.class_id]

    # Ignore images that do not have 2 players and 1 playable area
    if predicted_classes.count("tennis_player") != 2 or predicted_classes.count("playable_area") != 1:
        continue

    # Ignore images that the playable area is touching the borders of the image
    id_det_playable_area = predicted_classes.index("playable_area")
    xyxy_playable_area = annotation.xyxy[id_det_playable_area]
    # most left and most right points of the playable area
    x_min, y_min, x_max, y_max = xyxy_playable_area
    p_bottom_left = x_min, y_max
    p_bottom_right = x_max, y_max

    width_bbx = xyxy_playable_area[2] - xyxy_playable_area[0]
    high_bbx = xyxy_playable_area[3] - xyxy_playable_area[1]
    # Check if the playable area is touching the borders of the image (97% of the image)
    if high_bbx/img_h > THRESH_PLAYABLE_AREA or width_bbx/img_w > THRESH_PLAYABLE_AREA:
        continue

    # Get the mask of the playable area
    mask_playable_area = annotation.mask[id_det_playable_area]

    # Get the largest connected component of the playable area
    mask_uint8 = annotation.mask[id_det_playable_area].astype(np.uint8) * 255
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    # Identify the largest component (excluding background)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    # Create a mask for the largest component
    largest_mask = 1*(labels == largest_label)

    def get_upper_left_corner(mask, thresh_width=0.15):
        # Get the rows that have more than 15% of the image width
        rows_sum = np.sum(mask, axis=1)
        valid_rows = np.where(rows_sum > thresh_width*img_w)[0]
        first_row, last_row = valid_rows[0].item(), valid_rows[-1].item()
        # On the first row, get the first and last columns that has a 1
        first_row_first_col = np.where(mask[first_row] == 1)[0][0].item()
        return (first_row_first_col, first_row)

    p1 = get_upper_left_corner(largest_mask)
    # flip horizontally to get the upper right corner
    p2 = get_upper_left_corner(np.fliplr(largest_mask))
    p2 = (img_w - p2[0], p2[1])
    # flip vertically to get the lower left corner
    p3 = get_upper_left_corner(np.flipud(largest_mask), 0.30)
    p3 = (p3[0], img_h - p3[1])
    # flip horizontally and vertically to get the lower right corner
    p4 = get_upper_left_corner(np.flipud(np.fliplr(largest_mask)), 0.30)
    p4 = (img_w - p4[0], img_h - p4[1])

    # Distance from the bottom left and bottom right corners of the playable area
    # with the bounding box of the playable area
    d1 = np.linalg.norm(np.array(p3) - np.array(p_bottom_left))
    d2 = np.linalg.norm(np.array(p4) - np.array(p_bottom_right))

    # Get the bbx of the playable area
    true_coords = np.argwhere(mask_playable_area)
    min_y, min_x = true_coords.min(axis=0)  # Top-left corner (y, x)
    max_y, max_x = true_coords.max(axis=0)  # Bottom-right corner (y, x)

    # Merge the playable area mask with the players to remove the players
    masks_players = [annotation.mask[i] for i, class_name in enumerate(predicted_classes) if class_name == "tennis_player"]
    # Dilate the masks of the players
    masks_players = [cv2.dilate(mask_player.astype(np.uint8), np.ones((15, 15))) for mask_player in masks_players]
    mask_playable_area_players = mask_playable_area.copy()
    for mask_player in masks_players:
        mask_playable_area_players = np.logical_or(mask_playable_area_players, mask_player)

    # Crop the merged playable area mask with players mask to the bbx of the playable area
    mask_playable_area_players = mask_playable_area_players[min_y:max_y, min_x:max_x]
    area_mask_playable_area_players = np.sum(mask_playable_area_players)
    # Compute the theoritical area of the trapezoid (playable area)
    trapezoid_area = ((p4[0]-p3[0]) + (p2[0]-p1[0])) * (mask_playable_area_players.shape[0]) / 2
    # Check possible holes in the mask_playable_area_players
    if area_mask_playable_area_players/trapezoid_area < 0.97:
        continue

    if d1 > 15 or d2 > 15:
        continue

    # Save the bbx of the players in the dictionary
    bbx_players = [annotation.xyxy[i] for i, class_name in enumerate(predicted_classes) if class_name == "tennis_player"]
    # Make a dictionary with the player detections (player class id = 0)
    dict_imgs2players[image_path] = sv.Detections(xyxy=np.array(bbx_players), class_id=np.zeros(len(bbx_players)))

    # Draw the corners of the playable area
    cv2.circle(annotated_image, p1, 10, (0, 0, 255), -1)
    cv2.circle(annotated_image, p2, 5, (0, 0, 255), -1)
    cv2.circle(annotated_image, p3, 5, (0, 0, 255), -1)
    cv2.circle(annotated_image, p4, 5, (0, 0, 255), -1)
    cv2.imwrite(f"deleteme/mask{i}.jpg", annotated_image)
    image_names.append(Path(image_path).name)
    images.append(annotated_image)

    # Compute homography matrix that maps p1, p2, p3, p4 to COORDS_COURT_MAP
    src_pts = np.array([COORDS_COURT_MAP["p1"], COORDS_COURT_MAP["p2"], COORDS_COURT_MAP["p3"], COORDS_COURT_MAP["p4"]])
    dst_pts = np.array([p1, p2, p3, p4])
    H, _ = cv2.findHomography(src_pts, dst_pts)
    # Project the image to the court map
    tennis_court_map = cv2.imread("tennis_court_full_map.png")
    transformed_image = transform_image(tennis_court_map, H, (img_w, img_h))
    cv2.imwrite(f"deleteme/{i}_teste.jpg",transformed_image)

    # Transform the points of the court map to the image
    img_with_point = image.copy() # For drawing only
    lst_keypoints, lst_class_ids = [], []
    for point_name, point in COORDS_COURT_MAP.items():
        transformed_point = transform_point(H, point)
        transformed_point = tuple(map(int, transformed_point))
        lst_keypoints.append(transformed_point)
        lst_class_ids.append(points_ids2classes[point_name])
        #### DRAWING THE POINTS ON THE IMAGE
        # add an offset to the text to make it more readable
        img_with_point = cv2.circle(img_with_point, transformed_point, 5, (0, 0, 255), -1)
        offset = (10, -10)
        transformed_point_with_offset = (transformed_point[0]+offset[0], transformed_point[1]+offset[1])
        img_with_point = cv2.putText(img_with_point, point_name, transformed_point_with_offset, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    # Transform the list of keypoints to a (n,m,2) array
    n = len(lst_keypoints)  # Number of detected objects
    m = 1  # Assuming each object has one set of keypoints
    xy = np.array(lst_keypoints).reshape(n, m, 2)
    confidences = np.ones((n, m))
    class_ids = np.array(lst_class_ids)
    dict_imgs2kpts[image_path] = sv.KeyPoints(xy, class_ids, confidences)
    cv2.imwrite(f"deleteme/{i}_points.jpg", img_with_point)

    dict_xyxy_playable_area[image_path] = tuple(map(float, xyxy_playable_area))

    # AQUI TODO TIRAR
    # if len(dict_imgs2kpts) > 10:
    #     break
    #####

####################################################
# Player detector                                  #
####################################################
dir_output_players = Path("dataset/players")
dataset_players = sv.DetectionDataset(["player"], images=list(dict_imgs2players.keys()), annotations=dict_imgs2players)
# Split dataset into training and validation and save them
ds_train, ds_val = dataset_players.split(split_ratio=0.8)
dir_images_train = dir_output_players / "train" / "images"
dir_labels_train = dir_output_players / "train" / "labels"
ds_train.as_yolo(images_directory_path = dir_images_train, annotations_directory_path = dir_labels_train)
dir_images_val = dir_output_players / "val" / "images"
dir_labels_val = dir_output_players / "val" / "labels"
ds_val.as_yolo(images_directory_path = dir_images_val, annotations_directory_path = dir_labels_val)
data_yaml_path = dir_output_players / "data.yaml"
# Save the data.yaml file
data = {"path": str(dir_output_players.resolve()),
        "train": "train",
        "val": "val",
        "names": dataset_players.classes}
save_yaml_file(data=data, file_path=data_yaml_path)
# Train the player detector
yolo_player_model = YOLOv8("yolo11n.pt")
yolo_player_model.train(data_yaml_path, epochs=200, device=0)

####################################################
# Key points detector                              #
####################################################
# Creating the dataset for the keypoints
dir_output_keypoints = Path("dataset/court_keypoints")
dataset_keypoints = sv.DetectionDataset(list(dict_imgs2kpts.keys()), list(dict_imgs2kpts.keys()), dict_imgs2kpts)
dataset_keypoints_to_yolo(
    dataset_keypoints=dataset_keypoints,
    dict_keypoints_classes=points_ids2classes,
    dataset_bbxes=dict_xyxy_playable_area,
    dir_output=dir_output_keypoints,

)

yolo_keypoint_model = YOLOv8("yolo11n-pose.pt")
yolo_keypoint_model.train(dir_output_keypoints / "data.yaml", epochs=200, device=0)#, erasing=0, mosaic=0, crop_fraction=1.)
sv.plot_images_grid(
    images=images,
    titles=image_names,
    grid_size=SAMPLE_GRID_SIZE,
    size=SAMPLE_PLOT_SIZE)

# AQUI U
a = 123