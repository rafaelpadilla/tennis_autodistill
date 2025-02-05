from tqdm import tqdm
from pathlib import Path
import supervision as sv

# Function to split video into frames and save them to a folder
def split_video_to_frames(video_path: str, output_folder: str, frames_per_second: int = 1, start: int = 0, end: int = None, overwrite: bool = False):
    """
    Splits a video into frames and saves them to the specified folder.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the folder where frames will be saved.
        frames_per_second (int): Number of frames to save per second of video. Default is 1.
        start (int): Starting position from which video should generate frames. Default is 0.
        end (Optional[int]): Ending position at which video should stop generating frames. If None, reads to the end. Default is None.
        overwrite (bool): If True, overwrite existing frames. Default is False.
    """
    video_path = Path(video_path)
    output_folder = Path(output_folder)

    # Ensure the output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)
    # Ensure the video path exists
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_info = sv.VideoInfo.from_video_path(video_path)

    # Calculate the stride based on the frame rate
    stride = video_info.fps // frames_per_second

    # Get video name - used for naming the frames
    video_name = video_path.stem

    # Get video frames generator
    frames_generator = sv.get_video_frames_generator(
        source_path=video_path,
        stride=stride,
        start=start,
        end=end
    )
    # Calculate the total number of frames to save to inform in the progress bar
    total_frames = round(video_info.total_frames / stride)

    # Extract and save frames
    total_skipped, total_saved = 0, 0
    with sv.ImageSink(target_dir_path=output_folder) as sink:
        for image in tqdm(frames_generator, desc=f"Extracting frames ({video_path.name})", total=total_frames):
            image_name = f"{video_name}-frame_{sink.image_count}.jpg"
            if (output_folder / image_name).exists() and not overwrite:
                total_skipped += 1
                continue
            sink.save_image(image=image, image_name=image_name)
            total_saved += 1

    print(f"Completed extraction. Total frames saved: {total_saved}. Total frames skipped: {total_skipped}")
    print(f"✅ Completed extraction!")
    print(f"🎉 Total frames saved: {total_saved}.")
    print(f"🚫 Total frames skipped: {total_skipped}.")



