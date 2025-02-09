#!/bin/bash

# Default path to video links
default_file_path="dataset/video_links.txt"
# Default destination folder
default_destination_folder="dataset/videos/"

# Assign the file path and destination folder
file_path="$default_file_path"
destination_folder="$default_destination_folder"

# Create the destination folder if it doesn't exist
mkdir -p "$destination_folder"

# Initialize variables
name=""

# Read the file line by line
while IFS= read -r line
do
    # Trim leading and trailing whitespace
    line=$(echo "$line" | xargs)

    # Check if the line starts with "name:"
    if [[ "$line" == name:* ]]; then
        # Extract the name value
        name=$(echo "$line" | cut -d' ' -f2)
    fi

    # Check if the line starts with "url:"
    if [[ "$line" == url:* ]]; then
        # Extract the URL
        url=$(echo "$line" | cut -d' ' -f2)

        # Check if the URL is a YouTube URL
        if [[ "$url" == https://www.youtube.com* ]]; then
            echo "Downloading $url as $name"
            # Download using the name for the output file
            yt-dlp -f "bestvideo[height<=720]+bestaudio/best[height<=720]" --recode-video mp4 -o "$destination_folder/$name.%(ext)s" "$url"
        fi
    fi
done < "$file_path"

