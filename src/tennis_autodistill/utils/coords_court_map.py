import cv2

COORDS_COURT_MAP = {
    # Points on the corners of the court (clockwise)
    "p1": (370, 646),
    "p2": (1489, 646),
    "p3": (370, 3040),
    "p4": (1489, 3040),
    # Points on the upper part of the court
    "p5": (513, 646),
    "p6": (1345, 646),
    "p7": (513, 1199),
    "p8": (929, 1199),
    "p9": (1345, 1199),
    # Points on the lower part of the court
    "p10": (513, 2485),
    "p11": (929, 2485),
    "p12": (1345, 2485),
    "p13": (513, 3040),
    "p14": (1345, 3040),
}


def plot_coords_on_image(coords_map, image_path, output_path):
    """
    Plot the coordinates on the image and save it to the output path.
    """
    # Open the image using OpenCV
    img = cv2.imread(image_path)

    # Set the color to red
    color = (0, 0, 255)

    # Plot each point with the red color and larger label
    for key, (x, y) in coords_map.items():
        cv2.circle(img, (x, y), 20, color, -1)  # Draw the point
        # cv2.putText(img, key, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 3.5, color, 4, cv2.LINE_AA)  # Add the label with larger font size and offset

    # Save the image to the specified output path
    cv2.imwrite(output_path, img)

# Example usage
plot_coords_on_image(COORDS_COURT_MAP, '/home/rafael_shotquality_com/code/tennis-autodistill/tennis_court_full_map.png', '/home/rafael_shotquality_com/code/tennis-autodistill/tennis_court_full_map_with_points.png')

