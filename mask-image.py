import cv2
import numpy as np
import mediapipe as mp
import os

# Initialize MediaPipe solutions
mp_selfie_segmentation = mp.solutions.selfie_segmentation

def create_mask(image_path, output_path):
    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    # Resize the image for consistent processing
    input_image = cv2.resize(image, (640, 640))

    # Convert BGR image to RGB
    rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Process the image using MediaPipe Selfie Segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        result = selfie_segmentation.process(rgb_image)

        # Create a binary mask where 1 represents the segmented area
        mask = result.segmentation_mask

        # Convert the mask to binary (0 or 255)
        binary_mask = (mask > 0.5).astype(np.uint8) * 255

        # Convert binary mask to 3-channel (for saving as an image)
        binary_mask_3ch = cv2.merge([binary_mask, binary_mask, binary_mask])

        # Save the result
        cv2.imwrite(output_path, binary_mask_3ch)

    print(f"Mask saved to: {output_path}")


def process_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Check if it's an image file
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Construct output path
            output_path = os.path.join(output_folder, f"mask_{filename}")

            # Process the image
            create_mask(input_path, output_path)
        else:
            print(f"Skipping non-image file: {filename}")


# Example usage
input_folder_path = "test\input"  # Replace with your input folder path
output_folder_path = "test\output"  # Replace with your desired output folder path

process_folder(input_folder_path, output_folder_path)
