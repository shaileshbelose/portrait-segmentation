import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe solutions
mp_selfie_segmentation = mp.solutions.selfie_segmentation

def create_mask(image_path, output_path):
    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
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

# Example usage
input_image_path = "test/s5.jpeg"  # Replace with your image path
output_mask_path = "test/_s5selfie_mask.jpeg"  # Output path for the mask

create_mask(input_image_path, output_mask_path)
