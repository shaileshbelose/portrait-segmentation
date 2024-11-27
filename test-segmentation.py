import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import load_img, img_to_array

# Load the trained model
model = load_model('checkpoints/bilinear_segmodel-07-0.20.keras')  # Replace with your actual model file path

# Function to preprocess the input image
def preprocess_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)  # Resize image
    img_array = img_to_array(img) / 255.0               # Normalize to 0-1
    return np.expand_dims(img_array, axis=0)            # Add batch dimension

# Function to test the model on a sample image
def test_segmentation(image_path):
    # Preprocess the image
    test_image = preprocess_image(image_path)
    
    # Predict the mask
    predicted_mask = model.predict(test_image)
    
    # Convert the mask to binary format
    binary_mask = (predicted_mask[0, :, :, 0] > 0.5).astype(np.uint8)  # Threshold at 0.5
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(test_image[0])  # Remove batch dimension
    plt.title("Original Image")
    
    plt.subplot(1, 3, 2)
    plt.imshow(predicted_mask[0, :, :, 0], cmap='gray')  # Predicted mask
    plt.title("Predicted Mask")
    
    plt.subplot(1, 3, 3)
    overlay = test_image[0].copy()
    overlay[:, :, 0] = binary_mask  # Assuming red channel for mask overlay
    plt.imshow(overlay)
    plt.title("Overlay")
    
    plt.tight_layout()
    plt.show()

# Test the model with a sample image
test_segmentation('E:/tech5/AIML/src/portrait_segmentation/Portrait-Segmentation-master/Portrait-Segmentation-master/test/s2.jpeg')  # Replace with your test image path

test_segmentation('E:/tech5/AIML/src/portrait_segmentation/Portrait-Segmentation-master/Portrait-Segmentation-master/test/s3.jpeg')  # Replace with your test image path


test_segmentation('E:/tech5/AIML/src/portrait_segmentation/Portrait-Segmentation-master/Portrait-Segmentation-master/test/s4.jpeg')  # Replace with your test image path


