import time
import tritonclient.http as httpclient
import numpy as np
import cv2

# Triton Server and Model Configuration
TRITON_SERVER_URL = "localhost:8000"  # Update with your Triton server address
MODEL_NAME = "arcface"
MODEL_VERSION = "1"

def preprocess_image(image_path, target_shape=(112, 112)):
    """
    Preprocess the image to match the input requirements of the model.
    - Resize the image.
    - Normalize pixel values.
    - Convert to channels-first format.

    Args:
        image_path (str): Path to the input image.
        target_shape (tuple): Target shape of the input image (H, W).

    Returns:
        np.ndarray: Preprocessed image array.
    """
    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    
    # Resize to target shape
    image = cv2.resize(image, target_shape)
    
    # Normalize to [0, 1] and transpose to channels-first format
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    
    # Add batch dimension to match model input shape
    return np.expand_dims(image, axis=0)

def infer(image_array):
    """
    Perform inference on the Triton server with the given image array.

    Args:
        image_array (np.ndarray): Preprocessed image array.

    Returns:
        np.ndarray: Embeddings output from the model.
    """
    # Initialize Triton HTTP client
    client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)
    
    # Check if the server and model are ready
    if not client.is_server_live():
        raise RuntimeError("Triton server is not live.")
    if not client.is_model_ready(MODEL_NAME, MODEL_VERSION):
        raise RuntimeError(f"Model {MODEL_NAME} version {MODEL_VERSION} is not ready.")
    
    # Define input and output
    inputs = [
        httpclient.InferInput("data", image_array.shape, "FP32")
    ]
    inputs[0].set_data_from_numpy(image_array)

    outputs = [
        httpclient.InferRequestedOutput("fc1")
    ]

    # Perform inference
    response = client.infer(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        inputs=inputs,
        outputs=outputs
    )

    # Extract embeddings
    embeddings = response.as_numpy("fc1")
    return embeddings

if __name__ == "__main__":
    # Path to input image
    IMAGE_PATH = "download.jpeg"  # Update with the path to your test image

    # Preprocess the image
    image_array = preprocess_image(IMAGE_PATH)

    # Perform inference and get embeddings
    start_time = time.time()
    embeddings = infer(image_array)
    print(f"Inference time: {time.time() - start_time:.4f} seconds")
    print("Embeddings shape:", embeddings.shape)
    print("Embeddings:", embeddings)
