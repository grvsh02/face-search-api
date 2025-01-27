import torch
import torchvision.models as models  # Replace with your specific model if needed
import os

def convert_pytorch_to_onnx(pytorch_model_path, onnx_model_path, input_size):
    # Check if the PyTorch model file exists
    if not os.path.exists(pytorch_model_path):
        raise FileNotFoundError(f"The PyTorch model file {pytorch_model_path} does not exist.")

    # Load the PyTorch model
    model = torch.load(pytorch_model_path, map_location=torch.device('cpu'))
    model.eval()  # Set the model to evaluation mode

    # Generate dummy input for the model with the specified input size
    dummy_input = torch.randn(1, *input_size)

    # Export the model to ONNX format
    torch.onnx.export(
        model,                           # The model to be converted
        dummy_input,                     # A dummy input for tracing
        onnx_model_path,                 # The output path for the ONNX model
        export_params=True,              # Store the trained parameter weights inside the model file
        opset_version=11,                # The ONNX opset version to export to
        do_constant_folding=True,        # Optimize constant folding for ONNX
        input_names=['input'],           # Name of the input layer(s)
        output_names=['output'],         # Name of the output layer(s)
        dynamic_axes={                   # Specify axes that can vary for dynamic batching
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Model has been successfully converted to ONNX and saved at {onnx_model_path}")

if __name__ == "__main__":
    # Path to the local PyTorch model file
    pytorch_model_path = "model.pt"  # Replace with your model file

    # Path to save the ONNX model
    onnx_model_path = "model.onnx"

    # Input size for the model (C, H, W) - Update this as per your model's requirement
    input_size = (3, 112, 112)  # Example for an image model

    # Convert the PyTorch model to ONNX
    convert_pytorch_to_onnx(pytorch_model_path, onnx_model_path, input_size)
