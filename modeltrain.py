import os
from ultralytics import YOLO
from roboflow import Roboflow
import torch

# --- 1. Roboflow Configuration ---
# Your specific project details are now included here.
ROBOFLOW_API_KEY = "4VqXS6lsqkS3A8q6RU1O"
WORKSPACE_ID = "college-snpvc"
PROJECT_ID = "brain-tumour-gdfp4-v45eq"
VERSION_NUMBER = 1
MODEL_FORMAT = "yolov8-obb" # OBB for Oriented Bounding Boxes

# --- 2. Verify GPU Availability ---
def check_gpu():
    """Checks if a CUDA-enabled GPU is available for training."""
    if not torch.cuda.is_available():
        print("🔴 WARNING: No CUDA-enabled GPU detected!")
        print("Training will proceed on the CPU, which will be very slow.")
        print("Please ensure you have a compatible NVIDIA GPU and have installed PyTorch with CUDA support.")
        return False
    else:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU Detected: {gpu_name}")
        return True

# --- 3. Download Dataset ---
def download_dataset():
    """Downloads the specified dataset from Roboflow."""
    print("Downloading dataset from Roboflow...")
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
        dataset = project.version(VERSION_NUMBER).download(MODEL_FORMAT)
        print(f"Dataset downloaded successfully to: {dataset.location}")
        return dataset
    except Exception as e:
        print(f"❌ Error downloading dataset from Roboflow: {e}")
        print("Please double-check your API key, workspace, project, and version details.")
        return None

# --- 4. Train the YOLOv8 Model ---
def train_model(dataset):
    """
    Loads a pretrained YOLOv8-OBB model and trains it on the downloaded dataset using the GPU.
    """
    if dataset is None:
        print("Cannot start training without a dataset.")
        return

    # Load the pretrained model for Oriented Bounding Boxes (yolov8n-obb.pt)
    # This is crucial for matching your dataset's format.
    model = YOLO('yolov8n-obb.pt')
    print("Pretrained YOLOv8-OBB model loaded.")

    # Path to the data.yaml file from your Roboflow download
    data_yaml_path = os.path.join(dataset.location, 'data.yaml')

    print("\nStarting model training on GPU... This may take a while.")
    
    # Train the model
    # device=0 tells YOLO to use the first available GPU.
    results = model.train(
        data=data_yaml_path,
        imgsz=640,
        epochs=100,
        batch=8,
        device=0, # Crucial for GPU training!
        name='yolov8n_obb_brain_tumour_run' # A unique name for this training run
    )

    print("\n✅ Training complete!")
    print("Your trained model can be found in the 'runs/obb/train/' directory.")
    print("Look for a file named 'best.pt' in the 'weights' subfolder of your latest run.")

if __name__ == '__main__':
    if check_gpu():
        dataset = download_dataset()
        train_model(dataset)
