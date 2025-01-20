import torch
from torchvision import transforms
from PIL import Image
import io
from models.cifar100_resnet import CustomResNet18

def load_model(model_path):
    """
    Load the pre-trained PyTorch model for CIFAR-100.

    Returns:
        model (torch.nn.Module): Loaded model.
    """
    model_path = f"./checkpoints/{model_path}" 

    model = CustomResNet18() # Initialize the model
    model.fc = torch.nn.Linear(model.fc.in_features, 100)  # Change the output layer to have 100 classes

    state_dict = torch.load(model_path, map_location="cpu") 
    model.load_state_dict(state_dict)  # Load the model weights

    model.eval()  # Set the model to evaluation mode
    return model

def predict(model, image_bytes):
    """
    Predict the class of an image using the loaded model.

    Args:
        model (torch.nn.Module): The trained model.
        image_bytes (bytes): The raw image file in bytes.

    Returns:
        dict: Predicted class and confidence.
    """
    
    # Define the same transformations as during training
    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR-100 images are 32x32
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    # Load the image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        output = model(input_tensor)
        prob, predicted = torch.max(output, 1)  # Get the class index with the highest score

    # Convert to a readable format
    return {"class": predicted.item(), "confidence": prob.item()}