import torch
from torchvision import transforms
from PIL import Image
import io
from models.custom_CNN import CNN
from models.custom_ResNet import CustomResNet18

def load_model(model_type, checkpoint_path):
    """
    Load the pre-trained PyTorch model based on the specified model type.

    Args:
        model_type (str): Type of the model to load ("CNN" or "CustomResNet18").
        checkpoint_path (str): Path to the model checkpoint.

    Returns:
        torch.nn.Module: Loaded PyTorch model.
    """
    checkpoint_path = f"./checkpoints/{checkpoint_path}"

    # Initialize the specified model
    if model_type == "CNN":
        model = CNN()
    elif model_type == "CustomResNet18":
        model = CustomResNet18()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Load the model weights
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()
    return model

def predict(model, image_bytes):
    """
    Predict the top-3 classes for a given image using the trained model.

    Args:
        model (torch.nn.Module): Loaded model for prediction.
        image_bytes (bytes): Raw image file in bytes.

    Returns:
        dict: Top-3 predicted class labels and confidence scores.
    """
    # Define preprocessing transformations (should match training)
    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to CIFAR-100 image dimensions
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # Normalize as per training
    ])

    # Load and preprocess the image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top_probs, top_classes = torch.topk(probabilities, 3)  # Get top 3 predictions

    # Map class indices to labels
    result = [
        {"class": CIFAR100_CLASSES[class_idx], "confidence": prob.item()}
        for prob, class_idx in zip(top_probs[0], top_classes[0])
    ]
    return {"predictions": result}

# CIFAR-100 class labels (static list)
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]