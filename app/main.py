from fastapi import FastAPI, File, UploadFile
from app.model import load_model, predict
from fastapi.responses import JSONResponse

# Initialize FastAPI app
app = FastAPI(
    title="CIFAR-100 Image Classification API",
    description="An API for classifying CIFAR-100 images using a pre-trained PyTorch model.",
    version="1.0.0",
)

model_path = "run_20250120-193831_model.pth"

# Load the model when the API starts
model = load_model(model_path)

@app.get("/")
async def root():
    state_dict_keys = list(model.state_dict().keys())
    return {"message": "Welcome to the CIFAR-100 Image Classification API", "state_dict_keys": state_dict_keys}

@app.post("/predict/")
async def classify_image(file: UploadFile = File(...)):
    """
    Endpoint to classify an uploaded image.

    Args:
        file (UploadFile): Image file uploaded by the user.

    Returns:
        JSONResponse: Predicted class and confidence.
    """
    # Read the image file
    image_bytes = await file.read()
    
    # Perform the prediction
    prediction = predict(model, image_bytes)
    
    return JSONResponse(prediction)