import logging
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.model import load_model, predict

# Set up logging configuration
logging.basicConfig(
    filename="app.log",  # File to store logs
    level=logging.INFO,  # Capture INFO-level and above logs
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format: timestamp, level, message
)

# Initialize the FastAPI application
app = FastAPI(
    title="CIFAR-100 Image Classification API",
    description="A web app for CIFAR-100 image classification.",
    version="1.0.0",
)

# Configure template and static directories
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Dynamically load the model during application startup
MODEL_CONFIG = {
    "type": "CustomResNet18",  # Change this to "CustomResNet18"/"CNN" to use the ResNet/CNN models
    "checkpoint_path": "CustomResNet18_20250125-165329.pth"  # Update this to your checkpoint file
}

try:
    model = load_model(MODEL_CONFIG["type"], MODEL_CONFIG["checkpoint_path"])
    logging.info(f"Model ({MODEL_CONFIG['type']}) loaded successfully from {MODEL_CONFIG['checkpoint_path']}")
except Exception as e:
    logging.error(f"Failed to load the model: {str(e)}")
    raise RuntimeError("Model loading failed. Check logs for details.")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    Serve the main web page of the application.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def classify_image(file: UploadFile = File(...)):
    """
    Handle image upload and return the classification result.

    Args:
        file (UploadFile): Uploaded image file.

    Returns:
        JSONResponse: Predicted class labels and confidence scores.
    """
    try:
        # Validate the uploaded file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

        # Read the image file
        image_bytes = await file.read()

        # Perform model prediction
        prediction = predict(model, image_bytes)

        # Log the prediction details
        logging.info(f"Prediction successful: {prediction}")
        return prediction

    except HTTPException as e:
        # Handle and log client-side errors
        logging.warning(f"Client error: {str(e)}")
        return JSONResponse(status_code=e.status_code, content={"message": e.detail})

    except Exception as e:
        # Handle and log unexpected server-side errors
        logging.error(f"Prediction failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": "An error occurred during prediction. Please try again later."}
        )