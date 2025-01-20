# CIFAR-100 Image Classification API

This project is a web API for classifying images from the CIFAR-100 dataset using a pre-trained PyTorch model. The API is built with FastAPI.

## Features

- Classify images into one of the 100 CIFAR-100 classes.
- Upload images through an HTTP POST request.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/CIFAR-100-API.git
    cd CIFAR-100-API
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Start the FastAPI server:
    ```bash
    uvicorn app.main:app --reload
    ```

4. Open your browser and navigate to `http://127.0.0.1:8000/docs` to test the API.

## Usage

- Use the `/predict/` endpoint to upload an image and get the predicted class.
- The `/` endpoint provides a welcome message.

## Folder Structure