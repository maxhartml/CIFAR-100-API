# CIFAR-100 Image Classifier Web Application

A modern and responsive web application for classifying images from the **CIFAR-100 dataset** using a pre-trained deep learning model. This project combines a FastAPI backend with a user-friendly HTML/CSS/JavaScript interface for seamless image classification.

---

## 🚀 Features

- **Accurate Classification**: Utilizes a custom ResNet-18 model trained on the CIFAR-100 dataset.
- **Top-3 Predictions**: Displays the top-3 class labels with confidence percentages.
- **Real-time Image Preview**: Upload and preview the image in the interface.
- **Fully Responsive Design**: Modern, mobile-friendly UI with a dark theme.
- **Error Handling**: Detailed error messages for unsupported image types or server issues.
- **Logging**: Comprehensive server logs for debugging and monitoring.

---

## 📋 Requirements

### Backend Dependencies

- Python 3.8 or higher
- Required Python libraries (in `requirements.txt`):
  - `fastapi`
  - `uvicorn`
  - `torch`
  - `torchvision`
  - `Pillow`
  - `jinja2`

Install dependencies with:
```bash
pip install -r requirements.txt
```

### Frontend

- HTML5, CSS3, and Vanilla JavaScript (no frameworks required).

---

## 🛠️ Project Structure

```plaintext
CIFAR-100-API/
├── app/
│   ├── main.py               # FastAPI app entry point
│   ├── model.py              # Model loading and prediction logic
├── custom_models/
│   ├── cifar100_resnet.py    # Custom ResNet18 model
├── templates/
│   ├── index.html            # Frontend HTML file
├── static/
│   ├── css/
│   │   ├── styles.css        # Styling for the web interface
│   ├── js/
│   │   ├── main.js           # JavaScript for frontend interactivity
├── checkpoints/
│   ├── best_model.pth        # Pre-trained model weights
├── app.log                   # Server log file
├── requirements.txt          # Backend dependencies
├── README.md                 # Project documentation
```

---

## ⚙️ Usage

1. **Clone the Repository**
    ```bash
    git clone https://github.com/maxhartml/CIFAR-100-API.git
    cd CIFAR-100-API
    ```

2. **Run the Application**
    ```bash
    uvicorn app.main:app --reload
    ```

3. **Open in Browser**
    - Visit the app at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🖼️ How to Use the Web App

1. **Upload an Image**: Click on “Choose Image” and upload an image from your device.
2. **View Prediction**: Click “Classify” to get the top-3 predicted classes and their confidence scores.
3. **Error Handling**: If an invalid image is uploaded, you’ll see an error message.

---

## 🧠 Model Details

- **Architecture**: Custom ResNet-18.
- **Dataset**: CIFAR-100 (100 object classes with 40,000 training, 10,000 validation, and 10,000 testing images).
- **Preprocessing**:
  - Resizing to 32x32 pixels.
  - Normalization with mean (0.5, 0.5, 0.5) and standard deviation (0.5, 0.5, 0.5).

---

## 🌟 Features in Progress

- Deployment to cloud platforms (e.g., AWS, Azure, GCP).
- Docker containerization for easier deployment.
- HTTPS and domain name integration for secure public access.
- Caching for faster predictions.

---

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 🙌 Acknowledgements

- **CIFAR-100 Dataset**: Created by the Canadian Institute for Advanced Research.
- **FastAPI**: For the easy-to-use and performant web framework.
- **PyTorch**: For building and training the deep learning model.
- **Vanilla JavaScript**: For keeping the frontend lightweight and responsive.

---

## 📧 Contact

For questions, suggestions, or collaboration, feel free to reach out:

**Max Hart**
- Email: maxhart.ml.ai@gmail.com
- GitHub: [https://github.com/maxhartml](https://github.com/maxhartml)