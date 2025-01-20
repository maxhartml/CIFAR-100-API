// Get references to elements in the DOM
const form = document.getElementById("upload-form");
const resultDiv = document.getElementById("result");
const imagePreview = document.getElementById("image-preview");
const spinner = document.getElementById("loading-spinner");
const fileInput = document.getElementById("file");

// Display the uploaded image as a preview
fileInput.onchange = (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = () => {
            imagePreview.src = reader.result; // Set the image preview source
            imagePreview.style.display = "block"; // Show the preview
        };
        reader.readAsDataURL(file); // Read the file as a Data URL
    }
};

// Handle form submission for image classification
form.onsubmit = async (e) => {
    e.preventDefault(); // Prevent default form submission behavior
    spinner.style.display = "block"; // Show the loading spinner
    resultDiv.innerHTML = ""; // Clear previous results

    const formData = new FormData(form); // Collect form data

    try {
        // Send POST request to the backend API
        const response = await fetch("/predict/", {
            method: "POST",
            body: formData,
        });

        // Handle response errors
        if (!response.ok) {
            throw new Error("Failed to classify the image. Please try again.");
        }

        // Parse JSON response from the server
        const data = await response.json();

        // Display predictions
        resultDiv.innerHTML = `<h2>Top Predictions</h2>`;
        data.predictions.forEach((pred) => {
            resultDiv.innerHTML += `<p><strong>${pred.class}:</strong> ${(pred.confidence * 100).toFixed(2)}%</p>`;
        });
    } catch (error) {
        // Display error messages in the result container
        resultDiv.innerHTML = `<p style="color: red;">${error.message}</p>`;
    } finally {
        spinner.style.display = "none"; // Hide the spinner after completion
    }
};