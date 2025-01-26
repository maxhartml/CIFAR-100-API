// Get references to elements
const form = document.getElementById("upload-form");
const resultDiv = document.getElementById("result");
const imagePreview = document.getElementById("image-preview");
const spinner = document.getElementById("loading-spinner");
const progressBar = document.getElementById("progress-bar");
const progress = document.querySelector(".progress");
const fileInput = document.getElementById("file");

// Show image preview
fileInput.onchange = (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = () => {
            imagePreview.src = reader.result;
            imagePreview.style.display = "block";
        };
        reader.readAsDataURL(file);
    }
};

// Handle form submission
form.onsubmit = async (e) => {
    e.preventDefault();
    spinner.style.display = "block";
    progressBar.style.display = "block";
    resultDiv.innerHTML = "";
    progress.style.width = "0%";

    const formData = new FormData(form);

    try {
        // Simulate progress bar
        for (let i = 0; i <= 100; i += 10) {
            await new Promise((resolve) => setTimeout(resolve, 50));
            progress.style.width = `${i}%`;
        }

        const response = await fetch("/predict/", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error("Failed to classify the image. Please try again.");
        }

        const data = await response.json();

        // Display results
        resultDiv.innerHTML = `<h3>Top Predictions</h3>`;
        data.predictions.forEach((pred) => {
            resultDiv.innerHTML += `<p><strong>${pred.class}:</strong> ${(pred.confidence * 100).toFixed(2)}%</p>`;
        });
    } catch (error) {
        resultDiv.innerHTML = `<p style="color: red;">${error.message}</p>`;
    } finally {
        spinner.style.display = "none";
        progressBar.style.display = "none";
    }
};