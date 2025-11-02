const predictButton = document.getElementById('predictButton');
const imageUpload = document.getElementById('imageUpload');
const resultText = document.getElementById('predictionText');
const preview = document.getElementById('preview');
const loader = document.getElementById('loader');

predictButton.addEventListener('click', async () => {
    if (imageUpload.files.length === 0) {
        alert('Please select an image file first.');
        return;
    }

    const file = imageUpload.files[0];
    const formData = new FormData();
    formData.append('file', file);

    resultText.textContent = '';
    loader.classList.remove('hidden');

    try {
        // This 'fetch' call is the magic!
        // It calls the API service at http://localhost:8000
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.statusText}`);
        }

        const data = await response.json();
        resultText.textContent = `Prediction: ${data.prediction} (${(data.confidence * 100).toFixed(2)}%)`;

    } catch (error) {
        console.error('Error:', error);
        resultText.textContent = 'Failed to classify. Is the API running?';
    } finally {
        loader.classList.add('hidden');
    }
});

// Show image preview
imageUpload.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            preview.style.display = 'block';
        };
        reader.readAsDataURL(file);
        resultText.textContent = ''; // Clear previous prediction
    }
});