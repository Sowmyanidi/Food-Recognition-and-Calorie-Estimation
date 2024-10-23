const video = document.getElementById('video');
const captureBtn = document.getElementById('captureBtn');
const canvas = document.getElementById('canvas');
const imageInput = document.getElementById('imageInput');

// Start the camera
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (error) {
        console.error('Error accessing camera:', error);
    }
}

// Capture image on button click
captureBtn.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);

    // Convert the canvas to a Blob (binary image)
    canvas.toBlob((blob) => {
        const file = new File([blob], 'captured_image.png', { type: 'image/png' });

        // Update the hidden file input with the captured image
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        imageInput.files = dataTransfer.files;

        alert('Image captured! You can now upload it.');
    }, 'image/png');
});

// Initialize the camera
startCamera();