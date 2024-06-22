document.querySelector('form').addEventListener('submit', function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('result').textContent = 'Error: ' + data.error;
            document.getElementById('result').style.color = 'red';
        } else {
            document.getElementById('result').textContent = 'Image is a : ' + data.class_name;
            document.getElementById('result').style.border = '1px solid #b3d4fc';
            document.getElementById('result').style.color = '#31708f';
            
            // Display the uploaded image
            const uploadedImage = document.getElementById('uploaded-image');
            uploadedImage.src = data.file_url;
            uploadedImage.style.display = 'block';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').textContent = 'Error: ' + error;
        document.getElementById('result').style.color = 'red';
    });
});
