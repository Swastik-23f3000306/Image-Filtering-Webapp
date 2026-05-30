from flask import Flask, render_template, request
import cv2
import os
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':

        file = request.files['image']

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Read image in grayscale
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        # Mean Filter
        mean = cv2.blur(img, (5,5))

        # Gaussian Filter
        gaussian = cv2.GaussianBlur(img, (5,5), 0)

        # Laplacian Filter
        laplacian = cv2.Laplacian(img, cv2.CV_64F)

        # Convert Laplacian to uint8
        laplacian = np.uint8(np.absolute(laplacian))

        # Output paths
        mean_path = os.path.join(OUTPUT_FOLDER, 'mean.jpg')
        gaussian_path = os.path.join(OUTPUT_FOLDER, 'gaussian.jpg')
        laplacian_path = os.path.join(OUTPUT_FOLDER, 'laplacian.jpg')

        # Save outputs
        cv2.imwrite(mean_path, mean)
        cv2.imwrite(gaussian_path, gaussian)
        cv2.imwrite(laplacian_path, laplacian)

        return render_template(
            'index.html',
            original=filepath,
            mean=mean_path,
            gaussian=gaussian_path,
            laplacian=laplacian_path
        )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)