from flask import Flask, render_template, request, url_for
import cv2
import os
import numpy as np
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():

    original = None
    mean = None
    gaussian = None
    laplacian = None

    if request.method == 'POST':

        file = request.files['image']

        if file:

            filename = str(int(time.time())) + ".jpg"

            filepath = os.path.join(UPLOAD_FOLDER, filename)

            file.save(filepath)

            # Read grayscale image
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            # Filters
            mean_img = cv2.blur(img, (5,5))

            gaussian_img = cv2.GaussianBlur(img, (5,5), 0)

            laplacian_img = cv2.Laplacian(img, cv2.CV_64F)
            laplacian_img = np.uint8(np.absolute(laplacian_img))

            # Output names
            mean_filename = "mean_" + filename
            gaussian_filename = "gaussian_" + filename
            laplacian_filename = "laplacian_" + filename

            # Save outputs
            cv2.imwrite(
                os.path.join(OUTPUT_FOLDER, mean_filename),
                mean_img
            )

            cv2.imwrite(
                os.path.join(OUTPUT_FOLDER, gaussian_filename),
                gaussian_img
            )

            cv2.imwrite(
                os.path.join(OUTPUT_FOLDER, laplacian_filename),
                laplacian_img
            )

            # URL paths
            original = url_for(
                'static',
                filename='uploads/' + filename
            )

            mean = url_for(
                'static',
                filename='outputs/' + mean_filename
            )

            gaussian = url_for(
                'static',
                filename='outputs/' + gaussian_filename
            )

            laplacian = url_for(
                'static',
                filename='outputs/' + laplacian_filename
            )

    return render_template(
        'index.html',
        original=original,
        mean=mean,
        gaussian=gaussian,
        laplacian=laplacian
    )

if __name__ == '__main__':
    app.run(debug=True)