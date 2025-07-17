from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load model once at startup
model = tf.keras.models.load_model('mnist_model.h5')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        index = int(request.form['index'])
        if index < 1 or index > 60000:
            return jsonify({'error': 'Index must be between 1 and 60000'})

        # Adjust for 0-based indexing
        img = x_train[index - 1]
        img_resized = img.reshape(1, 28, 28, 1) / 255.0
        prediction = model.predict(img_resized)[0]
        predicted_digit = int(np.argmax(prediction))
        confidence = round(np.max(prediction) * 100, 2)

        # Convert image to base64 to display in browser
        image_pil = Image.fromarray(img)
        buffered = io.BytesIO()
        image_pil = image_pil.resize((150, 150))  # Make it more visible
        image_pil.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({
            'digit': predicted_digit,
            'confidence': confidence,
            'image': image_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
