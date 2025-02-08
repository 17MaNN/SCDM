from flask import Flask, render_template, request, redirect, url_for
from transformers import pipeline
import os

# Initialize Flask app
app = Flask(__name__)

# Load the Hugging Face model
classifier = pipeline("image-classification", model="VRJBro/skin-cancer-detection")

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        # Get the prediction from Hugging Face model
        result = classifier(filepath)
        
        # Display the predicted class
        prediction = result[0]['label']
        confidence = result[0]['score']
        
        return render_template('index.html', 
                               prediction=prediction, 
                               confidence=confidence, 
                               image_path=filepath)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
