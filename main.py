from flask import Flask, request, render_template, jsonify
import os
import pytesseract
from PIL import Image
import requests

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Read the API key from apikey.txt
with open("apikey.txt", "r") as file:
    OPENAI_API_KEY = file.read().strip()

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def fetch_related_link_if_needed(text):
    """
    Placeholder: If the extracted text is too short,
    return a sample related link for additional context.
    """
    if len(text.strip()) < 20:
        return "Additional context: https://example.com/more-info"
    return ""

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Extract text from the image using OCR
        try:
            image = Image.open(filepath)
            extracted_text = pytesseract.image_to_string(image)
        except Exception as e:
            return jsonify({'error': 'Error processing image: ' + str(e)})
        
        # If extracted text is insufficient, append a related link
        additional_context = fetch_related_link_if_needed(extracted_text)
        
        # Build the prompt with clear instructions and additional context if needed
        prompt = (
            f"Analyze the following social media post for potential misinformation. "
            f"Return a risk score between 1 and 100 and provide a clear explanation for why "
            f"the content might be misleading. Post text: {extracted_text} {additional_context}"
        )
        
        # Prepare the payload for the OpenAI API
        data = {
            "model": "gpt-4o-mini",
            "store": True,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        try:
            response = requests.post(OPENAI_URL, headers=headers, json=data)
            result = response.json()
            print("Full API Response:", result)  # For debugging
            if 'choices' in result and len(result['choices']) > 0:
                api_response = result['choices'][0]['message']['content']
            else:
                api_response = "No valid response from API."
        except Exception as e:
            return jsonify({'error': 'Error calling API: ' + str(e)})

        
        # Clean up the uploaded file
        os.remove(filepath)
        
        return jsonify({
            "extracted_text": extracted_text,
            "analysis": api_response
        })
    return jsonify({'error': 'Invalid file type'})

if __name__ == "__main__":
    app.run(debug=True)
