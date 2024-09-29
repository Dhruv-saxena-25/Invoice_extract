from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pdf2image import convert_from_path
import base64
import mimetypes
import os
import shutil
from dotenv import load_dotenv
load_dotenv()
import os 
import json
import html
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = 'uploads'

os.makedirs(UPLOAD_FOLDER, exist_ok= True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Function to convert PDF to images and save them
def pdf_to_images(pdf_path):
    # Convert PDF to images
    images = convert_from_path(pdf_path)

    # Ensure the output folder exists
    output_folder = "artifacts/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok= True)

    # Save each image
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f"page_{i + 1}.png")  # Save as PNG
        image.save(image_path, "PNG")  # You can change "PNG" to "JPEG" if needed
        return image_path
    
# Function to convert image to base64
def image_to_base64(path):
    os.makedirs("artifacts/", exist_ok= True)
    _, extension = os.path.splitext(path)
    if extension == ".pdf":
        image_path = pdf_to_images(path)
    else: 
        image_path = path
        shutil.copy(image_path, "artifacts/")
    mime_type, _ = mimetypes.guess_type(image_path)
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")
    base64_image = f"data:{mime_type};base64,{base64_string}"
    return base64_image



input_prompt = """
You are an expert in understanding invoices.
You will receive input images as invoices &
You will have to extract the following information
Please extract important information from invoice, such as:
    - Name 
    - Date of purchase
    - Customer Address
    - Description of product
    - Total Amount
    - Buyer Name
    - other details
    """


def generate_response(base64_image, input_prompt):
    llm = ChatGoogleGenerativeAI(model= "gemini-1.5-flash", temperature= 0.8)
    extracted_inforamtion = HumanMessage(
    content=[
        {
            "type": "text",
            "text": input_prompt,
        },  # You can optionally provide text parts
        {"type": "image_url", "image_url": image_to_base64(base64_image)},
    ]
)
    response = llm.invoke([extracted_inforamtion])
    print(response)
    result= response.content.strip().replace('```json', '').replace('```', '').strip()
    analysis = json.loads(result)
    return analysis


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return render_template('index.html')
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return render_template('index.html')
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                answer = generate_response(filepath, input_prompt)
                return render_template('result.html', response=answer)
            except ValueError as e:
                flash(str(e))
                return render_template('index.html')
            finally:
                pass 
                # os.remove(filepath)  # Clean up the uploaded file
        else:
            flash('Allowed file type is PDF')
            return render_template('index.html')
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host= "0.0.0.0", port= 5000, debug=True)
