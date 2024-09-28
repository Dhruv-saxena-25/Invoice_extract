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
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

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
    - Date
    - Description
    - Total Amount
    - etc
    
    Respond with a JSON structure like this:
    {{
      "company_name": "<company_name>",
      "date": "<date>",
      "description": "<description>",
      "amount": "<amount>"
      "ETC" : "<etc>"
    }}
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
    return response.content



if __name__=="__main__":
    path = "Asus_bill.jpeg"
    # image_path = image_to_base64(path)
    response = generate_response(path, input_prompt)
    print(response)