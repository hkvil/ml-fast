import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import io

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

# Mount the static directory to serve uploaded files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define the path to the upload directory
UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Load the model (make sure the model is in the same directory or provide the correct path)
# model = load_model('RPC_Model.h5')
model = load_model('Ensemble.keras')


def predict_image(path: str):
    image_path = path
    pict = image.load_img(image_path, target_size=(150, 150))
    x_array = image.img_to_array(pict)
    x_array = np.expand_dims(x_array, axis=0)
    prediction = model.predict(np.vstack([x_array]), batch_size=18)
    return str(prediction)
  

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Save the image to the UPLOAD_DIR
    file_location = UPLOAD_DIR / file.filename
    with file_location.open("wb") as buffer:
        buffer.write(image_data)
        
    predicted_class = predict_image(f"./static/uploads/{file.filename}")

    # http://127.0.0.1:8000/static/uploads/gunting.png
    return {
        "info": f"file '{file.filename}' saved at '{file_location}'",
        "predicted_class": predicted_class,
        "format": image.format,
        "size": image.size,
        "mode": image.mode
    }


@app.get("/")
async def main():
    content = """
    <body>
    <h1>Upload an image</h1>
    <form action="/upload/" enctype="multipart/form-data" method="post">
    <input name="file" type="file">
    <input type="submit">
    </form>
    </body>
    """
    return HTMLResponse(content=content)
    
