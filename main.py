import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

# Mount the static directory to serve uploaded files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define the path to the upload directories
UPLOAD_DIR = Path("static/uploads")
MODEL_DIR = Path("models")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the available models dictionary
AVAILABLE_MODELS = {}


def load_models():
    """Load all model names from the model directory into AVAILABLE_MODELS."""
    for model_file in MODEL_DIR.glob("*.h5"):
        model_name = model_file.stem  # Name without extension
        AVAILABLE_MODELS[model_name] = str(model_file)
        
    for model_file in MODEL_DIR.glob("*.keras"):
        model_name = model_file.stem  # Name without extension
        AVAILABLE_MODELS[model_name] = str(model_file)

# Load models initially
load_models()


def load_selected_model(model_name: str):
    """Dynamically load the selected model."""
    model_path = AVAILABLE_MODELS.get(model_name)
    if model_path:
        return load_model(model_path)
    else:
        raise ValueError(f"Model '{model_name}' not found.")


def predict_image(model, path: str):
    image_path = path

    input_shape = model.input_shape
    target_size = input_shape[1:3]
    
    pict = image.load_img(image_path, target_size=target_size)
    x_array = image.img_to_array(pict)
    x_array = np.expand_dims(x_array, axis=0)
    prediction = model.predict(np.vstack([x_array]), batch_size=18)
    return str(np.argmax(prediction))


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), model_name: str = Form(...)):
    # Load the selected model
    try:
        model = load_selected_model(model_name)
    except ValueError as e:
        return {"error": str(e)}

    # Read and save the uploaded image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Save the image to the UPLOAD_DIR
    file_location = UPLOAD_DIR / file.filename
    with file_location.open("wb") as buffer:
        buffer.write(image_data)
        
    # Perform prediction
    predicted_class = predict_image(model, f"./static/uploads/{file.filename}")

    return {
        "info": f"file '{file.filename}' saved at '{file_location}'",
        "predicted_class": predicted_class,
        "model_used": model_name,
        "format": image.format,
        "size": image.size,
        "mode": image.mode
    }


@app.post("/add_model/")
async def add_model(file: UploadFile = File(...), model_name: str = Form(...)):
    """API to upload and add a new model."""
    # Ensure the file is either .h5 or .keras
    if not (file.filename.endswith(".h5") or file.filename.endswith(".keras")):
        raise HTTPException(status_code=400, detail="Only .h5 and .keras model files are allowed.")
    
    # Save the model file
    model_file_path = MODEL_DIR / f"{model_name}{Path(file.filename).suffix}"
    with model_file_path.open("wb") as buffer:
        buffer.write(await file.read())
    
    # Add the model to the available models dictionary
    AVAILABLE_MODELS[model_name] = str(model_file_path)

    return {"info": f"Model '{model_name}' has been successfully uploaded and added."}


@app.get("/models/")
async def get_models():
    """API to retrieve the list of available models."""
    return {"available_models": list(AVAILABLE_MODELS.keys())}


@app.get("/")
async def main():
    """Main HTML form to upload an image and select a model."""
    load_models()  # Reload models in case any have been added
    model_options = "".join([f'<option value="{model}">{model}</option>' for model in AVAILABLE_MODELS.keys()])
    
    content = f"""
    <body>
    <h1>Upload an image</h1>
    <form action="/upload/" enctype="multipart/form-data" method="post">
        <label for="model_name">Select Model:</label>
        <select name="model_name" id="model_name">
            {model_options}
        </select>
        <br><br>
        <input name="file" type="file">
        <input type="submit">
    </form>
    
    <h2>Add a new model</h2>
    <form action="/add_model/" enctype="multipart/form-data" method="post">
        <label for="model_name">Model Name:</label>
        <input name="model_name" type="text">
        <br><br>
        <input name="file" type="file">
        <input type="submit">
    </form>
    </body>
    """
    return HTMLResponse(content=content)
