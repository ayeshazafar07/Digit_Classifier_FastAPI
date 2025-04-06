import pickle
import io
import numpy as np 
import PIL.Image
import PIL.ImageOps
from fastapi import FastAPI, File, UploadFile # FastAPI works with type hinting so we need to specify what type of file types or data parameters we have 
from fastapi.middleware.cors import CORSMiddleware # to connect to server

with open('mnist_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

# So we can send request to html file
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Url pattern and the async fucntion takes a file as parameter
@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    # We open the byte stream of image and store content
    pil_image = PIL.Image.open(io.BytesIO(contents)).convert('L')
    # Pillow processes image different than scikit learn so we need to convert black to white and white to black here
    pil_image = PIL.ImageOps.invert(pil_image)
    pil_image = pil_image.resize((28, 28), PIL.Image.Resampling.LANCZOS)
    # Converting to numpy array and predicting image content
    img_array = np.array(pil_image).reshape(1, -1)
    prediction = model.predict(img_array)
    return {"prediction": int(prediction[0])}
