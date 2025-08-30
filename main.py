from fastapi import FastAPI
from fastapi.responses import FileResponse
import os

app = FastAPI()

@app.get("/plot/sine")
def get_sine_plot():
    file_path = os.path.join(os.getcwd(), "sine_wave.html")  # path to your HTML
    return FileResponse(file_path)
