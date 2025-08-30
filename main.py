from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

GRAPH_BASE_DIR = os.path.join(os.getcwd(), "html_graphs")

# Allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4001"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Electricity (with suburbs) -----------------
@app.get("/html_graphs/electricity")
def list_suburbs():
    category_path = os.path.join(GRAPH_BASE_DIR, "electricity")
    if not os.path.exists(category_path):
        raise HTTPException(status_code=404, detail="Category not found")
    suburbs = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
    return {"suburbs": suburbs}

@app.get("/html_graphs/electricity/{suburb}")
def list_graphs(suburb: str):
    suburb_path = os.path.join(GRAPH_BASE_DIR, "electricity", suburb)
    if not os.path.exists(suburb_path):
        raise HTTPException(status_code=404, detail="Suburb not found")
    graphs = [f for f in os.listdir(suburb_path) if f.endswith(".html")]
    return {"suburb": suburb, "graphs": graphs}

@app.get("/html_graphs/electricity/{suburb}/{graph_name}")
def get_graph(suburb: str, graph_name: str):
    file_path = os.path.join(GRAPH_BASE_DIR, "electricity", suburb, graph_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Graph not found")
    return FileResponse(file_path)

# ----------------- Water (no suburbs) -----------------
@app.get("/html_graphs/water")
def list_water_graphs():
    water_path = os.path.join(GRAPH_BASE_DIR, "water", "html_dummyWater")
    if not os.path.exists(water_path):
        raise HTTPException(status_code=404, detail="Water graphs not found")
    graphs = [f for f in os.listdir(water_path) if f.endswith(".html")]
    return {"category": "water", "graphs": graphs}

@app.get("/html_graphs/water/{graph_name}")
def get_water_graph(graph_name: str):
    file_path = os.path.join(GRAPH_BASE_DIR, "water", "html_dummyWater", graph_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Graph not found")
    return FileResponse(file_path)

# ----------------- Air Pollution (no suburbs) -----------------
@app.get("/html_graphs/pollution")
def list_pollution_graphs():
    pollution_path = os.path.join(GRAPH_BASE_DIR, "pollution", "html_dummyPol")
    if not os.path.exists(pollution_path):
        raise HTTPException(status_code=404, detail="Pollution graphs not found")
    graphs = [f for f in os.listdir(pollution_path) if f.endswith(".html")]
    return {"category": "pollution", "graphs": graphs}

@app.get("/html_graphs/pollution/{graph_name}")
def get_pollution_graph(graph_name: str):
    file_path = os.path.join(GRAPH_BASE_DIR, "pollution", "html_dummyPol", graph_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Graph not found")
    return FileResponse(file_path)
