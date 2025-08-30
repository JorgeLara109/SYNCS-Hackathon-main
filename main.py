from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import os

app = FastAPI()

GRAPH_BASE_DIR = os.path.join(os.getcwd(), "html_graphs")

# Allow requests from your Next.js frontend (dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4001"],  # Next.js dev URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
