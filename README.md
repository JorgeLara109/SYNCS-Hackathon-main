# SYNCS Hackathon Backend

This is the **backend** of the WasteWise Resource Management Dashboard, built with **FastAPI**. It serves HTML graphs and resource data for electricity, water, and pollution.

---

## Features

- Serve **electricity graphs** organized by suburb.
- Serve **water** and **air pollution graphs** (dummy data).
- REST endpoints for fetching available graphs and serving HTML files.
- Configured **CORS** to allow requests from the frontend.

---

## Prerequisites

- Python 3.10+
- `pip` or `pipenv`
- Frontend running locally (optional, for integration)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Syncs-Hackathon-Back-End.git
cd Syncs-Hackathon-Back-End
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate Virtual Environment

MacOs/Linux:
```bash
source venv/bin/activate
```
Windows:

```bash
venv\Scripts\activate
```

4. Install Dependencies:

```bash
pip install fastapi uvicorn
```

5. Running Server:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

6. Endpoints

Electricity

```bash
GET /html_graphs/electricity – List all suburbs.

GET /html_graphs/electricity/{suburb} – List graphs for a suburb.

GET /html_graphs/electricity/{suburb}/{graph_name} – Serve a specific HTML graph.
```
Water

```bash
GET /html_graphs/water – List all water graphs.

GET /html_graphs/water/{graph_name} – Serve a specific water graph.
```

Pollution

```bash
GET /html_graphs/pollution – List all pollution graphs.

GET /html_graphs/pollution/{graph_name} – Serve a specific pollution graph.
```

7. CORS
```bash
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4002"],  # frontend URL and port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    ```
)
```
## License

This project is open-source and available under the MIT License.
