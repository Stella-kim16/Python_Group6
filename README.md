# Python Programming Course # 

## Title: AI-powered smart energy management system

### 1. Group Overview: 

   **Ecesu Yildiran [ESE 7226935]**

    Subsystem: Real-Time Energy Monitoring & Data Management
    
  **Stella (Sung Eun Kim) [ESE 7226046]**
    
    Subsystem: AI Modelling & Prediction
     
  **Devashish Deshpande [ESE 7225975]**
   
    Subsystem: Optimisation, Visualisation, Integration

### 2. UML
**Subsystem 1:  Real-Time Energy Monitoring & Data Management**

<img width="2560" height="1221" alt="subsystem1_uml" src="https://github.com/user-attachments/assets/c0ac611e-4abc-4bb2-8304-813fbc02e7aa" />




**Subsystem 2: AI Modelling & Prediction**

<img width="700" height="624" alt="uml_class" src="https://github.com/user-attachments/assets/31cbfb1c-0f1c-4bf5-930c-4d50005a6281" />




**Subsystem 3:  Optimization, Visualization, Integration**

<img width="700" height="624" alt="dede" src="https://github.com/user-attachments/assets/a6bcf277-8ade-4cde-b2c3-7d4f30281dcd" />




### 3. How to start

## How to Run with Docker

The easiest way to run the Smart Energy Management System is with Docker.
Docker installs Python and all required libraries inside a container, so nothing else needs to be installed on your computer.

### 1. Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows / macOS) or Docker Engine (Linux)
- [Git](https://git-scm.com/downloads)

Start Docker Desktop and wait until it shows **Engine running**.
Check the installation:

```bash
docker --version
```

### 2. Clone the repository

```bash
git clone https://github.com/Stella-kim16/Python_Group6.git
cd Python_Group6
```

### 3. Build the Docker image

```bash
docker build -t energy-ai .
```

The first build may take a few minutes, as all libraries are downloaded and installed.

### 4. Run the dashboard

```bash
docker run --rm -p 8501:8501 energy-ai
```

Then open **http://localhost:8501** in your browser.
The first page may take a few seconds to load, as the AI model is trained when the app starts.

**Optional – AI Operator Insights (KI:connect NRW):**
To enable the AI-generated summary on the *Optimisation & Visualisation* page, pass your API key:

```bash
docker run --rm -p 8501:8501 -e KICONNECT_API_KEY=your_api_key energy-ai
```

Without a key, all other pages work normally and the AI summary shows a short "unavailable" message.

### 5. Stop the application

Press **Ctrl + C** in the terminal.

---

### Alternative: Run the full pipeline with Docker Compose

This runs the complete pipeline first (Subsystem 1 data processing and Subsystem 2 model training, `main.py`) and then starts the dashboard.

```bash
docker compose up --build
```

Open **http://localhost:8501** once the dashboard has started.
To stop and remove the containers:

```bash
docker compose down
```

---

### Alternative: Run without Docker

Requires Python 3.11 or newer.

```bash
pip install -r requirements.txt
python main.py
streamlit run app.py
```

---

### Troubleshooting

| Problem | Solution |
|---|---|
| `Cannot connect to the Docker daemon` | Start Docker Desktop and wait until it shows *Engine running*. |
| `COPY failed` during the build | Run the command inside the project folder (where the `Dockerfile` is). |
| `port is already allocated` | Port 8501 is in use. Run with `-p 8502:8501` and open `http://localhost:8502`. |
| The page keeps loading | Wait a few seconds – the model is trained on the first start. |

### 4. Video link
https://drive.google.com/file/d/1lx9FxFV7E_EulXpa7_Hcb_Uf-fNIociH/view?usp=sharing

