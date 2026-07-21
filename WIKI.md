# KERI Command Wiki

This page collects the main commands used in this repository.

## 1) Environment Setup (Backend)

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Upgrade dependencies if needed:

```bash
pip install -r requirements.txt --upgrade
```

## 2) Run Backend API

```bash
cd api
uvicorn main:app --reload
```

Default backend URL: `http://localhost:8000`

## 3) Frontend Commands

Install frontend dependencies:

```bash
cd frontend
npm install
```

Start frontend dev server:

```bash
cd frontend
npm start
```

Build frontend for production:

```bash
cd frontend
npm run build
```

Run frontend tests:

```bash
cd frontend
npm test
```

Eject CRA config (irreversible):

```bash
cd frontend
npm run eject
```

Optional backend URL override for frontend:

```bash
export REACT_APP_API_URL=http://localhost:8000
```

Default frontend URL: `http://localhost:3000`

## 4) Rebuild NHANES Dataset

```bash
source .venv/bin/activate
python api/fetch_nhanes.py
```

## 5) Train Biomarker Model

Train or refresh the artifact:

```bash
source .venv/bin/activate
python api/train_biomarker_model.py --force
```

Train without force (reuses existing artifact if present):

```bash
source .venv/bin/activate
python api/train_biomarker_model.py
```

Train with an explicit dataset path/name:

```bash
source .venv/bin/activate
python api/train_biomarker_model.py --dataset nhanes_merged.csv
```

## 6) Backend Tests (unittest)

Run all backend tests in `api/`:

```bash
python -m unittest discover -s api -p "test_*.py"
```

Run individual test files:

```bash
python -m unittest api/test_biomarker_flow.py
python -m unittest api/test_main_routes.py
```

## 7) Typical Local Workflow

Terminal 1:

```bash
source .venv/bin/activate
cd api
uvicorn main:app --reload
```

Terminal 2:

```bash
cd frontend
npm start
```

Optional data/model prep before starting app:

```bash
source .venv/bin/activate
python api/fetch_nhanes.py
python api/train_biomarker_model.py --force
```
