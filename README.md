# AnoShift IDS
### Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install pip setuptools wheel
python3 -m pip install -e .
```
### API
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir IDSAnoShift --reload-dir app

