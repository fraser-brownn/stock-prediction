kill -9 $(lsof -t -i:8000)

poetry run uvicorn app.main:app &