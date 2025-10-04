
from fastapi import FastAPI
from apis.ping_api import router as ping_router

app = FastAPI(title="Practice API", version="1.0.0")

app.include_router(ping_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7071)
# python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
