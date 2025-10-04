
import db.db #db 초기화
from fastapi import FastAPI
from apis.ping_api import router as ping_router
from apis.research_api import router as research_router
from apis.hypothesis_api import router as hypothesis_router
from exception.handlers import register_exception_handlers

app = FastAPI(title="Practice API", version="1.0.0")

app.include_router(ping_router) #핑 컨트롤러
app.include_router(research_router)  #연구 간극 api
app.include_router(hypothesis_router)  # 가설 api

register_exception_handlers(app) #에러 핸들러 등록

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7071)
# python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
