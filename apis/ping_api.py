from fastapi import APIRouter, status, UploadFile, File, Depends
from fastapi.responses import JSONResponse, FileResponse

router = APIRouter(
    prefix="/ping",
    tags=["ml"],
    responses={status.HTTP_404_NOT_FOUND: {"description": "Not found"}}
)

@router.get(
    "",
    summary="ping api",
)
def ping() :
    return JSONResponse({"ping": "pong"})