from fastapi import FastAPI
from app.services.llm_service import init_session, close_session
from app.api.endpoints import ingest, query
from app.api.routers import cag, rag, vlrag, multirag, mmrrag

app = FastAPI()

app.include_router(cag.router)
app.include_router(rag.router)
app.include_router(vlrag.router)
app.include_router(multirag.router)
app.include_router(mmrrag.router)

app.include_router(ingest.router)
app.include_router(query.router)

@app.on_event("startup")
async def startup():
    await init_session()
    await cag.init()
    await rag.init()
    await vlrag.init()
    await multirag.init()
    await mmrrag.init()

@app.on_event("shutdown")
async def shutdown():
    await close_session()
