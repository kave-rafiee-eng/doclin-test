from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from graphrag.cli.query import (
    run_basic_search,
    run_drift_search,
    run_global_search,
    run_local_search,
)
from pydantic import BaseModel, Field

GRAPHRAG_ROOT = (Path(__file__).resolve().parent / "graphrag-yaskawa-l000a").resolve()

if not (GRAPHRAG_ROOT / "settings.yaml").is_file():
    raise FileNotFoundError(f"GraphRAG settings.yaml not found at {GRAPHRAG_ROOT}")

DEFAULT_RESPONSE_TYPE = "Multiple Paragraphs"
DEFAULT_COMMUNITY_LEVEL = 0

app = FastAPI(title="GraphRAG Query API", version="1.0.0")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to ask the index")
    method: Literal["local", "global", "drift", "basic"] = "local"
    community_level: int = DEFAULT_COMMUNITY_LEVEL
    response_type: str = DEFAULT_RESPONSE_TYPE
    streaming: bool = False
    verbose: bool = False
    dynamic_community_selection: bool = False


class QueryResponse(BaseModel):
    query: str
    method: str
    answer: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "graphrag_root": str(GRAPHRAG_ROOT)}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    common_kwargs = {
        "data_dir": None,
        "root_dir": GRAPHRAG_ROOT,
        "response_type": request.response_type,
        "streaming": request.streaming,
        "query": request.query,
        "verbose": request.verbose,
    }

    try:
        match request.method:
            case "local":
                answer, _ = run_local_search(
                    community_level=request.community_level,
                    **common_kwargs,
                )
            case "global":
                answer, _ = run_global_search(
                    community_level=request.community_level,
                    dynamic_community_selection=request.dynamic_community_selection,
                    **common_kwargs,
                )
            case "drift":
                answer, _ = run_drift_search(
                    community_level=request.community_level,
                    **common_kwargs,
                )
            case "basic":
                answer, _ = run_basic_search(**common_kwargs)
            case _:
                raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        query=request.query,
        method=request.method,
        answer=answer,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("graphrag-cli:app", host="0.0.0.0", port=8000, reload=False)
