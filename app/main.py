"""HTTP surface for the contour reconstruction service."""

from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse

from app.geometry import (
    ReconstructionError,
    parse_outer_hole_segments,
    parse_segments,
    reconstruct,
    reconstruct_with_hole,
)

app = FastAPI(title="Sheet-metal contour reconstruction", version="1.1.0")


@app.exception_handler(ReconstructionError)
async def reconstruction_error_handler(
    _request: Request, exc: ReconstructionError
) -> JSONResponse:
    # Every deterministic rejection (invalid input or broken topology) is a
    # 422 carrying a stable code plus its witness.
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "code": exc.code,
                "message": exc.message,
                "witness": exc.witness,
            }
        },
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/reconstruct")
def reconstruct_contour(payload: Any = Body(...)) -> dict:
    segments = parse_segments(payload)
    return reconstruct(segments)


@app.post("/reconstruct-with-hole")
def reconstruct_contour_with_hole(payload: Any = Body(...)) -> dict:
    outer, hole = parse_outer_hole_segments(payload)
    return reconstruct_with_hole(outer, hole)
