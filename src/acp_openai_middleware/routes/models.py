from __future__ import annotations

import time

from fastapi import APIRouter, Request

from ..schemas import ModelInfo, ModelsListResponse

router = APIRouter()


@router.get("/v1/models")
async def list_models(request: Request) -> ModelsListResponse:
    name = request.app.state.agent_name
    model = ModelInfo(
        id=name,
        created=int(time.time()),
        owned_by="acp-openai-middleware",
    )
    return ModelsListResponse(data=[model])
