import json
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.recsys import ModelHyperParams, build_engine_and_artifacts, load_engine

logger = logging.getLogger(__name__)


def _str_to_bool(s: str, default: bool = False) -> bool:
    return s.strip().lower() in {"1", "true", "yes", "y", "on"} if s is not None else default


class SimulateRequest(BaseModel):
    user_id: int = Field(..., ge=0)
    item_id: int = Field(..., ge=0)
    rating: float = Field(4.0, ge=0.5, le=5.0)
    strategy: str = Field("hybrid", description="user_cf | item_cf | content | svd | hybrid")
    k: int = Field(10, ge=1, le=100)


def configure_logging() -> None:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, "app.log"), encoding="utf-8"),
        ],
    )


def create_app() -> FastAPI:
    configure_logging()

    data_dir = os.environ.get("DATA_DIR", "data")
    model_path = os.environ.get("MODEL_PATH", "models/artifacts.joblib")
    auto_train = _str_to_bool(os.environ.get("AUTO_TRAIN", "true"), default=True)
    test_fraction = float(os.environ.get("TEST_FRACTION", "0.2"))
    min_user_ratings = int(os.environ.get("MIN_USER_RATINGS", "10"))

    hyperparams = ModelHyperParams()

    app = FastAPI(title="Shyft Recommendation Engine", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve frontend UI
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    engine = None
    if os.path.exists(model_path):
        logger.info("Loading artifacts from %s", model_path)
        engine = load_engine(model_path)
    else:
        if not auto_train:
            raise RuntimeError(f"Model artifacts missing: {model_path}. Set AUTO_TRAIN=true to build them.")
        logger.info("Artifacts missing. Training models (this can take a while)...")
        ensure_dir = lambda p: os.makedirs(p, exist_ok=True)
        ensure_dir(os.path.dirname(model_path) or ".")
        _, report = build_engine_and_artifacts(
            data_dir=data_dir,
            model_path=model_path,
            hyperparams=hyperparams,
            test_fraction=test_fraction,
            seed=42,
            min_user_ratings=min_user_ratings,
        )
        logger.info("Training finished: %s", report)
        engine = load_engine(model_path)

    if engine is None:
        raise RuntimeError("Engine failed to load.")

    app.state.stats = {
        "requests_total": 0,
        "recommend_calls_total": 0,
        "simulate_calls_total": 0,
    }

    @app.middleware("http")
    async def count_requests(request: Request, call_next):
        app.state.stats["requests_total"] += 1
        return await call_next(request)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(os.path.join(frontend_dir, "index.html"))

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "model_path": model_path,
            "user_count": len(engine.user_id_to_idx),
            "item_count": len(engine.item_id_to_idx),
        }

    @app.get("/recommend")
    def recommend(
        user_id: int = Query(..., ge=0),
        k: int = Query(10, ge=1, le=100),
        strategy: str = Query("hybrid", description="user_cf | item_cf | content | svd | hybrid"),
        exclude_seen: bool = Query(True),
    ) -> JSONResponse:
        try:
            recs = engine.recommend(user_id=user_id, k=k, strategy=strategy, exclude_seen=exclude_seen)
            logger.info("recommend user_id=%s strategy=%s k=%s", user_id, strategy, k)
            app.state.stats["recommend_calls_total"] += 1
            return JSONResponse({"user_id": user_id, "strategy": strategy, "recommendations": recs})
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/simulate")
    def simulate(req: SimulateRequest) -> JSONResponse:
        # Demo-only: we update an in-memory overlay and re-run scoring.
        try:
            engine.add_interaction_for_demo(req.user_id, req.item_id, req.rating)
            recs = engine.recommend(user_id=req.user_id, k=req.k, strategy=req.strategy, exclude_seen=True)
            payload = {"user_id": req.user_id, "item_id": req.item_id, "rating": req.rating, "recommendations": recs}
            logger.info("simulate user_id=%s item_id=%s rating=%s strategy=%s", req.user_id, req.item_id, req.rating, req.strategy)
            app.state.stats["simulate_calls_total"] += 1
            return JSONResponse(payload)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/metrics")
    def metrics() -> Dict[str, Any]:
        return {**app.state.stats}

    @app.post("/reset_demo")
    def reset_demo() -> JSONResponse:
        engine.overlay.user_extra_ratings.clear()
        return JSONResponse({"status": "ok"})

    @app.get("/models/meta")
    def models_meta() -> Dict[str, Any]:
        # Small endpoint to help debugging/demos.
        try:
            # artfacts are loaded in memory; keep it light.
            return {"hyperparams": engine.hyperparams.__dict__}
        except Exception:
            return {"hyperparams": None}

    return app

