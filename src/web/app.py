"""Aplicación FastAPI para la demo de acceso dual."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import PROJECT_ROOT
from src.web.database import Database
from src.web.schemas import ValidationError
from src.web.services import DemoService
from src.web.storage import ImageStorage


WEB_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class WebSettings:
    database_path: Path = PROJECT_ROOT / "data" / "demo" / "dual_access_demo.sqlite3"
    reference_dir: Path = PROJECT_ROOT / "data" / "demo" / "reference_images"
    capture_dir: Path = PROJECT_ROOT / "data" / "demo" / "captures"
    config_path: Path = PROJECT_ROOT / "config" / "model_config.json"


def _redirect(path: str, **query: str) -> RedirectResponse:
    suffix = f"?{urlencode(query)}" if query else ""
    return RedirectResponse(f"{path}{suffix}", status_code=303)


def create_app(
    settings: WebSettings | None = None,
    verifier_provider=None,
) -> FastAPI:
    settings = settings or WebSettings()
    database = Database(settings.database_path)
    storage = ImageStorage(settings.reference_dir, settings.capture_dir)
    service = DemoService(database, storage, settings.config_path, verifier_provider)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        database.initialize()
        storage.initialize()
        yield

    app = FastAPI(title="Control de acceso dual", version="1.0.0", lifespan=lifespan)
    app.state.database = database
    app.state.storage = storage
    app.state.service = service
    templates = Jinja2Templates(directory=str(WEB_ROOT / "templates"))
    app.mount("/static", StaticFiles(directory=str(WEB_ROOT / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "result": None,
                "threshold": service.config.threshold,
                "error": request.query_params.get("error"),
            },
        )

    @app.post("/verify", response_class=HTMLResponse)
    async def verify(
        request: Request,
        rfid_uid: str = Form(...),
        capture: UploadFile | None = File(None),
    ):
        if capture is None or not capture.filename:
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "result": None,
                    "threshold": service.config.threshold,
                    "error": "Debes cargar o tomar una imagen de captura.",
                },
                status_code=422,
            )
        capture_path: Path | None = None
        try:
            capture_path = await storage.save_capture(capture)
            result = service.verify_access(rfid_uid, capture_path)
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={"result": result, "threshold": service.config.threshold, "error": None},
            )
        except (ValidationError, ValueError) as error:
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "result": None,
                    "threshold": service.config.threshold,
                    "error": str(error),
                },
                status_code=422,
            )
        finally:
            if capture_path:
                storage.remove(capture_path)

    @app.get("/users", response_class=HTMLResponse)
    async def users(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="users.html",
            context={
                "users": database.list_users(),
                "success": request.query_params.get("success"),
                "error": request.query_params.get("error"),
            },
        )

    @app.get("/users/new", response_class=HTMLResponse)
    async def new_user(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={"error": request.query_params.get("error")},
        )

    @app.post("/users")
    async def create_user(
        full_name: str = Form(...),
        rfid_uid: str = Form(...),
        references: list[UploadFile] = File(...),
    ):
        try:
            user = await service.register_user(full_name, rfid_uid, references)
        except (ValidationError, ValueError) as error:
            return _redirect("/users/new", error=str(error))
        return _redirect("/users", success=f"Usuario {user.full_name} registrado.")

    @app.post("/users/{user_id}/toggle")
    async def toggle_user(user_id: int):
        user = database.get_user(user_id)
        if user is None:
            return _redirect("/users", error="El usuario no existe.")
        database.set_user_active(user_id, not user.is_active)
        state = "activado" if not user.is_active else "desactivado"
        return _redirect("/users", success=f"Usuario {state}.")

    @app.get("/users/{user_id}/references", response_class=HTMLResponse)
    async def references(request: Request, user_id: int):
        user = database.get_user(user_id)
        if user is None:
            return _redirect("/users", error="El usuario no existe.")
        return templates.TemplateResponse(
            request=request,
            name="references.html",
            context={"user": user, "references": database.list_references(user_id)},
        )

    @app.get("/history", response_class=HTMLResponse)
    async def history(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="history.html",
            context={"events": database.list_events()},
        )

    @app.get("/health")
    async def health():
        return {"status": "ok", "threshold": service.config.threshold}

    return app


app = create_app()


def main() -> None:
    uvicorn.run("src.web.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
