"""FastAPI application factory — template-locked scaffold."""

from fastapi import FastAPI

from src.routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="{{project_name}}", description="{{summary}}")
    app.include_router(router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
