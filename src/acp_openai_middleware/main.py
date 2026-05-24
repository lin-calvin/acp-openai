from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from .agent_manager import AgentSessionManager
from .routes.chat import router as chat_router
from .routes.models import router as models_router

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ACP-to-OpenAI middleware — expose ACP agents via OpenAI-compatible API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m acp_openai_middleware --agent "python examples/agent.py" --api-key sk-test
  python -m acp_openai_middleware --agent /usr/local/bin/my-agent --port 8080 --api-key sk-prod
  python -m acp_openai_middleware --agent uv --agent-args "run examples/agent.py" --api-key sk-test
""",
    )
    parser.add_argument(
        "--agent",
        required=True,
        help="Command or path to launch the ACP agent (e.g. 'python agent.py' or '/path/to/agent')",
    )
    parser.add_argument(
        "--agent-args",
        nargs="*",
        default=[],
        help="Additional arguments to pass to the agent subprocess",
    )
    parser.add_argument(
        "--agent-cwd",
        default=os.getcwd(),
        help="Working directory for the agent subprocess (default: current dir)",
    )
    parser.add_argument(
        "--agent-env",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variables for the agent subprocess",
    )
    parser.add_argument(
        "--api-key",
        action="append",
        dest="api_keys",
        default=[],
        help="Allowed API key (can be specified multiple times; empty = allow all)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP listen port (default: 8000)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="HTTP bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--session-ttl",
        type=int,
        default=3600,
        help="Seconds before idle session eviction (default: 3600)",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=50,
        help="Max sessions per API key namespace (default: 50)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)",
    )
    return parser.parse_args(argv)


def build_agent_env(env_args: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in env_args:
        if "=" in item:
            key, _, value = item.partition("=")
            result[key] = value
        else:
            logger.warning("Ignoring malformed --agent-env value: %s", item)
    return result


def _derive_agent_name(args: argparse.Namespace) -> str:
    agent = args.agent
    if agent in ("python", "python3") and args.agent_args:
        name = args.agent_args[-1]
    elif agent.endswith(".py"):
        name = agent
    else:
        name = agent
    return os.path.basename(name.rstrip(".py"))


def create_app(args: argparse.Namespace) -> FastAPI:
    env = build_agent_env(args.agent_env)
    manager = AgentSessionManager(
        agent_command=args.agent,
        agent_args=args.agent_args,
        cwd=args.agent_cwd,
        env=env,
        session_ttl=args.session_ttl,
        max_sessions=args.max_sessions,
    )

    agent_name = _derive_agent_name(args)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.agent_manager = manager
        app.state.agent_name = agent_name
        app.state.api_keys = args.api_keys
        logger.info("ACP-OpenAI middleware started (agent=%s, port=%d)", args.agent, args.port)
        yield
        logger.info("Shutting down...")
        await manager.close_all()

    app = FastAPI(title="ACP OpenAI Middleware", version="0.1.0", lifespan=lifespan)
    app.include_router(chat_router)
    app.include_router(models_router)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": "internal_error", "code": 500}},
        )

    return app


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    app = create_app(args)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
