"""Command-line interface."""

from __future__ import annotations

import argparse
from dataclasses import replace

import uvicorn

from .config import RealtimeConfig
from .server import TurnBridge, create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voxtral-realtime")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the FastAPI turn bridge")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--model")
    serve.add_argument("--vllm-url")
    serve.add_argument("--log-level", default="info")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = RealtimeConfig.from_env()
    config = replace(
        config,
        **{
            key: value
            for key, value in {
                "host": args.host,
                "port": args.port,
                "model_id": args.model,
                "vllm_url": args.vllm_url,
            }.items()
            if value is not None
        },
    )
    app = create_app(TurnBridge(config))
    uvicorn.run(app, host=config.host, port=config.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
