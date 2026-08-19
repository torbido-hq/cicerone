"""Console script: ``cicerone [--config PATH] <command>``."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from cicerone import __version__

logger = logging.getLogger(__name__)


def _apply_config(path: str | None) -> None:
    if path:
        os.environ["CICERONE_CONFIG_PATH"] = path


def _cmd_start() -> None:
    from cicerone.config import load_settings
    from cicerone.job import run
    from cicerone.scheduler import main as scheduler_main
    from cicerone.serve.app import main as serve_main

    settings = load_settings()
    if settings.mode == "serve":
        logger.info("mode=serve, starting read API")
        serve_main()
        return
    logger.info("mode=batch, running initial job")
    try:
        run()
    except Exception:
        logger.exception("Recommendation job failed")
        raise SystemExit(1) from None
    logger.info("entering schedule loop (cron_schedule=%s)", settings.cron_schedule)
    scheduler_main()


def _cmd_job() -> None:
    from cicerone.job import run

    try:
        run()
    except Exception:
        logger.exception("Recommendation job failed")
        raise SystemExit(1) from None


def _cmd_serve() -> None:
    from cicerone.serve.app import main as serve_main

    serve_main()


def _cmd_dashboard() -> None:
    from cicerone.dashboard import main as dashboard_main

    dashboard_main()


def _cmd_scheduler() -> None:
    from cicerone.scheduler import main as scheduler_main

    scheduler_main()


def _cmd_users(config: str | None, argv: list[str]) -> None:
    from cicerone.manage_dashboard_users import main as users_main

    if config and "--users-path" not in argv:
        from cicerone.config import load_settings

        argv = ["--users-path", load_settings().dashboard.users_path, *argv]
    users_main(argv)


def _cmd_export_openapi(argv: list[str]) -> None:
    from cicerone.export_serve_openapi import main as export_main

    raise SystemExit(export_main(argv))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="cicerone",
        description="Batch recommender, serve API, and dashboard. Stop with SIGTERM / Ctrl-C.",
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="PATH",
        help="TOML config path (sets CICERONE_CONFIG_PATH for this process)",
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "command",
        choices=("start", "run", "job", "serve", "dashboard", "scheduler", "users", "export-openapi"),
        help="start/run: serve or job+scheduler from job.mode",
    )
    parser.add_argument("argv", nargs=argparse.REMAINDER, help="Command-specific arguments")
    args = parser.parse_args(argv)
    _apply_config(args.config)

    command = args.command
    if command in {"start", "run"}:
        _cmd_start()
    elif command == "job":
        _cmd_job()
    elif command == "serve":
        _cmd_serve()
    elif command == "dashboard":
        _cmd_dashboard()
    elif command == "scheduler":
        _cmd_scheduler()
    elif command == "users":
        _cmd_users(args.config, args.argv)
    else:
        _cmd_export_openapi(args.argv)


if __name__ == "__main__":
    main(sys.argv[1:])
