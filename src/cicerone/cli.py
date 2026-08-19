"""Console script: ``cicerone [--config PATH] <command>``."""

from __future__ import annotations

import argparse
import logging
import os

from cicerone import __version__

logger = logging.getLogger(__name__)

_FORWARDING_COMMANDS = frozenset({"users", "export-openapi"})
_DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_USERS_CONFIG_ERROR = (
    "cicerone users: loaded config has no enabled dashboard.users_path; "
    "set [dashboard] enabled = true and users_path, or pass --users-path"
)
_PEELABLE = {
    "-c": "config",
    "--config": "config",
    "--log-level": "log_level",
    "--log-format": "log_format",
}
_PEELABLE_EQ = (
    ("--config=", "config"),
    ("--log-level=", "log_level"),
    ("--log-format=", "log_format"),
)


def _apply_config(path: str | None) -> None:
    if path:
        os.environ["CICERONE_CONFIG_PATH"] = path


def _has_flag(argv: list[str], name: str) -> bool:
    prefix = name + "="
    return any(arg == name or arg.startswith(prefix) for arg in argv)


def _peel_globals(parser: argparse.ArgumentParser, args: argparse.Namespace, rest: list[str]) -> list[str]:
    """Allow global flags after the command (``cicerone start --config PATH``)."""
    kept: list[str] = []
    i = 0
    while i < len(rest):
        token = rest[i]
        dest = _PEELABLE.get(token)
        if dest is not None:
            if i + 1 >= len(rest):
                parser.error(f"argument {token}: expected one argument")
            setattr(args, dest, rest[i + 1])
            i += 2
            continue
        matched = False
        for prefix, eq_dest in _PEELABLE_EQ:
            if token.startswith(prefix):
                setattr(args, eq_dest, token.split("=", 1)[1])
                matched = True
                break
        if matched:
            i += 1
            continue
        kept.append(token)
        i += 1
    return kept


def _resolve_log_level(name: str) -> int:
    try:
        return logging.getLevelNamesMapping()[name.upper()]
    except KeyError:
        raise SystemExit(f"invalid log level {name!r}") from None


def _configure_logging(level_name: str | None, log_format: str | None) -> None:
    name = level_name or os.environ.get("CICERONE_LOG_LEVEL") or "INFO"
    fmt = log_format or os.environ.get("CICERONE_LOG_FORMAT") or _DEFAULT_LOG_FORMAT
    logging.basicConfig(level=_resolve_log_level(name), format=fmt)


def _parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        help="Logging level (default INFO; or CICERONE_LOG_LEVEL)",
    )
    parser.add_argument(
        "--log-format",
        metavar="FORMAT",
        help="logging.basicConfig format (default timestamp/level/name; or CICERONE_LOG_FORMAT)",
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("start", aliases=["run"], help="Serve, or one job then the scheduler, from job.mode")
    sub.add_parser("job", help="Run one training job")
    sub.add_parser("serve", help="Read API (requires job.mode = serve)")
    sub.add_parser("dashboard", help="Basic-Auth status page")
    sub.add_parser("scheduler", help="Cron loop (and optional retrain trigger)")
    sub.add_parser("users", add_help=False, help="Manage dashboard Basic Auth users")
    sub.add_parser("export-openapi", add_help=False, help="Write the serve OpenAPI document")
    return parser


def _run_job() -> int:
    from cicerone.job import run

    try:
        run()
    except Exception:
        logger.exception("Recommendation job failed")
        return 1
    return 0


def _cmd_start() -> int:
    from cicerone.config import load_settings
    from cicerone.scheduler import main as scheduler_main
    from cicerone.serve.app import main as serve_main

    settings = load_settings()
    if settings.mode == "serve":
        logger.info("mode=serve, starting read API")
        serve_main()
        return 0
    logger.info("mode=batch, running initial job")
    status = _run_job()
    if status != 0:
        return status
    logger.info("entering schedule loop (cron_schedule=%s)", settings.cron_schedule)
    scheduler_main()
    return 0


def _cmd_serve() -> int:
    from cicerone.serve.app import main as serve_main

    serve_main()
    return 0


def _cmd_dashboard() -> int:
    from cicerone.dashboard import main as dashboard_main

    dashboard_main()
    return 0


def _cmd_scheduler() -> int:
    from cicerone.scheduler import main as scheduler_main

    scheduler_main()
    return 0


def dashboard_users_path(settings: object) -> str:
    dashboard = getattr(settings, "dashboard", None)
    users_path = getattr(dashboard, "users_path", None)
    enabled = bool(getattr(dashboard, "enabled", False))
    if dashboard is None or not enabled or not users_path:
        raise SystemExit(_USERS_CONFIG_ERROR)
    return str(users_path)


def _cmd_users(argv: list[str]) -> int:
    from cicerone.manage_dashboard_users import main as users_main

    if os.environ.get("CICERONE_CONFIG_PATH") and not _has_flag(argv, "--users-path"):
        from cicerone.config import load_settings

        argv = ["--users-path", dashboard_users_path(load_settings()), *argv]
    users_main(argv)
    return 0


def _cmd_export_openapi(argv: list[str]) -> int:
    from cicerone.export_serve_openapi import main as export_main

    return export_main(argv)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args, rest = parser.parse_known_args(argv)
    rest = _peel_globals(parser, args, rest)
    if args.command not in _FORWARDING_COMMANDS and rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    _apply_config(args.config)
    _configure_logging(args.log_level, args.log_format)

    command = args.command
    if command in {"start", "run"}:
        return _cmd_start()
    if command == "job":
        return _run_job()
    if command == "serve":
        return _cmd_serve()
    if command == "dashboard":
        return _cmd_dashboard()
    if command == "scheduler":
        return _cmd_scheduler()
    if command == "users":
        return _cmd_users(rest)
    return _cmd_export_openapi(rest)


if __name__ == "__main__":
    raise SystemExit(main())
