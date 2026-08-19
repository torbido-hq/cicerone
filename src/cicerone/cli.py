"""Console script: ``cicerone [--config PATH] <command>``."""

from __future__ import annotations

import argparse
import logging
import os

from cicerone import __version__

logger = logging.getLogger(__name__)

_FORWARDING_COMMANDS = frozenset({"users", "export-openapi"})
_DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _apply_config(path: str | None) -> None:
    if path:
        os.environ["CICERONE_CONFIG_PATH"] = path


def _has_flag(argv: list[str], name: str) -> bool:
    prefix = name + "="
    return any(arg == name or arg.startswith(prefix) for arg in argv)


def _add_global_flags(parser: argparse.ArgumentParser, *, suppress: bool = False) -> None:
    default: object = argparse.SUPPRESS if suppress else None
    parser.add_argument(
        "-c",
        "--config",
        default=default,
        metavar="PATH",
        help="TOML config path (sets CICERONE_CONFIG_PATH for this process)",
    )
    parser.add_argument(
        "--log-level",
        default=default,
        metavar="LEVEL",
        help="Logging level (default INFO; or CICERONE_LOG_LEVEL)",
    )
    parser.add_argument(
        "--log-format",
        default=default,
        metavar="FORMAT",
        help="logging.basicConfig format (default timestamp/level/name; or CICERONE_LOG_FORMAT)",
    )


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
    _add_global_flags(parser)
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_global_flags(
        sub.add_parser("start", aliases=["run"], help="Serve, or one job then the scheduler, from job.mode"),
        suppress=True,
    )
    _add_global_flags(sub.add_parser("job", help="Run one training job"), suppress=True)
    _add_global_flags(sub.add_parser("serve", help="Read API (requires job.mode = serve)"), suppress=True)
    _add_global_flags(sub.add_parser("dashboard", help="Basic-Auth status page"), suppress=True)
    _add_global_flags(
        sub.add_parser("scheduler", help="Cron loop (and optional retrain trigger)"),
        suppress=True,
    )
    _add_global_flags(
        sub.add_parser("users", add_help=False, help="Manage dashboard Basic Auth users"),
        suppress=True,
    )
    _add_global_flags(
        sub.add_parser("export-openapi", add_help=False, help="Write the serve OpenAPI document"),
        suppress=True,
    )
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


def _users_config_error(
    config_path: str | None,
    dashboard: object | None,
    enabled: bool,
    users_path: object | None,
) -> str:
    loaded = config_path or os.environ.get("CICERONE_CONFIG_PATH") or "the default config path"
    if dashboard is None:
        detail = "no [dashboard] section"
    elif not enabled:
        detail = f"dashboard.enabled = false (users_path = {users_path!r})"
    else:
        detail = f"dashboard.users_path is missing or empty ({users_path!r})"
    return (
        f"cicerone users: cannot use dashboard.users_path from {loaded}: {detail}. "
        "Set [dashboard] enabled = true and users_path, or pass --users-path PATH."
    )


def dashboard_users_path(settings: object, *, config_path: str | None = None) -> str:
    dashboard = getattr(settings, "dashboard", None)
    users_path = getattr(dashboard, "users_path", None)
    enabled = bool(getattr(dashboard, "enabled", False))
    if dashboard is None or not enabled or not users_path:
        raise SystemExit(_users_config_error(config_path, dashboard, enabled, users_path))
    return str(users_path)


def _cmd_users(argv: list[str]) -> int:
    from cicerone.manage_dashboard_users import main as users_main

    if os.environ.get("CICERONE_CONFIG_PATH") and not _has_flag(argv, "--users-path"):
        from cicerone.config import load_settings

        config_path = os.environ.get("CICERONE_CONFIG_PATH")
        argv = ["--users-path", dashboard_users_path(load_settings(), config_path=config_path), *argv]
    users_main(argv)
    return 0


def _cmd_export_openapi(argv: list[str]) -> int:
    from cicerone.export_serve_openapi import main as export_main

    return export_main(argv)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args, rest = parser.parse_known_args(argv)
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
