"""Command-line entry point: ``gridsight <command>``.

Works the same on Windows, macOS and Linux (no ``make`` needed). Run ``gridsight --help``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence

log = logging.getLogger("gridsight")

Handler = Callable[[argparse.Namespace], int]
_COMMANDS: dict[str, tuple[str, Handler, Callable[[argparse.ArgumentParser], None]]] = {}


def command(name: str, help_text: str, configure: Callable[[argparse.ArgumentParser], None] | None = None):
    def deco(fn: Handler) -> Handler:
        _COMMANDS[name] = (help_text, fn, configure or (lambda p: None))
        return fn

    return deco


@command("info", "Show resolved paths and settings")
def _info(args: argparse.Namespace) -> int:
    from gridsight import REGIONS, __version__
    from gridsight.config import get_settings, project_root

    s = get_settings()
    print(f"GridSight {__version__}")
    print(f"project root : {project_root()}")
    print(f"data dir     : {s.data_dir}")
    print(f"reports dir  : {s.reports_dir}")
    print(f"NEMweb       : {s.nemweb['base_url']}")
    print(f"regions      : {', '.join(REGIONS)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    # Import modules that register extra commands.
    from gridsight import commands  # noqa: F401

    parser = argparse.ArgumentParser(prog="gridsight", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (help_text, fn, configure) in sorted(_COMMANDS.items()):
        p = sub.add_parser(name, help=help_text)
        configure(p)
        p.set_defaults(_handler=fn)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return int(args._handler(args))


if __name__ == "__main__":
    sys.exit(main())
