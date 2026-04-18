"""
OpenClaw V7.5 — Entry point proxy

Delegates entirely to bot.py so the systemd unit can use either:
    ExecStart=.../python main.py
    ExecStart=.../python bot.py
"""
import runpy, pathlib, sys

# Make sure the project root is on the path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

runpy.run_path(
    str(pathlib.Path(__file__).parent / "bot.py"),
    run_name="__main__",
)
