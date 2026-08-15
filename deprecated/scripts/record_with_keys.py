"""lerobot-record wrapper that reads n/r/q from stdin instead of pynput.
Works in any terminal — no X11 focus needed.

  n  advance phase (end episode, or end reset window)
  r  redo current episode
  q  stop and finalize
"""
from __future__ import annotations

import select
import sys
import termios
import threading
import tty


KEY_TO_ACTION = {"n": "advance", "r": "redo", "q": "stop"}


class StdinKeyListener:
    def __init__(self, events: dict[str, bool]):
        self.events = events
        self._stop = threading.Event()
        self._old_termios: list | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if not sys.stdin.isatty():
            print("[stdin-keys] stdin is not a TTY; manual advancement disabled.")
            return
        fd = sys.stdin.fileno()
        self._old_termios = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._thread.start()
        print("[stdin-keys] n=next, r=redo, q=stop")

    def stop(self) -> None:
        self._stop.set()
        if self._old_termios is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_termios)
            self._old_termios = None

    def join(self, *_args, **_kwargs) -> None:
        return None

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                if not select.select([sys.stdin], [], [], 0.1)[0]:
                    continue
                action = KEY_TO_ACTION.get(sys.stdin.read(1).lower())
                if action == "advance":
                    self.events["exit_early"] = True
                elif action == "redo":
                    self.events["rerecord_episode"] = True
                    self.events["exit_early"] = True
                elif action == "stop":
                    self.events["stop_recording"] = True
                    self.events["exit_early"] = True
        finally:
            if self._old_termios is not None:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_termios)


def patched_init_keyboard_listener():
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    listener = StdinKeyListener(events)
    listener.start()
    return listener, events


def main() -> int:
    # lerobot_record imports init_keyboard_listener by name, so patch it on
    # that module rather than on its source.
    import lerobot.scripts.lerobot_record as lr_record
    lr_record.init_keyboard_listener = patched_init_keyboard_listener
    return lr_record.main() or 0


if __name__ == "__main__":
    sys.exit(main())
