"""Interactive server shutdown: Ctrl+C requires a simple confirmation."""
import builtins
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from app import serve_with_interrupt_prompt


class FakeServer:
    def __init__(self, always_interrupt=False):
        self.calls = 0
        self.always_interrupt = always_interrupt

    def serve_forever(self):
        self.calls += 1
        if self.always_interrupt or self.calls == 1:
            raise KeyboardInterrupt


@pytest.mark.parametrize("answer", ["y", "Y", "yes", ""])
def test_interrupt_accepts_yes_or_blank(answer, monkeypatch):
    server = FakeServer()
    prompts = []
    monkeypatch.setattr(builtins, "input",
                        lambda prompt: prompts.append(prompt) or answer)
    serve_with_interrupt_prompt(server)
    assert server.calls == 1
    assert len(prompts) == 1 and "[Y/n]" in prompts[0]


def test_no_resumes_and_next_interrupt_can_stop(monkeypatch):
    server = FakeServer(always_interrupt=True)
    answers = iter(["n", "y"])
    monkeypatch.setattr(builtins, "input", lambda prompt: next(answers))
    serve_with_interrupt_prompt(server)
    assert server.calls == 2


def test_eof_during_prompt_stops_instead_of_trapping_terminal(monkeypatch):
    server = FakeServer()
    def no_input(prompt):
        raise EOFError
    monkeypatch.setattr(builtins, "input", no_input)
    serve_with_interrupt_prompt(server)
    assert server.calls == 1
