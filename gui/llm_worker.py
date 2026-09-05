"""Background-thread wrapper for src/llm_assist.py calls, same QThread
pattern as data_bridge.py (kept in a separate file/class so the naming
collision that broke the refresh worker -- self.event shadowing
QObject.event() -- can't accidentally happen here either; this worker
avoids any attribute named `event`)."""
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class AskWorker(QObject):
    answered = Signal(str)
    failed = Signal(str)

    def __init__(self, question: str, data: dict):
        super().__init__()
        self.question = question
        self.data = data

    def run(self):
        try:
            import llm_assist
            if not llm_assist.is_available():
                self.failed.emit("Ollama isn't reachable at localhost:11434 -- is it running?")
                return
            answer = llm_assist.ask(self.question, self.data)
            self.answered.emit(answer)
        except Exception as e:
            self.failed.emit(str(e))


def ask_llm(question: str, data: dict, on_answer, on_error):
    thread = QThread()
    worker = AskWorker(question, data)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.answered.connect(on_answer)
    worker.failed.connect(on_error)
    worker.answered.connect(thread.quit)
    worker.failed.connect(thread.quit)
    # both terminal signals must release the worker -- connecting only
    # `answered` leaks an AskWorker every time Ollama errors out
    worker.answered.connect(worker.deleteLater)
    worker.failed.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread, worker
