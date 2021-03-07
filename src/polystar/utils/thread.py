from threading import Thread
from typing import List


class MyThread(Thread):
    THREADS: List["MyThread"] = []

    def __init__(self):
        super().__init__()
        self.running = False
        self.THREADS.append(self)

    def run(self) -> None:
        self.running = True
        self.loop()
        self.running = False

    def loop(self):
        while self.running:
            self.step()

    def step(self):
        pass

    def stop(self):
        self.running = False

    @staticmethod
    def close():
        for thread in MyThread.THREADS:
            thread.stop()
