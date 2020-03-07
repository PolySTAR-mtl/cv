from __future__ import annotations

from pathlib import Path
from typing import TextIO, List

from pandas import DataFrame
from tabulate import tabulate


class MarkdownFile:
    def __init__(self, markdown_path: Path):
        self.markdown_path = markdown_path

    def __enter__(self):
        self.file: TextIO = self.markdown_path.open("w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def title(self, text: str, level: int = 1) -> MarkdownFile:
        self.file.write(f'{"#"*level} {text}\n\n')
        return self

    def paragraph(self, text: str) -> MarkdownFile:
        self.file.write(f"{text}\n\n")
        return self

    def list(self, texts: List[str]) -> MarkdownFile:
        for text in texts:
            self.file.write(f" - {text}\n")
        self.file.write("\n")
        return self

    def image(self, relative_path: str, alt: str = "img") -> MarkdownFile:
        self.paragraph(f"![{alt}]({relative_path})")
        return self

    def table(self, data: DataFrame) -> MarkdownFile:
        self.file.write(tabulate(data, tablefmt="pipe", headers="keys"))
        self.file.write("\n\n")
        return self
