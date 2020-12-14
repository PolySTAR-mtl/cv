from pathlib import Path
from typing import Any, Iterable, TextIO

from matplotlib.figure import Figure
from pandas import DataFrame
from tabulate import tabulate


class MarkdownFile:
    def __init__(self, markdown_path: Path):
        self.markdown_path = markdown_path

    def __enter__(self):
        self.markdown_path.parent.mkdir(exist_ok=True, parents=True)
        self.file: TextIO = self.markdown_path.open("w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def title(self, text: Any, level: int = 1) -> "MarkdownFile":
        self.file.write(f'{"#"*level} {text}\n\n')
        return self

    def paragraph(self, text: Any) -> "MarkdownFile":
        self.file.write(f"{text}\n\n")
        return self

    def list(self, texts: Iterable[Any]) -> "MarkdownFile":
        for text in texts:
            self.file.write(f" - {text}\n")
        self.file.write("\n")
        return self

    def image(self, relative_path: str, alt: str = "img") -> "MarkdownFile":
        self.paragraph(f"![{alt}]({relative_path})")
        return self

    def figure(self, figure: Figure, name: str, alt: str = "img"):
        figure.savefig(self.markdown_path.parent / name)
        return self.image(name, alt)

    def table(self, data: DataFrame) -> "MarkdownFile":
        self.file.write(tabulate(data, tablefmt="pipe", headers="keys").replace(".0 ", "   "))
        self.file.write("\n\n")
        return self
