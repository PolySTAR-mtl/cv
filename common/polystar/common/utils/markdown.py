from pathlib import Path
from typing import Any, Iterable, TextIO

from markdown.core import markdown
from matplotlib.figure import Figure
from pandas import DataFrame
from tabulate import tabulate
from xhtml2pdf.document import pisaDocument

from polystar.common.utils.working_directory import working_directory


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
        self.paragraph(f"![{alt}]({str(relative_path).replace(' ', '%20')})")
        return self

    def figure(self, figure: Figure, name: str, alt: str = "img"):
        name = name.replace(" ", "_")
        figure.savefig(self.markdown_path.parent / name)
        return self.image(name, alt)

    def table(self, data: DataFrame) -> "MarkdownFile":
        self.file.write(tabulate(data, tablefmt="pipe", headers="keys").replace(".0 ", "   "))
        self.file.write("\n\n")
        return self


def markdown_to_pdf(markdown_path: Path):
    html_text = markdown(markdown_path.read_text(), output_format="html", extensions=["markdown.extensions.tables"])
    html_text += """<style>
    td, th { 
        border: 1px solid #666666; 
        text-align:center;
    }
    td, th {
        padding-top:4px;
    }
    tr:nth-child(odd) {
        background-color: red;
    }
    th {
        width: 50%;
    }
    </style>"""
    for b in ["td", "th"]:
        for p in ["left", "right"]:
            html_text = html_text.replace(f"""<{b} align="{p}">""", "<td>")
    with markdown_path.with_suffix(".pdf").open("wb") as f, working_directory(markdown_path.parent):
        pisaDocument(html_text, dest=f)
