from pathlib import Path
from typing import Any, Iterator, List, Tuple

from kivy.app import App
from kivy.core.window import Window
from kivy.input.providers.hidinput import Keyboard
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar

from polystar.models.image import load_image
from research.common.datasets.image_file_dataset_builder import DirectoryDatasetBuilder
from research.dataset.cleaning.dataset_changes import DatasetChanges


class DatasetCleanerApp(App):
    """
    To use it, simply call the run method, it will launch a GUI (see the armor digits project for a more detailed usage)

    :Example:

    >>> your_dataset_builder: DirectoryDatasetBuilder
    >>> DatasetCleanerApp(your_dataset_builder).run()
    
    """

    def __init__(
        self,
        dataset_builder: DirectoryDatasetBuilder,
        size: Tuple[int, int] = (500, 130),
        validate_key: str = "enter",
        invalidate_key: str = "backspace",
        cancel_key: str = "space",
    ):
        super().__init__()
        Window.size = size
        self.dataset_builder = dataset_builder
        dataset = dataset_builder.build()
        self.cleaning_widget = CleanDatasetWidget(
            iter(dataset),
            dataset_size=len(dataset),
            validate_key=validate_key,
            invalidate_key=invalidate_key,
            cancel_key=cancel_key,
        )
        self.saved = False

    def build(self):
        return self.cleaning_widget

    def on_stop(self):
        if self.saved:
            return
        self.saved = True

        if not self.cleaning_widget.invalidated:
            print("No image invalidated")
            return

        print(f"Invalidating {len(self.cleaning_widget.invalidated)} images (out of {self.cleaning_widget.n-1})")
        DatasetChanges(self.dataset_builder.images_dir).invalidate(self.cleaning_widget.invalidated)


class CleanDatasetWidget(BoxLayout):
    def __init__(
        self,
        iterator: Iterator[Tuple[Path, Any, str]],
        validate_key: str,
        invalidate_key: str,
        cancel_key: str,
        dataset_size: int,
    ):
        super().__init__(orientation="vertical", spacing=5)
        self.cancel_key = cancel_key
        self.invalidate_key = invalidate_key
        self.validate_key = validate_key
        self.iterator = iterator
        self._set_up_keyboard()
        self.invalidated: List[str] = []
        self.n = 0
        self.dataset_size = dataset_size
        self.progress_bar = ProgressBar(max=dataset_size, size_hint=(1, 0.05))
        self._next_image()

    def _set_up_keyboard(self):
        self.keyboard = Window.request_keyboard(None, self, "text")
        self.keyboard.bind(on_key_down=self._on_key_pressed)

    def _on_key_pressed(self, keyboard: Keyboard, keycode: Tuple[int, str], text: str, modifiers: List) -> bool:
        code, key = keycode

        if key == self.validate_key:
            self._next_image()
        elif key == self.invalidate_key:
            self.invalidated.append(self.path.name)
            self._next_image()
        elif key == self.cancel_key:
            if self.invalidated:
                self.invalidated.pop()

        return key != "escape"

    def _next_image(self):
        self.clear_widgets()
        self.n += 1
        try:
            self.path, value, name = next(self.iterator)
            self.add_widget(
                Label(
                    text=f"commands: [escape] close / [{self.validate_key}] validate\n"
                    f"[{self.invalidate_key}] invalidate / [{self.cancel_key}] remove last",
                    size_hint=(1, 0.3),
                )
            )
            self.add_widget(Image(source=str(self.path), allow_stretch=True, size_hint=(1, 0.4)))
            self.add_widget(Label(text=str(value), size_hint=(1, 0.15)))
            h, w, _ = load_image(self.path).shape
            self.add_widget(Label(text=f"{name} ({w}x{h})", size_hint=(1, 0.15),))
            self.add_widget(
                Label(text=f"{self.n} / {self.dataset_size} ({self.n / self.dataset_size:.2%})", size_hint=(1, 0.15),)
            )
            self.progress_bar.value = self.n
            self.add_widget(self.progress_bar)
        except StopIteration:
            App.get_running_app().stop()
