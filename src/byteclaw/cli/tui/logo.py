"""Animated Rich logo used by the ByteClaw Textual interface."""

from rich.text import Text
from textual.timer import Timer
from textual.widgets import Static


LOGO_TITLE = "\n".join(
    (
        " █▄█   █▄█ ",
        "▄▄▀▀███▀▀▄▄",
        "▀█████████▀",
        "▀ ▀ ▀ ▀ ▀ ▀",
    )
)
_ANIMATION_STYLES = (
    "bold #d0b000",
    "bold #d8bb1f",
    "bold #e0c43a",
    "bold #e8d266",
    "bold #e0c43a",
    "bold #d8bb1f",
    "bold #d0b000",
)


def render_logo(frame: int = 0) -> Text:
    """Render one colored animation frame as a Rich Text object."""

    title_style = _ANIMATION_STYLES[frame % len(_ANIMATION_STYLES)]
    logo = Text(justify="center")
    logo.append(LOGO_TITLE, style=title_style)
    return logo


class ByteClawLogo(Static):
    """Display a brief startup color pulse and then a stable Rich logo."""

    DEFAULT_CSS = """
    ByteClawLogo {
        width: 13;
        height: 4;
        padding: 0 1;
        content-align: center middle;
        background: #101010;
    }
    """

    ANIMATION_INTERVAL = 0.14

    def __init__(self, *, animate: bool = True, id: str | None = None) -> None:
        super().__init__(render_logo(), id=id)
        self.animate = animate
        self.animation_frame = 0
        self._animation_timer: Timer | None = None

    def on_mount(self) -> None:
        if self.animate:
            self._animation_timer = self.set_interval(
                self.ANIMATION_INTERVAL,
                self._advance_animation,
            )

    def on_unmount(self) -> None:
        if self._animation_timer is not None:
            self._animation_timer.stop()

    def _advance_animation(self) -> None:
        self.animation_frame += 1
        if self.animation_frame >= len(_ANIMATION_STYLES):
            if self._animation_timer is not None:
                self._animation_timer.stop()
            self.animation_frame = 0
            self.update(render_logo())
            return
        self.update(render_logo(self.animation_frame))
