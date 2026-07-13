"""Animated Rich logo used by the ByteClaw Textual interface."""

from rich.text import Text
from textual.timer import Timer
from textual.widgets import Static


LOGO_TITLE = " 🐾 ByteClaw"
LOGO_RULE = " ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
LOGO_STAGE = "  Stage 6 · MultiAgent + Context/Harness"

_ANIMATION_STYLES = (
    ("bold bright_cyan", "cyan", "bright_white"),
    ("bold bright_blue", "bright_cyan", "white"),
    ("bold bright_magenta", "bright_blue", "bright_cyan"),
    ("bold bright_white", "bright_magenta", "bright_blue"),
    ("bold bright_magenta", "bright_blue", "bright_cyan"),
    ("bold bright_blue", "bright_cyan", "white"),
    ("bold bright_cyan", "cyan", "bright_white"),
)


def render_logo(frame: int = 0) -> Text:
    """Render one colored animation frame as a Rich Text object."""

    title_style, rule_style, stage_style = _ANIMATION_STYLES[
        frame % len(_ANIMATION_STYLES)
    ]
    logo = Text(justify="center")
    logo.append(f"{LOGO_TITLE}\n", style=title_style)
    logo.append(f"{LOGO_RULE}\n", style=rule_style)
    logo.append(f"{LOGO_STAGE}\n", style=stage_style)
    logo.append(LOGO_RULE, style=rule_style)
    return logo


class ByteClawLogo(Static):
    """Display a brief startup color pulse and then a stable Rich logo."""

    DEFAULT_CSS = """
    ByteClawLogo {
        width: 100%;
        height: 5;
        padding: 0 1;
        content-align: center middle;
        background: $background;
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
