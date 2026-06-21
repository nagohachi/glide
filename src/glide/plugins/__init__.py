"""Built-in plugins (reward functions) + the plugin-loading entry point.

Importing this package registers the built-in reward functions. User plugins are
loaded from ``config.plugins`` via :func:`glide.registry.load_plugins`.
"""

from . import rewards  # noqa: F401  (import side effect: registers built-in rewards)
from ..registry import load_plugins

__all__ = ["load_plugins", "rewards"]
