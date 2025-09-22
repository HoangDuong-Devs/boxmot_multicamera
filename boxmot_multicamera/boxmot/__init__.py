# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

__version__ = '13.0.17'

from boxmot.tracker_zoo import create_tracker, get_tracker_config
from boxmot.trackers.botsort.botsort_pending_non_idsd import BotSort

TRACKERS = [
    "botsort",
]

__all__ = (
    "__version__",
    "BotSort",
    "create_tracker",
    "get_tracker_config",
)
