"""Read one card set's 17Lands public data through the `spells` package.

This is the only module that imports spells. spells downloads the free
17Lands Premier Draft files and MTGJSON card data with `uv run spells add
<set>`, converts them to Parquet, and keeps them in its own data home:
`~/.local/share/spells` unless the environment variable `SPELLS_DATA_HOME`
names another folder. `read_set` never downloads. It raises
`SetNotDownloadedError` with the command to run when a file is missing.
The card statistics are aggregated by spells over every draft and game of
the set, which takes about ten seconds the first time and is cached by
spells under `cache/<set>` in its data home.
"""

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import spells
from spells import cache
from spells.enums import EventType, View

__all__ = ("SetData", "SetNotDownloadedError", "read_set")

_STATISTICS = ["alsa", "ata", "gih_wr", "oh_wr", "gp_wr", "iwd", "pct_gp", "num_gih"]
_PICK_COLUMNS = [
    "draft_id",
    "rank",
    "user_game_win_rate_bucket",
    "event_match_wins",
    "event_match_losses",
    "pack_number",
    "pick_number",
    "pick",
    "pack_card",
    "pool",
]
_GAME_COLUMNS = ["draft_id", "build_index", "won", "deck", "sideboard"]


@dataclass(frozen=True, slots=True)
class SetData:
    """The tables of one card set, as `jace.data`'s loaders take them.

    `attributes` has one row per card, from MTGJSON. `statistics` has one
    row per card with its 17Lands numbers. `picks` has one row per human
    pick and `games` one row per game. The last two are lazy: nothing is
    read from disk until a loader collects them.
    """

    attributes: pl.DataFrame
    statistics: pl.DataFrame
    picks: pl.LazyFrame
    games: pl.LazyFrame


class SetNotDownloadedError(FileNotFoundError):
    def __init__(self, set_code: str, *, absent: Path) -> None:
        super().__init__(f"{absent} is missing; download the set with `uv run spells add {set_code}`")


def read_set(set_code: str) -> SetData:
    files = [
        Path(cache.data_file_path(set_code, View.CARD)),
        Path(cache.data_file_path(set_code, View.DRAFT, EventType.PREMIER)),
        Path(cache.data_file_path(set_code, View.GAME, EventType.PREMIER)),
    ]
    for file in files:
        if not file.is_file():
            raise SetNotDownloadedError(set_code, absent=file)
    return SetData(
        attributes=pl.read_parquet(files[0]),
        statistics=spells.summon(set_code, columns=_STATISTICS),
        picks=spells.lazy_select(set_code, View.DRAFT, _PICK_COLUMNS),
        games=spells.lazy_select(set_code, View.GAME, _GAME_COLUMNS),
    )
