import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from avtv.models import Block, Candidate, Selection, VisualBrief


class RunDir:
    BLOCKS = "01_blocks.json"
    BRIEFS = "02_visual_briefs.json"
    SEARCH = "03_search_results.json"
    SELECTIONS = "04_selections.json"

    def __init__(self, base_dir: Path, run_id: str) -> None:
        self.base_dir = Path(base_dir)
        self.run_id = run_id

    @classmethod
    def new(cls, base_dir: Path) -> "RunDir":
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        return cls(base_dir=base_dir, run_id=run_id)

    @property
    def path(self) -> Path:
        return self.base_dir / self.run_id

    @property
    def logs_path(self) -> Path:
        return self.path / "logs"

    def ensure(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)

    def _save_models(self, name: str, items: Sequence[BaseModel]) -> None:
        out = [m.model_dump() for m in items]
        (self.path / name).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _load_models(self, name: str, model_cls: type[BaseModel]) -> list[Any]:
        raw = json.loads((self.path / name).read_text(encoding="utf-8"))
        return [model_cls.model_validate(item) for item in raw]

    def save_blocks(self, blocks: list[Block]) -> None:
        self._save_models(self.BLOCKS, blocks)

    def load_blocks(self) -> list[Block]:
        return self._load_models(self.BLOCKS, Block)

    def save_briefs(self, briefs: list[VisualBrief]) -> None:
        self._save_models(self.BRIEFS, briefs)

    def load_briefs(self) -> list[VisualBrief]:
        return self._load_models(self.BRIEFS, VisualBrief)

    def save_search_results(self, results: dict[int, list[Candidate]]) -> None:
        out = {str(k): [c.model_dump() for c in v] for k, v in results.items()}
        (self.path / self.SEARCH).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def load_search_results(self) -> dict[int, list[Candidate]]:
        raw = json.loads((self.path / self.SEARCH).read_text(encoding="utf-8"))
        return {int(k): [Candidate.model_validate(c) for c in v] for k, v in raw.items()}

    def save_selections(self, selections: list[Selection]) -> None:
        self._save_models(self.SELECTIONS, selections)

    def load_selections(self) -> list[Selection]:
        return self._load_models(self.SELECTIONS, Selection)

    def has(self, name: str) -> bool:
        return (self.path / name).exists()
