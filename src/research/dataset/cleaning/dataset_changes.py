import json
import logging
from pathlib import Path
from typing import Dict, List, Set

from more_itertools import flatten

from polystar.utils.git import get_git_username
from polystar.utils.time import create_time_id
from research.common.gcloud.gcloud_storage import GCStorages

INVALIDATED_KEY: str = "invalidated"
logger = logging.getLogger(__name__)


class DatasetChanges:
    def __init__(self, dataset_directory: Path):
        self.changes_file: Path = dataset_directory / ".changes"
        try:
            GCStorages.DEV.download_file_if_missing(self.changes_file)
        except FileNotFoundError:
            self.changes_file.write_text("{}")
        except ConnectionError:
            logger.warning(f"Can't load {self.changes_file} because of no internet connection")

    @property
    def invalidated(self) -> Set[str]:
        return set(flatten(self.changes[INVALIDATED_KEY].values()))

    @property
    def changes(self) -> Dict:
        changes = json.loads(self.changes_file.read_text()) if self.changes_file.exists() else {}
        changes.setdefault(INVALIDATED_KEY, {})
        return changes

    def invalidate(self, names: List[str]):
        entry_id = f"{create_time_id()} ({get_git_username()})"
        changes = self.changes
        changes[INVALIDATED_KEY][entry_id] = names
        self.changes_file.write_text(json.dumps(changes, indent=2))
        print(f"changes saved, see entry {entry_id} in file://{self.changes_file}")
        self.upload()

    def upload(self):
        GCStorages.DEV.upload_file(self.changes_file)
