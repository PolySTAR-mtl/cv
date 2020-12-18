import logging
import shutil
import tarfile
from contextlib import contextmanager
from enum import Enum
from pathlib import Path, PurePath
from tempfile import TemporaryDirectory
from typing import Iterable, Optional

from google.cloud.storage import Blob, Bucket, Client

from polystar.common.constants import PROJECT_DIR

logger = logging.getLogger(__name__)
EXTENSIONS_TO_EXCLUDE = (".changes",)


class GCStorage:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.client: Optional[Client] = None
        self._bucket: Optional[Bucket] = None
        self.url = f"https://console.cloud.google.com/storage/browser/{bucket_name}"

    def upload_file(self, local_path: Path, remote_path: Optional[PurePath] = None):
        blob = self._make_remote_blob(local_path, remote_path)
        blob.upload_from_filename(str(local_path), timeout=60 * 5)
        logger.info(f"File file:///{local_path} uploaded")

    def upload_directory(self, local_path: Path, extensions_to_exclude: Iterable[str] = EXTENSIONS_TO_EXCLUDE):
        extensions_to_exclude = set(extensions_to_exclude)
        with TemporaryDirectory() as td_name:
            tar_path = (Path(td_name) / local_path.name).with_suffix(".tar.gz")
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(
                    str(local_path),
                    arcname="",
                    exclude=lambda name: any(name.endswith(ext) for ext in extensions_to_exclude),
                )
            return self.upload_file(tar_path, _make_remote_path(local_path.with_suffix(".tar.gz")))

    def download_file(self, local_path: Path, remote_path: Optional[PurePath] = None):
        blob = self._make_remote_blob(local_path, remote_path)
        blob.download_to_filename(str(local_path), timeout=60 * 5)
        logger.info(f"File file:///{local_path} downloaded")

    def download_file_if_missing(self, local_path: Path, remote_path: Optional[PurePath] = None):
        if not local_path.exists():
            self.download_file(local_path, remote_path)

    def download_directory(
        self, local_path: Path,
    ):
        with TemporaryDirectory() as td_name:
            zip_temp = Path(td_name) / f"temp_{local_path.name}.tar.gz"
            self.download_file(zip_temp, _make_remote_path(local_path.with_suffix(".tar.gz")))
            local_path.mkdir(exist_ok=True, parents=True)
            shutil.unpack_archive(zip_temp, local_path, "gztar")

    def download_directory_if_missing(
        self, local_path: Path, extensions_to_exclude: Iterable[str] = EXTENSIONS_TO_EXCLUDE
    ):
        extensions_to_exclude = set(extensions_to_exclude)
        if local_path.exists() and any(
            f.is_file() and not any(f.name.endswith(ext) for ext in extensions_to_exclude)
            for f in local_path.glob("**/*")
        ):
            return
        self.download_directory(local_path)

    @contextmanager
    def open(self, local_path: Path, mode: str):
        if "r" in mode:
            self.download_file_if_missing(local_path)
            with local_path.open(mode) as f:
                yield f
        elif "w" in mode:
            local_path.parent.mkdir(exist_ok=True, parents=True)
            with local_path.open(mode) as f:
                yield f
            self.upload_file(local_path)
        else:
            raise ValueError(f"mode {mode} is not supported")

    def _make_remote_blob(self, local_path: Path, remote_path: Optional[PurePath]) -> Blob:
        if remote_path is None:
            remote_path = _make_remote_path(local_path)

        return self.bucket.blob(str(remote_path), chunk_size=10 * 1024 * 1024)

    @property
    def bucket(self) -> Bucket:
        if self._bucket is not None:
            return self._bucket
        self.client = Client()
        self._bucket = self.client.bucket(self.bucket_name)
        return self._bucket


def _make_remote_path(local_path: Path):
    return local_path.relative_to(PROJECT_DIR)


class GCStorages(GCStorage, Enum):
    DEV = "poly-cv-dev"
    PROD = "poly-cv-prod"
