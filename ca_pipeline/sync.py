"""Keep training state off the molab machine.

molab sessions end after at most 12 hours (and GPU sandboxes have been seen
reclaimed much sooner), and only ~1 GB of a notebook's files persist. So the
dataset and checkpoints live in a remote store (a private Hugging Face repo by
default), every session pulls them on start, and new checkpoints are pushed
every few minutes while training runs.
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import Protocol

CHECKPOINT_PATTERNS = ("*.state", "*.safetensors", "*.pth")


class Remote(Protocol):
    def download(self, prefix: str, dest: Path) -> None: ...
    def upload(self, files: list[Path], base: Path, prefix: str) -> None: ...
    def upload_folder(self, folder: Path, prefix: str) -> None: ...


class LocalRemote:
    """A directory standing in for remote storage (tests, or a mounted drive)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def download(self, prefix: str, dest: Path) -> None:
        src = self.root / prefix
        if src.exists():
            shutil.copytree(src, dest, dirs_exist_ok=True)

    def upload(self, files: list[Path], base: Path, prefix: str) -> None:
        for f in files:
            target = self.root / prefix / f.relative_to(base)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)

    def upload_folder(self, folder: Path, prefix: str) -> None:
        shutil.copytree(folder, self.root / prefix, dirs_exist_ok=True)


class HFRemote:
    """A private Hugging Face repo (dataset type), via ``huggingface_hub``.

    Needs an ``HF_TOKEN`` with write access set as a molab secret/env var.
    """

    def __init__(self, repo_id: str, token: str | None = None):
        from huggingface_hub import HfApi

        self.api = HfApi(token=token)
        self.repo_id = repo_id
        self.api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    def download(self, prefix: str, dest: Path) -> None:
        from huggingface_hub import snapshot_download

        tmp = dest.parent / f".{dest.name}.download"
        snapshot_download(self.repo_id, repo_type="dataset", allow_patterns=[f"{prefix}/**"],
                          local_dir=tmp, token=self.api.token)
        if (tmp / prefix).exists():
            shutil.copytree(tmp / prefix, dest, dirs_exist_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)

    def upload(self, files: list[Path], base: Path, prefix: str) -> None:
        from huggingface_hub import CommitOperationAdd

        ops = [CommitOperationAdd(f"{prefix}/{f.relative_to(base).as_posix()}", str(f)) for f in files]
        if ops:
            self.api.create_commit(self.repo_id, ops, repo_type="dataset",
                                   commit_message=f"sync {len(ops)} file(s)")

    def upload_folder(self, folder: Path, prefix: str) -> None:
        # upload_large_folder is resumable and chunks tens of thousands of files.
        staging = folder.parent / f".{folder.name}.upload"
        shutil.rmtree(staging, ignore_errors=True)
        (staging / prefix).parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(folder, staging / prefix)
        self.api.upload_large_folder(self.repo_id, staging, repo_type="dataset")
        shutil.rmtree(staging, ignore_errors=True)


def newest_checkpoints(run_dir: Path, keep: int = 2) -> list[Path]:
    """The ``keep`` newest files per checkpoint type, so uploads stay small."""
    picked: list[Path] = []
    for pattern in CHECKPOINT_PATTERNS:
        files = sorted(run_dir.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        picked += files[:keep]
    return picked


class CheckpointSync:
    """Push new checkpoints in the background every ``interval`` seconds."""

    def __init__(self, remote: Remote, run_dir: Path, prefix: str = "runs/hat-l",
                 interval: float = 600, keep: int = 2):
        self.remote, self.run_dir, self.prefix = remote, Path(run_dir), prefix
        self.interval, self.keep = interval, keep
        self._seen: dict[Path, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: Exception | None = None

    def pull(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.remote.download(self.prefix, self.run_dir)
        self._seen = {p: p.stat().st_mtime for p in newest_checkpoints(self.run_dir, self.keep)}

    def push_new(self) -> list[Path]:
        new = [p for p in newest_checkpoints(self.run_dir, self.keep)
               if self._seen.get(p) != p.stat().st_mtime]
        if new:
            self.remote.upload(new, self.run_dir, self.prefix)
            self._seen.update({p: p.stat().st_mtime for p in new})
        return new

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.push_new()
            except Exception as e:  # network blips must not kill training
                self.last_error = e

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[Path]:
        self._stop.set()
        if self._thread:
            self._thread.join()
        return self.push_new()


def session_deadline(start: float | None = None, max_hours: float = 12.0,
                     margin_minutes: float = 45.0) -> float:
    """When to stop training so the final checkpoint uploads before shutdown."""
    return (start if start is not None else time.time()) + max_hours * 3600 - margin_minutes * 60
