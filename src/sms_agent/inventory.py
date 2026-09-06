import os
import hashlib
from datetime import datetime, timezone
from typing import List, Dict
from pathlib import Path
from .models import FileMetadata

class InventoryCollector:
    """Collects metadata from a local directory to represent file inventory."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)

    def _hash_file(self, filepath: Path, chunk_size: int = 8192) -> str:
        """Compute SHA-256 hash of a file's content deterministically."""
        hasher = hashlib.sha256()
        try:
            with open(filepath, 'rb') as f:
                while chunk := f.read(chunk_size):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""

    def collect(self) -> List[FileMetadata]:
        """Walk the directory and collect FileMetadata for all files."""
        if not self.root_dir.exists() or not self.root_dir.is_dir():
            return []

        inventory: List[FileMetadata] = []
        hash_registry: Dict[str, bool] = {} # Tracks if a hash has been seen multiple times

        # First pass: collect all files, their basic metadata, and compute hashes
        for root, dirs, files in os.walk(self.root_dir):
            for file_name in files:
                file_path = Path(root) / file_name
                
                try:
                    stat = file_path.stat()
                except OSError:
                    continue # Skip broken symlinks or unreadable files

                rel_key = file_path.relative_to(self.root_dir).as_posix()
                content_hash = self._hash_file(file_path)
                
                # Determine timezone-aware timestamps
                created_at = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
                last_modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                last_accessed_at = datetime.fromtimestamp(stat.st_atime, tz=timezone.utc)

                meta = FileMetadata(
                    key=rel_key,
                    filename=file_path.name,
                    extension=file_path.suffix.lower() if file_path.suffix else "",
                    size_bytes=stat.st_size,
                    created_at=created_at,
                    last_modified_at=last_modified_at,
                    last_accessed_at=last_accessed_at,
                    content_hash=content_hash,
                    is_duplicate=False # Will update in second pass
                )
                inventory.append(meta)

                if content_hash:
                    if content_hash in hash_registry:
                        hash_registry[content_hash] = True # Mark as duplicated
                    else:
                        hash_registry[content_hash] = False

        # Second pass: mark duplicates
        for meta in inventory:
            if meta.content_hash and hash_registry.get(meta.content_hash, False):
                meta.is_duplicate = True

        return inventory
