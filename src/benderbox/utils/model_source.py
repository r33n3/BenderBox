"""
Model Source Handler for BenderBox

Handles model resolution from multiple sources:
- Local files
- HTTP/HTTPS URLs
- Hugging Face model repositories
"""

import asyncio
import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ModelSource(Enum):
    """Type of model source."""
    LOCAL = "local"
    URL = "url"
    HUGGINGFACE = "huggingface"


@dataclass
class ResolvedModel:
    """Result of model resolution."""
    local_path: Path
    source: ModelSource
    original_target: str
    cached: bool = False
    size_bytes: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DownloadProgress:
    """Progress information for downloads."""
    total_bytes: int
    downloaded_bytes: int
    speed_bytes_per_sec: float
    filename: str

    @property
    def percent(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return (self.downloaded_bytes / self.total_bytes) * 100


class ModelSourceError(Exception):
    """Error during model source resolution."""
    pass


class ModelSourceHandler:
    """
    Handles model resolution from various sources.

    Supports:
    - Local file paths
    - HTTP/HTTPS direct download URLs
    - Hugging Face model IDs (org/model format)
    """

    # Hugging Face model ID pattern: org/model-name or org/model-name/file
    HF_MODEL_PATTERN = re.compile(r'^[\w.-]+/[\w.-]+(?:/[\w.-]+)*$')

    # Common model file extensions
    MODEL_EXTENSIONS = {'.gguf', '.bin', '.safetensors', '.pt', '.pth', '.onnx'}

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        cache_ttl_days: int = 30,
        download_timeout: int = 600,
        max_size_gb: float = 50.0,
    ):
        """
        Initialize ModelSourceHandler.

        Args:
            cache_path: Path for caching downloaded models.
            cache_ttl_days: Days to keep cached models.
            download_timeout: Download timeout in seconds.
            max_size_gb: Maximum download size in GB.
        """
        self.cache_path = Path(cache_path) if cache_path else Path("data/models")
        self.cache_ttl_days = cache_ttl_days
        self.download_timeout = download_timeout
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)

        # Ensure cache directories exist
        self.cache_path.mkdir(parents=True, exist_ok=True)
        (self.cache_path / "huggingface").mkdir(exist_ok=True)
        (self.cache_path / "url").mkdir(exist_ok=True)

    @classmethod
    def from_config(cls, config: Any) -> "ModelSourceHandler":
        """
        Create a ModelSourceHandler from a BenderBox Config object.

        Args:
            config: A BenderBox Config object with storage settings.

        Returns:
            Configured ModelSourceHandler instance.
        """
        storage = getattr(config, 'storage', None)
        if storage:
            return cls(
                cache_path=Path(storage.model_cache_path),
                cache_ttl_days=storage.model_cache_ttl_days,
                download_timeout=storage.download_timeout_seconds,
                max_size_gb=storage.max_download_size_gb,
            )
        # Fallback to defaults if no storage config
        return cls()

    def detect_source(self, target: str) -> ModelSource:
        """
        Detect the source type of a target.

        Args:
            target: Model target string.

        Returns:
            ModelSource enum value.
        """
        target = target.strip()

        # Check for Hugging Face URL first
        if 'huggingface.co' in target.lower():
            return ModelSource.HUGGINGFACE

        # Check for HTTP/HTTPS URL
        if target.startswith(('http://', 'https://')):
            return ModelSource.URL

        # Check if local path exists (highest priority for existing files)
        if Path(target).exists():
            return ModelSource.LOCAL

        # Check if it looks like a local path pattern (starts with ./, ../, /, or drive letter)
        if (target.startswith(('./', '../', '/')) or
            (len(target) > 2 and target[1] == ':')):  # Windows drive letter
            return ModelSource.LOCAL

        # Check for Hugging Face model ID pattern (org/model)
        # Pattern: org-name/model-name with optional /filename
        # But NOT paths like data/models or src/test
        if '/' in target and target.count('/') <= 2:
            # Exclude common local path patterns
            local_patterns = ['data/', 'src/', 'models/', 'test/', 'docs/', 'lib/', 'bin/']
            if not any(target.lower().startswith(p) for p in local_patterns):
                if self.HF_MODEL_PATTERN.match(target):
                    return ModelSource.HUGGINGFACE

        # Default to local file
        return ModelSource.LOCAL

    async def resolve(
        self,
        target: str,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> ResolvedModel:
        """
        Resolve a target to a local path.

        Args:
            target: Model target (path, URL, or HF model ID).
            progress_callback: Optional callback for download progress.

        Returns:
            ResolvedModel with local path and metadata.

        Raises:
            ModelSourceError: If resolution fails.
        """
        source = self.detect_source(target)

        if source == ModelSource.LOCAL:
            return await self._resolve_local(target)
        elif source == ModelSource.URL:
            return await self._resolve_url(target, progress_callback)
        elif source == ModelSource.HUGGINGFACE:
            return await self._resolve_huggingface(target, progress_callback)
        else:
            raise ModelSourceError(f"Unknown source type for: {target}")

    async def _resolve_local(self, target: str) -> ResolvedModel:
        """Resolve a local file path."""
        path = Path(target).resolve()

        if not path.exists():
            raise ModelSourceError(f"Local file not found: {target}")

        size = path.stat().st_size if path.is_file() else None

        return ResolvedModel(
            local_path=path,
            source=ModelSource.LOCAL,
            original_target=target,
            cached=False,
            size_bytes=size,
        )

    async def _resolve_url(
        self,
        url: str,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> ResolvedModel:
        """Resolve and download from a URL."""
        # Check cache first
        cache_dir = self._get_url_cache_dir(url)
        cached_file = self._find_cached_model(cache_dir)

        if cached_file:
            logger.info(f"Using cached model: {cached_file}")
            return ResolvedModel(
                local_path=cached_file,
                source=ModelSource.URL,
                original_target=url,
                cached=True,
                size_bytes=cached_file.stat().st_size,
                metadata=self._load_metadata(cache_dir),
            )

        # Download the file
        cache_dir.mkdir(parents=True, exist_ok=True)
        filename = self._extract_filename_from_url(url)
        dest_path = cache_dir / filename

        try:
            await self._download_file(url, dest_path, progress_callback)
        except Exception as e:
            # Clean up on failure
            if dest_path.exists():
                dest_path.unlink()
            raise ModelSourceError(f"Download failed: {e}") from e

        # Save metadata
        metadata = {
            "source": "url",
            "original_url": url,
            "download_time": datetime.now().isoformat(),
            "size_bytes": dest_path.stat().st_size,
        }
        self._save_metadata(cache_dir, metadata)

        return ResolvedModel(
            local_path=dest_path,
            source=ModelSource.URL,
            original_target=url,
            cached=False,
            size_bytes=dest_path.stat().st_size,
            metadata=metadata,
        )

    async def _resolve_huggingface(
        self,
        target: str,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> ResolvedModel:
        """Resolve and download from Hugging Face."""
        # Parse Hugging Face target
        model_id, filename = self._parse_hf_target(target)

        # Check cache first
        cache_dir = self._get_hf_cache_dir(model_id)
        if filename:
            cached_file = cache_dir / filename
            if cached_file.exists():
                logger.info(f"Using cached HF model: {cached_file}")
                return ResolvedModel(
                    local_path=cached_file,
                    source=ModelSource.HUGGINGFACE,
                    original_target=target,
                    cached=True,
                    size_bytes=cached_file.stat().st_size,
                    metadata=self._load_metadata(cache_dir),
                )
        else:
            # Look for any cached model file
            cached_file = self._find_cached_model(cache_dir)
            if cached_file:
                logger.info(f"Using cached HF model: {cached_file}")
                return ResolvedModel(
                    local_path=cached_file,
                    source=ModelSource.HUGGINGFACE,
                    original_target=target,
                    cached=True,
                    size_bytes=cached_file.stat().st_size,
                    metadata=self._load_metadata(cache_dir),
                )

        # Download from Hugging Face
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            downloaded_path = await self._download_from_hf(
                model_id, filename, cache_dir, progress_callback
            )
        except Exception as e:
            raise ModelSourceError(f"Hugging Face download failed: {e}") from e

        # Save metadata
        metadata = {
            "source": "huggingface",
            "model_id": model_id,
            "filename": filename,
            "download_time": datetime.now().isoformat(),
            "size_bytes": downloaded_path.stat().st_size,
        }
        self._save_metadata(cache_dir, metadata)

        return ResolvedModel(
            local_path=downloaded_path,
            source=ModelSource.HUGGINGFACE,
            original_target=target,
            cached=False,
            size_bytes=downloaded_path.stat().st_size,
            metadata=metadata,
        )

    def _parse_hf_target(self, target: str) -> tuple[str, Optional[str]]:
        """Parse Hugging Face target into model_id and optional filename."""
        # Handle full HF URLs
        if 'huggingface.co' in target:
            parsed = urlparse(target)
            path_parts = parsed.path.strip('/').split('/')

            if len(path_parts) >= 2:
                model_id = f"{path_parts[0]}/{path_parts[1]}"

                # Check for specific file in URL
                if 'resolve' in path_parts:
                    # URL format: /org/model/resolve/branch/filename
                    resolve_idx = path_parts.index('resolve')
                    if len(path_parts) > resolve_idx + 2:
                        filename = path_parts[-1]
                        return model_id, filename
                elif 'blob' in path_parts:
                    # URL format: /org/model/blob/branch/filename
                    blob_idx = path_parts.index('blob')
                    if len(path_parts) > blob_idx + 2:
                        filename = path_parts[-1]
                        return model_id, filename

                return model_id, None

        # Handle model ID format: org/model or org/model/filename
        parts = target.split('/')
        if len(parts) >= 2:
            model_id = f"{parts[0]}/{parts[1]}"
            filename = parts[2] if len(parts) > 2 else None
            return model_id, filename

        raise ModelSourceError(f"Invalid Hugging Face target: {target}")

    async def _download_file(
        self,
        url: str,
        dest: Path,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> None:
        """Download a file from URL with progress tracking."""
        try:
            import httpx
        except ImportError:
            raise ModelSourceError(
                "httpx is required for URL downloads. Install with: pip install httpx"
            )

        async with httpx.AsyncClient(timeout=self.download_timeout, follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))

                if total_size > self.max_size_bytes:
                    raise ModelSourceError(
                        f"File too large: {total_size / (1024**3):.1f}GB "
                        f"(max: {self.max_size_bytes / (1024**3):.1f}GB)"
                    )

                downloaded = 0
                start_time = asyncio.get_event_loop().time()

                with open(dest, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback:
                            elapsed = asyncio.get_event_loop().time() - start_time
                            speed = downloaded / elapsed if elapsed > 0 else 0

                            progress_callback(DownloadProgress(
                                total_bytes=total_size,
                                downloaded_bytes=downloaded,
                                speed_bytes_per_sec=speed,
                                filename=dest.name,
                            ))

    async def _download_from_hf(
        self,
        model_id: str,
        filename: Optional[str],
        cache_dir: Path,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> Path:
        """Download from Hugging Face using huggingface_hub."""
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError:
            raise ModelSourceError(
                "huggingface_hub is required for HF downloads. "
                "Install with: pip install huggingface-hub"
            )

        # If no specific filename, find a model file
        if not filename:
            try:
                files = list_repo_files(model_id)
                model_files = [
                    f for f in files
                    if any(f.endswith(ext) for ext in self.MODEL_EXTENSIONS)
                ]

                if not model_files:
                    raise ModelSourceError(
                        f"No model files found in {model_id}. "
                        f"Supported extensions: {self.MODEL_EXTENSIONS}"
                    )

                # Prefer GGUF files, then pick the first
                gguf_files = [f for f in model_files if f.endswith('.gguf')]
                filename = gguf_files[0] if gguf_files else model_files[0]
                logger.info(f"Auto-selected model file: {filename}")

            except Exception as e:
                raise ModelSourceError(f"Failed to list HF repo files: {e}") from e

        # Download the file
        def hf_progress_callback(progress: float):
            if progress_callback:
                # HF hub gives progress as fraction 0-1
                progress_callback(DownloadProgress(
                    total_bytes=100,
                    downloaded_bytes=int(progress * 100),
                    speed_bytes_per_sec=0,
                    filename=filename or "model",
                ))

        try:
            # Run in executor since hf_hub_download is synchronous
            loop = asyncio.get_event_loop()
            downloaded_path = await loop.run_in_executor(
                None,
                lambda: hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    local_dir=str(cache_dir),
                    local_dir_use_symlinks=False,
                )
            )
            return Path(downloaded_path)

        except Exception as e:
            raise ModelSourceError(f"HF download failed: {e}") from e

    def _get_url_cache_dir(self, url: str) -> Path:
        """Get cache directory for a URL."""
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        return self.cache_path / "url" / url_hash

    def _get_hf_cache_dir(self, model_id: str) -> Path:
        """Get cache directory for a Hugging Face model."""
        safe_id = model_id.replace("/", "--")
        return self.cache_path / "huggingface" / safe_id

    def _extract_filename_from_url(self, url: str) -> str:
        """Extract filename from URL."""
        parsed = urlparse(url)
        path = parsed.path

        # Get the last path component
        filename = path.split("/")[-1] if "/" in path else path

        # Clean up query strings
        if "?" in filename:
            filename = filename.split("?")[0]

        # Default filename if none found
        if not filename or filename == "":
            filename = "model.bin"

        return filename

    def _find_cached_model(self, cache_dir: Path) -> Optional[Path]:
        """Find a cached model file in a directory."""
        if not cache_dir.exists():
            return None

        # Check if cache is expired
        metadata = self._load_metadata(cache_dir)
        if metadata:
            download_time = metadata.get("download_time")
            if download_time:
                try:
                    dt = datetime.fromisoformat(download_time)
                    if datetime.now() - dt > timedelta(days=self.cache_ttl_days):
                        logger.info(f"Cache expired for {cache_dir}")
                        return None
                except (ValueError, TypeError):
                    pass

        # Look for model files
        for ext in self.MODEL_EXTENSIONS:
            for path in cache_dir.glob(f"**/*{ext}"):
                if path.is_file():
                    return path

        return None

    def _load_metadata(self, cache_dir: Path) -> Dict[str, Any]:
        """Load metadata from cache directory."""
        metadata_file = cache_dir / ".metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _save_metadata(self, cache_dir: Path, metadata: Dict[str, Any]) -> None:
        """Save metadata to cache directory."""
        metadata_file = cache_dir / ".metadata.json"
        try:
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to save metadata: {e}")

    def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear cached models.

        Args:
            older_than_days: Only clear items older than this. None clears all.

        Returns:
            Number of items cleared.
        """
        cleared = 0
        cutoff = None
        if older_than_days is not None:
            cutoff = datetime.now() - timedelta(days=older_than_days)

        for subdir in ["url", "huggingface"]:
            cache_subdir = self.cache_path / subdir
            if not cache_subdir.exists():
                continue

            for item_dir in cache_subdir.iterdir():
                if not item_dir.is_dir():
                    continue

                should_clear = True
                if cutoff:
                    metadata = self._load_metadata(item_dir)
                    download_time = metadata.get("download_time")
                    if download_time:
                        try:
                            dt = datetime.fromisoformat(download_time)
                            should_clear = dt < cutoff
                        except (ValueError, TypeError):
                            pass

                if should_clear:
                    try:
                        shutil.rmtree(item_dir)
                        cleared += 1
                        logger.info(f"Cleared cache: {item_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to clear {item_dir}: {e}")

        return cleared

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache."""
        total_size = 0
        item_count = 0
        items = []

        for subdir in ["url", "huggingface"]:
            cache_subdir = self.cache_path / subdir
            if not cache_subdir.exists():
                continue

            for item_dir in cache_subdir.iterdir():
                if not item_dir.is_dir():
                    continue

                metadata = self._load_metadata(item_dir)
                size = sum(f.stat().st_size for f in item_dir.rglob("*") if f.is_file())
                total_size += size
                item_count += 1

                items.append({
                    "path": str(item_dir),
                    "source": subdir,
                    "size_bytes": size,
                    "download_time": metadata.get("download_time"),
                    "original_target": metadata.get("original_url") or metadata.get("model_id"),
                })

        return {
            "cache_path": str(self.cache_path),
            "total_size_bytes": total_size,
            "total_size_gb": total_size / (1024 ** 3),
            "item_count": item_count,
            "ttl_days": self.cache_ttl_days,
            "items": items,
        }
