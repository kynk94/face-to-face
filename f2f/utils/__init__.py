import hashlib
import os
import re
import shutil
import sys
import tempfile
from typing import Optional, cast
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import tqdm

HASH_REGEX = re.compile(r"-([a-f0-9]*)\.")

ENV_TORCH_HOME = "TORCH_HOME"
ENV_ONNX_HOME = "ONNX_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"
READ_DATA_CHUNK = 8192


def get_cache_dir() -> str:
    cache_home = os.path.expanduser(
        os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR)
    )
    if not os.path.exists(cache_home):
        os.makedirs(cache_home, exist_ok=True)
    return cache_home


def get_torch_cache_dir() -> str:
    """
    Return `os.path.join(torch.hub.get_dir(), "checkpoints")` without torch
    dependency.
    """
    torch_home = os.path.expanduser(
        os.getenv(ENV_TORCH_HOME, os.path.join(get_cache_dir(), "torch"))
    )
    model_dir = os.path.join(torch_home, "hub", "checkpoints")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    return model_dir


def get_onnx_cache_dir() -> str:
    onnx_home = os.path.expanduser(
        os.getenv(ENV_ONNX_HOME, os.path.join(get_cache_dir(), "onnx"))
    )
    if not os.path.exists(onnx_home):
        os.makedirs(onnx_home, exist_ok=True)
    return onnx_home


def encode_file(path: str, algorithm: str = "sha256") -> str:
    """
    Encode a file with a given hashing algorithm.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found.")
    algorithm = algorithm.lower()
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Algorithm {algorithm} not available in hashlib.")
    hash = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(READ_DATA_CHUNK), b""):
            hash.update(byte_block)
    if algorithm == "shake_128":
        return cast(hashlib._VarLenHash, hash).hexdigest(16)
    if algorithm == "shake_256":
        return cast(hashlib._VarLenHash, hash).hexdigest(32)
    return hash.hexdigest()


def assert_hash(path: str, hash: str, algorithm: str = "sha256") -> None:
    """
    Assert the hash of a file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found.")
    encoded_hash = encode_file(path, algorithm)
    if encoded_hash != hash:
        raise RuntimeError(
            'invalid hash value (expected "{}", got "{}")'.format(
                hash, encoded_hash
            )
        )


def rename_file_with_hash(
    path: str,
    algorithm: str = "sha256",
    length: int = 8,
) -> str:
    """
    Rename a file with its hash.
    """
    hash = encode_file(path, algorithm)[:length]
    file_name, file_ext = os.path.splitext(path)
    search = HASH_REGEX.search(file_name)
    exist_hash = search.group(1) if search else None
    if exist_hash == hash:
        return path
    hash_path = f"{file_name}-{hash}{file_ext}"
    shutil.move(path, hash_path)
    return hash_path


def url_to_local_path(path_or_url: str, check_hash: bool = False) -> str:
    """
    Return the local cache path from a given URL.
    Reference to `torch.hub.load_state_dict_from_url`.
    """
    parts = urlparse(path_or_url)

    file_name = os.path.basename(parts.path)
    if check_hash:
        search = re.search(HASH_REGEX, file_name)
        if search is None:
            raise ValueError(f"No hash found in {file_name}")
        hash = cast(str, search.group(1))
    else:
        hash = None

    # If it's a local file, return it directly.
    maybe_local_path = os.path.abspath(os.path.expanduser(parts.path))
    if os.path.exists(maybe_local_path):
        if hash is not None:
            assert_hash(maybe_local_path, hash)
        return maybe_local_path

    if os.path.splitext(file_name)[-1] == ".onnx":
        cache_dir = get_onnx_cache_dir()
    else:
        cache_dir = get_torch_cache_dir()
    cached_file = os.path.join(cache_dir, file_name)
    if os.path.exists(cached_file):
        if hash is not None:
            assert_hash(maybe_local_path, hash)
        return cached_file

    # If not exist, download the file.
    sys.stderr.write(
        'Downloading: "{}" to {}\n'.format(path_or_url, cached_file)
    )
    download_url_to_file(path_or_url, cached_file, hash=hash, progress=True)
    return cached_file


def download_url_to_file(
    url: str, path: str, hash: Optional[str] = None, progress: bool = True
) -> None:
    """
    Download object at the given URL to a local path.
    Reference to `torch.hub.download_url_to_file`.
    """
    request = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(request)  # noqa: S310
    meta = u.info()
    if hasattr(meta, "getheaders"):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    path = os.path.expanduser(path)
    directory = os.path.dirname(path)
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir=directory)

    if hash is not None:
        sha256 = hashlib.sha256()
    try:
        with tqdm.tqdm(
            total=file_size,
            disable=not progress,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            while True:
                buffer = u.read(READ_DATA_CHUNK)
                if len(buffer) == 0:
                    break
                temp_file.write(buffer)
                if hash is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))
        temp_file.close()
        if hash is not None:
            digest = sha256.hexdigest()
            if digest[: len(hash)] != hash:
                raise RuntimeError(
                    'invalid hash value (expected "{}", got "{}")'.format(
                        hash, digest
                    )
                )
        shutil.move(temp_file.name, path)
    finally:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
