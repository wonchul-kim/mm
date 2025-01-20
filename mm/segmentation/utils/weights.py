import base64
import errno
import hashlib
import os
import re
import shutil
import sys
import tempfile
from urllib.request import Request, urlopen

import torch
from dotenv import load_dotenv
from tqdm.auto import tqdm

from ..utils.file_manager import _is_legacy_zip_format, _legacy_zip_load, get_dir

load_dotenv()
NEXUS_WEIGHTS_URL = os.getenv(
    "NEXUS_WEIGHTS_URL", default="https://aivdl.nexus.aiv.ai/repository/aiv-weights"
)
NEXUS_USER = os.getenv("NEXUS_USER", default="download")
NEXUS_PASSWORD = os.getenv("NEXUS_PASSWORD", default="1234")
HASH_REGEX = re.compile(r"-([a-f0-9]*)\.")


def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    file_size = None
    # req = Request(url, headers={"User-Agent": "torch.hub"})
    auth_str = base64.b64encode(f"{NEXUS_USER}:{NEXUS_PASSWORD}".encode()).decode(
        "utf-8"
    )
    req = Request(url, headers={"Authorization": "Basic " + auth_str})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, "getheaders"):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(
            total=file_size,
            disable=not progress,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))
        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[: len(hash_prefix)] != hash_prefix:
                raise RuntimeError(
                    'invalid hash value (expected "{}", got "{}")'.format(
                        hash_prefix, digest
                    )
                )
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


def get_weights_from_nexus(
    task,
    ml_framework,
    model_name,
    backbone,
    ext,
    load=False,
    output_dir=None,
    map_location=None,
    progress=True,
    check_hash=False,
    file_name=None,
):
    hub_dir = get_dir()
    output_dir = os.path.join(hub_dir, "checkpoints")

    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    url = f"{NEXUS_WEIGHTS_URL}/{task}/{ml_framework}_{model_name}/{model_name}_{backbone}.{ext}"
    filename = f"{model_name}_{backbone}.{ext}"
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(output_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    if _is_legacy_zip_format(cached_file):
        return _legacy_zip_load(cached_file, output_dir, map_location)

    if load:
        return torch.load(cached_file, map_location=map_location)
    else:
        return cached_file
