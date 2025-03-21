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

load_dotenv()
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"
NEXUS_WEIGHTS_URL = os.getenv(
    "NEXUS_WEIGHTS_URL", default="https://aivdl.nexus.aiv.ai/repository/aiv-weights"
)
NEXUS_USER = os.getenv("NEXUS_USER", default="download")
NEXUS_PASSWORD = os.getenv("NEXUS_PASSWORD", default="1234")
HASH_REGEX = re.compile(r"-([a-f0-9]*)\.")

def _is_legacy_zip_format(filename):
    import zipfile

    if zipfile.is_zipfile(filename):
        infolist = zipfile.ZipFile(filename).infolist()
        return len(infolist) == 1 and not infolist[0].is_dir()
    return False


def _legacy_zip_load(filename, model_dir, map_location):
    warnings.warn(
        "Falling back to the old format < 1.6. This support will be "
        "deprecated in favor of default zipfile format introduced in 1.6. "
        "Please redo torch.save() to save it in the new zipfile format."
    )
    # Note: extractall() defaults to overwrite file if exists. No need to clean up beforehand.
    #       We deliberately don't handle tarfile here since our legacy serialization format was in tar.
    #       E.g. resnet18-5c106cde.pth which is widely used.
    with zipfile.ZipFile(filename) as f:
        members = f.infolist()
        if len(members) != 1:
            raise RuntimeError("Only one file(not dir) is allowed in the zipfile")
        f.extractall(model_dir)
        extraced_name = members[0].filename
        extracted_file = os.path.join(model_dir, extraced_name)
    return torch.load(extracted_file, map_location=map_location)


def get_dir():
    torch_home = os.path.expanduser(
        os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), "aiv")
    )
    return torch_home

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
    weights_name=None,
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

    if weights_name:
        url = f"{NEXUS_WEIGHTS_URL}/{task}/{ml_framework}_{model_name}/{weights_name}.{ext}"
    else:
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
