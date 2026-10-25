"""Source and environment fingerprints for checkpoints and evaluation results."""
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def source_hashes():
    """Hash kwm sources and record software; git metadata is null outside git."""
    files = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((Path(ROOT) / "kwm").rglob("*.py"))}
    head = dirty = None
    try:
        opts = dict(cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
                    env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], **opts).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", "kwm", "scripts"], **opts).strip())
    except (OSError, subprocess.CalledProcessError):
        head = dirty = None
    return dict(git_head=head, git_dirty=dirty, files=files,
                software=dict(python=platform.python_version(), torch=str(torch.__version__),
                              numpy=np.__version__, cuda_available=torch.cuda.is_available()))


def env_fingerprint(cfg):
    """Return the config and SHA-256 of sorted, compact, UTF-8 canonical JSON."""
    cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return dict(cfg=cfg, sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest())
