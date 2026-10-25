"""Source and environment fingerprints for checkpoints and evaluation results."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def source_hashes():
    """Hash every Python source under kwm; git metadata is null outside git."""
    files = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((ROOT / "kwm").rglob("*.py"))}
    head = dirty = None
    try:
        opts = dict(cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
                    env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], **opts).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], **opts).strip())
    except (OSError, subprocess.CalledProcessError):
        head = dirty = None
    return dict(git_head=head, git_dirty=dirty, files=files)


def env_fingerprint(cfg):
    """Return the config and SHA-256 of sorted, compact, UTF-8 canonical JSON."""
    cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return dict(cfg=cfg, sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest())
