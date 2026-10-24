import os
import sys
import types
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# the FloWM reference file imports `lightning` only for the LightningModule base class
if "lightning" not in sys.modules:
    try:
        import lightning  # noqa: F401
    except ImportError:
        import torch.nn as nn
        stub = types.ModuleType("lightning")
        stub.LightningModule = nn.Module
        sys.modules["lightning"] = stub

# load the FloWM reference by file path (a system-wide `tests` package can shadow `tests.reference`)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("flowm_ref", os.path.join(ROOT, "tests", "reference", "flowm_models_2d.py"))
flowm_ref = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(flowm_ref)
sys.modules["flowm_ref"] = flowm_ref


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: dataset regressions; opt in with -m slow or KWM_SLOW=1")


def pytest_collection_modifyitems(config, items):
    # An explicit positive slow selector opts in, including expressions such
    # as 'slow or other'; '-m not slow' still leaves these tests disabled.
    expression = config.getoption("markexpr")
    positive_slow = re.search(r"\bslow\b", re.sub(r"\bnot\s+slow\b", "", expression))
    if os.environ.get("KWM_SLOW") == "1" or positive_slow:
        return
    skip = pytest.mark.skip(reason="dataset regression: use -m slow or KWM_SLOW=1")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
