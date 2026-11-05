import os
import sys
import types
import re

import pytest
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# load the FloWM reference by file path (a system-wide `tests` package can shadow `tests.reference`)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("flowm_ref", os.path.join(ROOT, "tests", "reference", "flowm_models_2d.py"))
flowm_ref = _ilu.module_from_spec(_spec)
# The reference only needs nn.Module, not Lightning's training framework.
# Scope the existing lightweight substitute to this import to avoid startup
# cost and leave other tests' imports unaffected.
_lightning = sys.modules.get("lightning")
_stub = types.ModuleType("lightning")
_stub.LightningModule = nn.Module
sys.modules["lightning"] = _stub
try:
    _spec.loader.exec_module(flowm_ref)
finally:
    if _lightning is None:
        del sys.modules["lightning"]
    else:
        sys.modules["lightning"] = _lightning
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
