"""Canonical phase-11 alias for secondary mitigation setup."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

spec = spec_from_file_location("mitigation", Path(__file__).with_name("10_run_mitigation.py"))
module = module_from_spec(spec); spec.loader.exec_module(module)

if __name__ == "__main__": module.main()
