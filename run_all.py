import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
# The estimation (12 models, saved in data/dict/) only needs to run once.
#   "auto" -> run only the models whose saved results are missing or unreadable
#             (so a stopped run picks up where it left off)
#   True   -> re-run all 12 models
#   False  -> never run it
RUN_ESTIMATION = "auto"

# Threads for numpy's math library, per process. Leave at 1: pyblp's calculations are
# small, and on Apple chips extra threads mostly wait on each other (seen as ~1,700% CPU,
# mostly "system" time in Activity Monitor) and make the run many times slower.
MATH_THREADS = 1

# ---------------------------------------------------------------------------
# Required packages (Python 3.8-3.10; e.g. the Python 3.9 built into macOS)
# ---------------------------------------------------------------------------
# If any are missing or too new, the script stops and prints the exact
# install command for the Python VS Code is using.
PACKAGES = ("numpy==1.23.5 scipy==1.9.3 pandas==1.5.3 matplotlib==3.5.3 tabulate==0.9.0 "
            "pyarrow==11.0.0 brotli statsmodels==0.13.5 pyblp==0.13.0")

# Comment out any line to skip that step.
SCRIPTS = [
    "tab34_params.py",      # Tables 3, 4
    "tab56_diversion.py",   # Tables 5, 6, B.2
    "tab7_wtp.py",          # Table 7, Figure B.1
    "fig12_decomp.py",      # Figures 1, 2
    "fig34_late.py",        # Figures 3, 4
]

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
DICT = ROOT / "data" / "dict"
RAW = ROOT / "data" / "raw"

# The 12 models, in the same order and with the same functions as code/run_all_cases.py
MODELS = [  # (saved file, module, function)
    ("nevo_results_nl.npy",      "aux_nevo_cases", "get_nevo_nested"),
    ("nevo_results_logit.npy",   "aux_nevo_cases", "get_nevo_logit"),
    ("nevo_results_triple.npy",  "aux_nevo_cases", "get_nevo_triple"),
    ("nevo_results_noconst.npy", "aux_nevo_cases", "get_nevo_nocons"),
    ("nevo_results_noalpha.npy", "aux_nevo_cases", "get_nevo_noalpha"),
    ("nevo_results_base.npy",    "aux_nevo_cases", "get_nevo_base"),
    ("blp_results_nl.npy",       "aux_blp_cases",  "get_blp_nested"),
    ("blp_results_logit.npy",    "aux_blp_cases",  "get_blp_logit"),
    ("blp_results_triple.npy",   "aux_blp_cases",  "get_blp_triple"),
    ("blp_results_noconst.npy",  "aux_blp_cases",  "get_blp_nocons"),
    ("blp_results_noalpha.npy",  "aux_blp_cases",  "get_blp_noalpha"),
    ("blp_results_base.npy",     "aux_blp_cases",  "get_blp_base"),
]
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def fail(msg):
    print(f"\nSTOPPED: {msg}")
    sys.exit(1)


def install_hint():
    py = f'"{sys.executable}"'
    return ("\n\nInstall the right versions into this Python by pasting these two lines "
            "into the VS Code terminal, then click Run again:\n\n"
            f"    {py} -m pip install --user --upgrade pip\n"
            f"    {py} -m pip install --user {PACKAGES}\n")


def run(label, command):
    print(f"\n=== {label} " + "=" * max(0, 60 - len(label)), flush=True)
    start = time.time()
    env = dict(os.environ, MPLBACKEND="Agg")   # save figures without opening windows
    for var in THREAD_VARS:
        env[var] = str(MATH_THREADS)
    result = subprocess.run([sys.executable] + command, cwd=CODE, env=env)
    minutes = (time.time() - start) / 60
    if result.returncode != 0:
        fail(f"{label} failed after {minutes:.1f} min (see the error above).")
    print(f"=== {label} finished in {minutes:.1f} min", flush=True)


def usable(path):
    """A saved result counts only if it exists AND loads with this Python's numpy
    (files saved by a different numpy version, e.g. numpy 2, cannot be read)."""
    if not path.exists():
        return False
    try:
        import numpy as np
        np.load(path, allow_pickle=True)
        return True
    except Exception:
        return False


def main():
    print(f"Python: {sys.executable}")

    # Basic checks before anything slow starts
    if not CODE.is_dir():
        fail(f"no code/ folder next to this file ({ROOT}). Move run_all.py to the repo's top folder.")
    for name in ("blp_product_data_opt.parquet", "nevo_product_data_opt.parquet"):
        f = RAW / name
        if not f.exists():
            fail(f"missing data file {f}")
        if f.read_bytes()[:7] == b"version":
            fail(f"{name} is a Git LFS placeholder, not the data. Run 'git lfs pull' in the repo folder.")
    if not (3, 8) <= sys.version_info[:2] <= (3, 10):
        fail(f"Python {sys.version.split()[0]} is too new for the 2021 package versions. "
             "Click the Python version in VS Code's bottom-right corner and pick a Python 3.9.")
    try:
        import pyblp, statsmodels, pandas, matplotlib  # noqa: F401
    except ImportError as e:
        fail(f"{e.name} is not installed in this Python." + install_hint())

    # The 2021 code crashes on newer packages (pandas 2 removed DataFrame.append,
    # matplotlib 3.8 removed the 'seaborn-whitegrid' style). Catch that now,
    # not after hours of estimation.
    def version(v):
        return tuple(int(x) for x in re.findall(r"\d+", v)[:2])
    too_new = []
    if version(pandas.__version__) >= (2, 0):
        too_new.append(f"pandas {pandas.__version__} (needs < 2.0)")
    if version(matplotlib.__version__) >= (3, 8):
        too_new.append(f"matplotlib {matplotlib.__version__} (needs < 3.8)")
    if too_new:
        fail("this Python has packages that are too new for the code: " + ", ".join(too_new)
             + "." + install_hint())

    DICT.mkdir(parents=True, exist_ok=True)
    (ROOT / "tables").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)

    if RUN_ESTIMATION is True:
        todo = MODELS
    elif RUN_ESTIMATION == "auto":
        print("\nChecking saved results in data/dict/ ...", flush=True)
        todo = [m for m in MODELS if not usable(DICT / m[0])]
    else:
        todo = []

    if not todo:
        print("Skipping estimation: all 12 saved results found in data/dict/.")
    else:
        if len(todo) < len(MODELS):
            print(f"Resuming the estimation: {len(MODELS) - len(todo)} of {len(MODELS)} models "
                  f"already done; running the other {len(todo)}.")
        else:
            print("Running the estimation (all 12 models). Results are saved in data/dict/.")
        for n, (filename, module, func) in enumerate(todo, 1):
            run(f"model {n}/{len(todo)}: {filename}", ["-c", f"import {module}; {module}.{func}()"])

    total = time.time()
    for script in SCRIPTS:
        run(script, [script])

    print(f"\nAll done in {(time.time() - total) / 60:.1f} min.")
    print(f"Tables:  {ROOT / 'tables'}")
    print(f"Figures: {ROOT / 'figures'}")


if __name__ == "__main__":
    main()