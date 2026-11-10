"""
One-step setup for the Gu Lab Minian pipeline.

Run from the Anaconda Prompt or the Anaconda PowerShell Prompt, in the repo folder:

    python setup_repo.py

It follows the README setup steps:
    1. create (or update / recreate) the minian-gu-lab conda environment from environment.yml
    2. install this repo into it in editable mode (pip install -e .)
    3. verify that minian imports (env_test.py minian)
    4. install the plot-export tools (shot-scraper + Playwright Chromium)
    5. register the "Python (minian-gu-lab)" Jupyter kernel
    6. check GPU / CUDA support (env_test.py cuda) - optional, CPU fallback otherwise
    7. optionally make your personal *_local.ipynb copy of the pipeline template

The environment is activated once through "conda run" and its variables are
reused for every later step, so the script never needs "conda activate" in
the calling shell and behaves the same in both prompts.

Options (all optional):
    --yes              never ask; keep an existing environment and skip the notebook copy
    --update           update an existing environment from environment.yml
    --recreate         delete and recreate an existing environment
    --name NAME        create Minian_pipeline_NAME_local.ipynb from the template
    --skip-plots       skip shot-scraper / Playwright
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = "Minian_pipeline_TEMPLATE.ipynb"
KERNEL_DISPLAY = "Python ({})"

# Enables ANSI colours in the Windows console (cmd and PowerShell).
os.system("")
RED, GREEN, YELLOW, BLUE, BOLD, RESET = (
    "\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[1m", "\033[0m")


def say(msg, colour=""):
    print(f"{colour}{msg}{RESET}", flush=True)


def step(n, total, title):
    say(f"\n[{n}/{total}] {title}", BOLD + BLUE)


def fail(msg):
    say(f"\nSetup stopped: {msg}", RED)
    sys.exit(1)


ACTIVE_ENV = None  # environment variables of the activated conda env, set by activate()


def run(cmd, check=True, capture=False, activated=True):
    """Run a command (list of args) from the repo root, streaming its output.

    Once activate() has run, commands get the activated env's variables,
    unless activated=False (used to test what an unactivated kernel sees).
    """
    say("  $ " + subprocess.list2cmdline(cmd), GREEN)
    result = subprocess.run(cmd, cwd=REPO, text=True,
                            env=ACTIVE_ENV if activated else None,
                            capture_output=capture)
    if check and result.returncode != 0:
        if capture:
            print(result.stdout, result.stderr)
        fail(f"command failed with exit code {result.returncode}: {cmd[0]} {cmd[1] if len(cmd) > 1 else ''}")
    return result


def ask(question, default, assume_default):
    """Ask a question; return the default when non-interactive or --yes."""
    if assume_default:
        return default
    try:
        answer = input(f"{question} ").strip()
    except EOFError:
        return default
    return answer or default


def find_conda():
    """
    Locate the conda executable in either Anaconda prompt.

    Prefers the conda install that owns the python running this script (the
    prompt you opened), because CONDA_EXE can be overwritten by another
    install's cmd AutoRun hook when a machine has more than one Anaconda.
    """
    exe = "Scripts\\conda.exe" if sys.platform == "win32" else "bin/conda"
    roots = [sys.prefix, os.path.dirname(os.path.dirname(sys.prefix))]  # base, or base/envs/<env>
    candidates = [os.path.join(r, exe) for r in roots]
    candidates += [os.environ.get("CONDA_EXE"), shutil.which("conda")]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            other = os.environ.get("CONDA_EXE")
            if other and os.path.normcase(other) != os.path.normcase(candidate):
                say(f"  Note: CONDA_EXE points at another install ({other}); using {candidate}.", YELLOW)
            return candidate
    fail("conda was not found. Open the 'Anaconda Prompt' or 'Anaconda PowerShell "
         "Prompt' from the Start menu, cd to this folder and run 'python setup_repo.py' again.")


def conda_version(conda):
    """
    Assumes format of 'conda <major>.<minor>.<patch>'.
    """
    out = run([conda, "--version"], capture=True).stdout
    try:
        major, minor = out.split()[-1].split(".")[:2]
        return int(major), int(minor)
    except (ValueError, IndexError):
        return (0, 0)



def env_name_from(env_file):
    with open(env_file, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("name:"):
                return line.split(":", 1)[1].strip()
    fail(f"{env_file} has no 'name:' line.")


def env_exists(conda, name):
    envs = json.loads(run([conda, "env", "list", "--json"], capture=True).stdout)["envs"]
    return any(os.path.basename(os.path.normpath(p)) == name for p in envs)


def activate(conda, name):
    """
    Return the env's python and capture the variables 'conda activate <name>'
    sets, so later commands run exactly as in an activated prompt.

    Activation matters: it puts <env>\\Library\\bin on PATH and runs the
    packages' activate.d scripts. Without it, CuPy cannot find the NVRTC
    DLLs from cudatoolkit and every GPU kernel fails to compile.
    """
    global ACTIVE_ENV
    marker = "@@ENV@@"
    probe = ("import json, os, sys; "
             f"print('{marker}' + json.dumps({{'python': sys.executable, 'env': dict(os.environ)}}))")
    result = run([conda, "run", "-n", name, "python", "-c", probe], check=False, capture=True)
    payload = next((l[len(marker):] for l in result.stdout.splitlines() if l.startswith(marker)), None)
    if result.returncode != 0 or payload is None:
        print(result.stdout, result.stderr)
        fail(f"could not activate the '{name}' environment.")
    info = json.loads(payload)
    ACTIVE_ENV = info["env"]
    return info["python"]


# Runs at every interpreter start in the env (via a .pth file), before any user
# import. Kept free of non-stdlib imports and safe to run more than once.
CUDA_HOOK = '''\
"""
Installed by setup_repo.py. Does for CuPy what "conda activate" does.

cupy's conda activation sets CUDA_PATH to the env and puts <env>\\\\Library\\\\bin on
PATH. Jupyter kernels (VS Code, notebook) often start the env's python without
activating it; CuPy then follows a system-wide CUDA_PATH (e.g. a CUDA 12 install),
cannot load the env's NVRTC, and every GPU kernel fails to compile, so the
pipeline silently falls back to the CPU. This must run before anything imports
cupy (xarray and dask.array both do), hence the .pth hook.
"""
import os
import sys


def _apply():
    if sys.platform != "win32":
        return
    if os.path.normcase(os.environ.get("CUDA_PATH", "")) == os.path.normcase(sys.prefix):
        return  # already activated
    lib = os.path.join(sys.prefix, "Library", "bin")
    try:
        names = os.listdir(lib)
    except OSError:
        return
    if not any(n.lower().startswith("nvrtc64_") for n in names):
        return  # no cudatoolkit in this env
    if "CUDA_PATH" in os.environ:
        os.environ.setdefault("CONDA_CUPY_CUDA_PATH", os.environ["CUDA_PATH"])
    os.environ["CUDA_PATH"] = sys.prefix
    paths = os.environ.get("PATH", "").split(os.pathsep)
    if os.path.normcase(lib) not in [os.path.normcase(p) for p in paths]:
        os.environ["PATH"] = os.pathsep.join([lib] + paths)


_apply()
'''


def install_cuda_hook(py):
    """Install CUDA_HOOK into the env's site-packages with a .pth that loads it."""
    site = run([py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
               capture=True).stdout.strip().splitlines()[-1]
    with open(os.path.join(site, "minian_cuda_env.py"), "w", encoding="utf-8") as fh:
        fh.write(CUDA_HOOK)
    with open(os.path.join(site, "minian_cuda_env.pth"), "w", encoding="utf-8") as fh:
        fh.write("import minian_cuda_env\n")
    say(f"  Installed CUDA startup hook in {site}")


def setup_environment(conda, args, env_file, name):
    solver = []
    if conda_version(conda) >= (23, 10):
        solver = ["--solver=libmamba"]  # fast solver, bundled since conda 23.10
    else:
        say("  Your conda is older than 23.10; solving may be slow. "
            "If it fails, run 'conda update -n base conda' and try again.", YELLOW)

    create = [conda, "env", "create", "-n", name, "-f", env_file] + solver
    if not env_exists(conda, name):
        run(create)
        return

    if args.recreate:
        choice = "r"
    elif args.update:
        choice = "u"
    else:
        say(f"  The '{name}' environment already exists.", YELLOW)
        choice = ask("  [k]eep it as is, [u]pdate it from environment.yml, or [r]ecreate it from scratch? [k]",
                     "k", args.yes).lower()[:1]

    if choice == "u":
        run([conda, "env", "update", "-n", name, "-f", env_file, "--prune"] + solver)
    elif choice == "r":
        run([conda, "env", "remove", "-n", name, "-y"])
        run(create)
    else:
        say("  Keeping the existing environment.")


def make_local_notebook(args):
    template = os.path.join(REPO, TEMPLATE)
    existing = [f for f in os.listdir(REPO) if f.endswith("_local.ipynb")]
    if not os.path.exists(template):
        say(f"  {TEMPLATE} not found; skipping.", YELLOW)
        return None
    if existing and not args.name:
        say(f"  You already have: {', '.join(existing)}")
        return existing[0]

    name = args.name or ask(
        "  Your name for a personal notebook copy (e.g. 'jdoe'), or press Enter to skip:", "", args.yes)
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    if not name:
        say("  Skipped.")
        return None
    target = f"Minian_pipeline_{name}_local.ipynb"
    if os.path.exists(os.path.join(REPO, target)):
        say(f"  {target} already exists; leaving it untouched.")
    else:
        shutil.copyfile(template, os.path.join(REPO, target))
        say(f"  Created {target} (ignored by git, so pulls never change it).", GREEN)
    return target


def main():
    parser = argparse.ArgumentParser(description="Set up the Gu Lab Minian pipeline.")
    parser.add_argument("--yes", action="store_true", help="never ask questions")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--update", action="store_true", help="update an existing environment")
    group.add_argument("--recreate", action="store_true", help="delete and recreate an existing environment")
    parser.add_argument("--name", help="create Minian_pipeline_NAME_local.ipynb from the template")
    parser.add_argument("--env-name", help="use this environment name instead of the one in the "
                        "yml (e.g. to test a fresh install without touching minian-gu-lab)")
    parser.add_argument("--skip-plots", action="store_true", help="skip shot-scraper / Playwright")
    args = parser.parse_args()

    total = 7
    say("Gu Lab Minian pipeline setup", BOLD)
    conda = find_conda()
    say(f"  Using conda: {conda}")

    # environment.yml pins Windows builds; other systems use the older, unpinned spec.
    env_file = "environment.yml" if sys.platform == "win32" else "old.environment.yml"
    if not os.path.exists(os.path.join(REPO, env_file)):
        fail(f"{env_file} not found in {REPO}.")
    name = args.env_name or env_name_from(os.path.join(REPO, env_file))

    step(1, total, f"Conda environment '{name}' from {env_file} (can take 5-15 minutes)")
    setup_environment(conda, args, env_file, name)
    py = activate(conda, name)
    say(f"  Using {py}")

    step(2, total, "Installing this repo in editable mode")
    run([py, "-m", "pip", "install", "-e", "."])

    step(3, total, "Checking that minian imports")
    if run([py, "env_test.py", "minian"], check=False).returncode != 0:
        fail("minian is not importable in the environment. Try 'python setup_repo.py --recreate'.")

    step(4, total, "Plot export tools (shot-scraper, Playwright Chromium)")
    if args.skip_plots:
        say("  Skipped (--skip-plots).")
    else:
        ok = run([py, "-m", "pip", "install", "shot-scraper"], check=False).returncode == 0
        ok = ok and run([py, "-m", "playwright", "install", "chromium"], check=False).returncode == 0
        if not ok:
            say("  Plot export tools failed to install; the pipeline still works without them.", YELLOW)

    step(5, total, "Registering the Jupyter kernel")
    run([py, "-m", "ipykernel", "install", "--user", "--name", name,
         "--display-name", KERNEL_DISPLAY.format(name)])

    step(6, total, "Checking GPU / CUDA support")
    install_cuda_hook(py)
    # Run unactivated on purpose: that is how Jupyter kernels start the env.
    gpu = run([py, "env_test.py", "cuda"], check=False, activated=False).returncode == 0
    if not gpu:
        say("  Motion correction will run on the CPU (slower, but works).", YELLOW)

    step(7, total, "Personal notebook copy")
    notebook = make_local_notebook(args)

    say("\nSetup complete.", BOLD + GREEN)
    say(f"  GPU motion correction: {'available' if gpu else 'not available (CPU fallback)'}")
    say("\nNext steps (in VS Code):")
    say("  1. File > Open Folder > select this folder")
    say(f"  2. Ctrl+Shift+P > 'Python: Select Interpreter' > {name}")
    say(f"  3. Open {notebook or 'a copy of ' + TEMPLATE + ' ending in _local.ipynb'} "
        f"and pick the '{KERNEL_DISPLAY.format(name)}' kernel (top right)")


if __name__ == "__main__":
    main()
