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

Every step calls the environment's own python directly, so nothing depends on
"conda activate" and the script behaves the same in both prompts.

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


def run(cmd, check=True, capture=False):
    """Run a command (list of args) from the repo root, streaming its output."""
    say("  $ " + subprocess.list2cmdline(cmd), GREEN)
    result = subprocess.run(cmd, cwd=REPO, text=True,
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
    out = run([conda, "--version"], capture=True).stdout
    match = re.search(r"(\d+)\.(\d+)", out)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def env_name_from(env_file):
    with open(env_file, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("name:"):
                return line.split(":", 1)[1].strip()
    fail(f"{env_file} has no 'name:' line.")


def env_exists(conda, name):
    envs = json.loads(run([conda, "env", "list", "--json"], capture=True).stdout)["envs"]
    return any(os.path.basename(os.path.normpath(p)) == name for p in envs)


def env_python(conda, name):
    """Path of the python that 'conda activate <name>' would give."""
    result = run([conda, "run", "-n", name, "python", "-c",
                  "import sys; print(sys.executable)"], check=False, capture=True)
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    if result.returncode != 0 or not lines or not os.path.exists(lines[-1]):
        print(result.stdout, result.stderr)
        fail(f"could not find python inside the '{name}' environment.")
    return lines[-1]


def setup_environment(conda, args, env_file, name):
    solver = []
    if conda_version(conda) >= (23, 10):
        solver = ["--solver=libmamba"]  # fast solver, bundled since conda 23.10
    else:
        say("  Your conda is older than 23.10; solving may be slow. "
            "If it fails, run 'conda update -n base conda' and try again.", YELLOW)

    create = [conda, "env", "create", "-f", env_file] + solver
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
        run([conda, "env", "update", "-f", env_file, "--prune"] + solver)
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
    name = env_name_from(os.path.join(REPO, env_file))

    step(1, total, f"Conda environment '{name}' from {env_file} (can take 5-15 minutes)")
    setup_environment(conda, args, env_file, name)
    py = env_python(conda, name)
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
    gpu = run([py, "env_test.py", "cuda"], check=False).returncode == 0
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
