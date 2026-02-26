import subprocess
import sys

# Color Codes 
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

def run(cmd, **kwargs):
    """Run a shell command, exit on failure unless check=False."""
    print(f"${GREEN} {cmd} {RESET}")
    return subprocess.run(cmd, shell=True, **kwargs)
def conda_env_exists(name):
    result = run(f"conda env list", capture_output=True, text=True)
    return name in result.stdout
def setup():
    env_name = "minian-gu-lab"
    # Check if env already exists
    if conda_env_exists(env_name):
        choice = input(
            "It seems like the conda environment already exists "
            "(you may want to run startup.sh)! Do you still want to proceed? y/[n] "
        ).strip().lower()
        if choice != "y":
            return
    print("Setting up your minian pipeline for the first time!")
    print("This first part might take ~5 minutes...")
    run("conda init")
    run("conda env create -f winenv.yml --solver=libmamba", check=True)
    run(f"conda run -n {env_name} pip install -e .", check=True)
    # Test minian
    result = run(f"conda run -n {env_name} python st.py")
    if result.returncode == 0:
        print("Minian installation successful")
        run(f"conda run -n {env_name} pip install shot-scraper", check=True)
        run(f"conda run -n {env_name} python -m playwright install chromium", check=True)
        print(f"{BLUE} Setup complete {RESET}")
    else:
        print(f"{RED}Minian installation not detected, exiting...{RESET}")
        sys.exit(1)
if __name__ == "__main__":
    setup()
