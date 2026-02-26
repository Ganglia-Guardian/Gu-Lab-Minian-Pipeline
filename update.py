import subprocess
import sys

# Color Codes 
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

def run(cmd, **kwargs):
    print(f"$ {GREEN} {cmd} {RESET}")
    return subprocess.run(cmd, shell=True, **kwargs)
def conda_env_exists(name):
    result = run("conda env list", capture_output=True, text=True)
    return name in result.stdout
def start():
    env_name = "minian-gu-lab"
    branch = "startup-script"
    # Check if env exists
    if not conda_env_exists(env_name):
        print("It seems like the conda environment doesn't exist. Try running setup.sh")
        choice = input("Continue anyway? y/[n] ").strip().lower()
        if choice != "y":
            return
    else:
        print("Minian environment found!")
    # Ask about branch
    choice = input(
        "Is there a specific branch you're working on? "
        "(If you don't know what this means type n) y/[n] "
    ).strip().lower()
    if choice == "y":
        branch = input("Input branch name: ").strip()
    # Pull and checkout
    print("Checking upstream updates")
    run("git pull")
    run(f"git checkout origin/{branch}")
    print("Finished updates")
    print(f"{BLUE} Run 'jupyter notebook to startup the notebook server {RESET}")
if __name__ == "__main__":
    start()
