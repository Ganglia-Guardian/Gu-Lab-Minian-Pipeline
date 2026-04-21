#RED = '\e[31m'
#GREEN = '\e[32m'
#BLUE = '\e[34m'
#NC = '\e[0m'

# Eventually add some 'git checkout origin/whateverbranch' to make sure we're on the right branch
#

#Check Status
function checkenv {
  isEnv=$(conda env list | grep "minian-gu-lab")
  if [[ -z "$isEnv" ]]; then
    echo "It seems like the conda environment already exists (you may want to run startup.sh)! Do you still want to proceed? y/[n]"
    read -r -n 1 choice1
    if [[ "$choice1" != 'y' ]]; then
      return 0
    fi
  fi
}

checkenv
echo "Setting up your minian pipeline for the first time!"
echo "This first part might take ~5 minutes..."
conda init
conda env create -f winenv.yml --solver=libmamba
conda activate minian-gu-lab
#DEBUG
# echo "${BLUE}Minian setup done, moving to other stuff. Pres any key to continue...${NC}"
# read -n 1 -s -r -p ""
#ENDDEBUG
pip install -e .
pip install shot-scraper
#Test minian
if python st.py; then
  echo "Minian installation successful"
  pip install shot-scraper
  python -m playwright install chromium
  return 0
else
  echo "Minian installation not detected, exiting..."
  return 1
fi
