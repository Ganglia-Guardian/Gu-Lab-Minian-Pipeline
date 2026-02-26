function checkenv {
  isEnv=$(conda env list | grep "minian-gu-lab")
  if [[ -z "$isEnv" ]]; then
    echo "It seems like the conda environment doesn't exist. Try running setup.sh"
    read -r -n 1 choice1
    if [[ "$choice1" = 'y' ]]; then
      return 0
    fi
  else;
    echo "Minian environment found!"
  fi
}

branch="startup-script"

checkenv
echo "Is there a specific branch you're working on? (If you don't know what this means type n) y/[n]"
read -r -n 1 choice2
if [[ "$choice2" = 'y' ]]; then
  echo "Input branch name"
  read -r branch
fi

echo "Checking upstream updates"
git pull
git checkout "origin/$branch"

echo "Finished updates"
conda activate minian-gu-lab
jupyter notebook
echo "Enjoy :)"
