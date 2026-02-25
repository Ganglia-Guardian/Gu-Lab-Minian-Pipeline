echo "Setting up your minian pipeline for the first time!"
echo "This first part might take a minute..."
conda env create -f winenv.yml --solver=libmamba
conda activate minian-gu-lab
pip install -e .
#Test minian
if python st.py; then
    echo "Minian installation successful"
    pip install shot-scraper
    python -m playwright install chromium
else
    echo "Minian installation not detected, exiting..."
fi


