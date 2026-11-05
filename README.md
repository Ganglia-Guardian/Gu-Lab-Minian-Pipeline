# Gu Lab Minian Pipeline

A customized analysis pipeline and visualization tool for Miniscope data, based on the MiniAn framework. This version includes optimized notebooks and modified source code tailored for our lab's workflow.

## Prerequisites

- [Visual Studio Code](https://code.visualstudio.com/)
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git (usually comes with Anaconda/VS Code)

## Quick Setup (Windows)

After cloning (step 1 below), open the **Anaconda Prompt** or **Anaconda PowerShell Prompt**, go to the repo folder and run:

```bash
python setup_repo.py
```

This performs steps 3–8 below: it creates the `minian-gu-lab` environment, installs the repo, verifies minian, installs the plot-export tools, registers the Jupyter kernel, checks GPU/CUDA support and can create your personal `_local.ipynb` notebook copy. Run `python setup_repo.py --help` for options (for example `--update` after `environment.yml` changes).

To re-check an existing environment at any time:

```bash
conda activate minian-gu-lab
python env_test.py          # exit code 0 = minian and CUDA OK, 2 = no CUDA (CPU fallback), 1/3 = minian broken
```

The manual steps are below if you prefer them or the script fails.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Sarvesh30/Gu-Lab-Minian-Pipeline.git
cd Gu-Lab-Minian-Pipeline
```

### 2. Open in VS Code

Open the cloned folder in Visual Studio Code:
- Launch VS Code
- File → Open Folder → Select `Gu-Lab-Minian-Pipeline`

### 3. Create the Conda Environment

The repository includes an `environment.yml` file with all required dependencies. Create the environment using one of these methods:

#### Option A: Using VS Code Terminal (Go to Option B if conda commands do not work)
1. Open the integrated terminal in VS Code (`Ctrl+` ` or `View → Terminal`)
2. Run:
```bash
conda env create -f environment.yml
```

#### Option B: Using Anaconda Prompt (Windows) or Terminal (Mac/Linux)
```bash
cd path/to/Gu-Lab-Minian-Pipeline
conda env create -f environment.yml --solver=libmamba
```

This will create a new conda environment named `minian-gu-lab` with all necessary packages.

### 4. Activate the Environment

```bash
conda activate minian-gu-lab
```

### 5. Install the Package in Development Mode

This allows you to make changes to the source code and have them immediately available:

```bash
pip install -e .
```

The `-e` flag installs the package in "editable" mode, so any changes you make to the code will be reflected without reinstalling.

### 6. Select Python Interpreter in VS Code

1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
2. Type "Python: Select Interpreter"
3. Choose the `minian-gu-lab` conda environment from the list

### 7. Verify Installation

Open a Python terminal or Jupyter notebook and verify the installation:

```python
import minian
print(minian.__version__)
```

### 8. Exporting Plots
```bash
pip install shot-scraper
python -m playwright install chromium
```

## Working with Notebooks

`Minian_pipeline_TEMPLATE.ipynb` is the shared pipeline notebook. **Never edit or run the template itself.** It is updated by `git pull`, and any local edits to it will block the next pull.

1. Make a copy of the template **in the project root** (next to `minian/` and `manual_curation_update.py`, so the imports work) with a name ending in `_local.ipynb`, e.g. `Minian_pipeline_YOURNAME_local.ipynb`.
   Files ending in `_local.ipynb` are ignored by git, so pulls never change them and they never show up as changes.
2. Open your copy in VS Code and select the **Python (minian-gu-lab)** kernel (top right).
3. Set your paths, video pattern and parameters in your copy.

When a pull updates the template, either make a fresh copy and re-enter your settings, or move the changed cells into your existing copy.

## Project Structure

```
Gu-Lab-Minian-Pipeline/
├── .github/workflows/    # CI/CD workflows
├── demo_data/           # Example datasets for testing
├── docs/                # Documentation
├── img/                 # Images and figures
├── minian/              # Source code (modified for our lab)
├── notebooks/           # Optimized analysis notebooks
├── environment.yml      # Conda environment (exact Windows builds, incl. CUDA/cupy)
├── old.environment.yml  # Previous unpinned environment (non-Windows, CI)
├── setup_repo.py        # One-step setup script
├── env_test.py          # Checks minian install and CUDA support
├── setup.py            # Package installation configuration
├── pyproject.toml      # Modern Python package metadata
└── README.md           # This file
```

## Contributing & Making Changes

### Workflow for Lab Members

1. **Create a new branch** for your work (never work directly on `main`):
   ```bash
   git checkout -b your-name/feature-description
   ```

2. **Make your changes** to notebooks or source code

3. **Test your changes** thoroughly with your own data

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Brief description of your changes"
   ```

5. **Push your branch** to GitHub:
   ```bash
   git push origin your-name/feature-description
   ```

6. **Create a Pull Request** on GitHub:
   - Go to the repository page
   - Click "Pull Requests" → "New Pull Request"
   - Select your branch
   - Add a description of your changes
   - Request review from team members

### Branch Naming Convention

Use descriptive branch names:
- `yourname/add-new-preprocessing-step`
- `yourname/fix-motion-correction`
- `yourname/update-visualization`

## Troubleshooting

### Common Issues

**Issue: Conda environment creation fails**
- Solution: Update conda first: `conda update -n base conda`
- Try creating environment with: `conda env create -f environment.yml --solver=libmamba` (faster solver)

**Issue: Jupyter kernel not found**
- Solution: Install ipykernel in the environment:
  ```bash
  conda activate minian-gu-lab
  conda install ipykernel
  python -m ipykernel install --user --name minian-gu-lab --display-name "Python (minian-gu-lab)"
  ```

**Issue: Import errors after modifying source code**
- Solution: Restart your Jupyter kernel or Python session

**Issue: Package conflicts**
- Solution: Delete and recreate the environment:
  ```bash
  conda deactivate
  conda env remove -n minian-gu-lab
  conda env create -f environment.yml
  ```

**Issue: VS Code doesn't recognize the conda environment**
- Solution: Reload VS Code window (`Ctrl+Shift+P` → "Reload Window")

### Getting Help

If you encounter issues not listed here:
1. Check the [original MiniAn documentation](https://minian.readthedocs.io/)
2. Ask in our lab Slack/Teams channel
3. Create an issue on this repository with:
   - Your operating system
   - Error message (full traceback)
   - Steps to reproduce the problem

## Updating Your Local Repository

To get the latest changes from the main repository:

1. Check that nothing tracked has been edited locally:
   ```bash
   git status
   ```
   If git lists modified tracked files (for example the template notebook), save a copy of anything you need and discard or stash them (`git stash`), otherwise the pull will stop with "Your local changes would be overwritten". Your `_local.ipynb` copies and pipeline outputs are not affected.

2. Pull:
   ```bash
   git checkout main
   git pull origin main
   ```

3. Restart the notebook kernel so the updated source code is loaded.

Then update your conda environment if `environment.yml` has changed:

```bash
conda env update -f environment.yml --prune
```

## Additional Resources

- [Original MiniAn Documentation](https://minian.readthedocs.io/)
- [CaImAn GitHub](https://github.com/flatironinstitute/CaImAn)
- [MIN1PIPE GitHub](https://github.com/JinghaoLu/MIN1PIPE)

## License

This project is licensed under GNU GPLv3 - see the LICENSE file for details.

## Acknowledgments

This pipeline is based on the MiniAn framework and adapted for the Gu Lab's specific needs. We thank the original MiniAn developers for their excellent work.
