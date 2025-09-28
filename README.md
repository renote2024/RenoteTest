# RenoteTest
This is an extension of [our previous work](https://ieeexplore.ieee.org/document/11025746) (accepted to MSR'25). In process of submitting to EMSE journal as a special issue invitee.

Click here for [GitHub repository of previous work](https://github.com/renote2024/ReNote2024)

### Authors

* **Tien Nguyen**: Ph.D. Candidate at Virginia Tech (<tiennguyen@vt.edu>)
* **Waris Gill**: Ph.D. Candidate at Virginia Tech (<waris@vt.edu>)
* **Muhammad Ali Gulzar**: Assistant Professor at Virginia Tech (<gulzar@vt.edu>)

**Description:** A lightweight utility that automatically generates test cases to validate different aspects of Jupyter notebooks. It produces module-availability tests (to check required imports), file-access tests (to ensure referenced input files exist and are readable), and linear-execution tests (to verify whether the notebook runs sequentially without interruption). Together, these tests help identify missing dependencies, broken data paths, and execution failures, providing deeper insights into notebook executability.


## Prerequisites

- Python 3.12+
- Virtual environment management capabilities
- Sufficient disk space for caching, virtual environments, and cloned repositories
- LLM model [Gemma3:12b](https://ollama.com/library/gemma3). You can use other LLM models of preference. Make sure to change it in [localOllama.py](https://github.com/renote2024/RenoteTest/blob/main/main_pipeline/renote_utils/localOllama.py)

## Pre-execution Preparation
To replicate the experiment, you will need to clone a number of repositories onto your local machine, preferably outside the program’s directory.
We provide a folder containing GitHub links to all repositories analyzed in the 2024 and 2025 datasets.

**Note:** Because of the large number of repositories, we are unable to distribute the complete dataset.


## Installation and Preparation

1. Clone the repository:
```bash
git clone https://github.com/renote2024/RenoteTest.git
cd RenoteTest
```

2. Install the required dependencies:
```bash
pip install requirements.txt
```

3. Set up virtual environments for repository analysis in `create_envs.py`:
```bash
python main_pipeline/all_utils/create_venv.py --source_path <path/to/source/envs> --backup_path <path/to/backup/envs> --num_envs <number of venv>
```

If you want to run the program sequentially, 1 venv is enough. If you want to run in parallel, you can decide the number of venvs based on your CPU and GPU capacities.

4. Prepare the dataset
```bash
python -m main_pipeline.all_utils.prepare --root_dir <path/to/all/repos> --output <path/to/repo/nb/csv>
```

 **Note**: <path/to/repo/nb/csv> is the path that contains all repository paths and their corresponding lists of notebook paths within each repository.

## Run the program
1. Execute analysis:
```bash
sudo chattr -R +i ../RenoteTest/ && python -m main_pipeline.main101.main102.main --repo_nb_csv_path <path/to/repo/nb/csv> --json_paths <path/to/a/json/dir> --results_cache_path <path/to/results/cache/dir> --err_cache_path <path/to/error/cache/dir> --resume <0 or 1> --backup_envs_path <path/to/backup/envs/dir> --source_envs_path <path/to/source/envs/dir>
```
- <path/to/repo/nb/csv>: path to a .csv that has information on repositories and notebooks (required fields: repo_path (path to the repository in your working directory) and nb_paths (list of notebook files in that repository)
- <path/to/a/json/dir>: a directory containing temporary JSON files supporting analysis
- <path/to/results/cache/dir>: path to a directory to store results
- <path/to/error/cache/dir>: path to a directory to store error notebooks
- resume: 1 when you want to run all notebooks and check if notebooks have already been evaluated, and 0 otherwise.
- <path/to/backup/envs/dir>: path to your backup venv directory
- <path/to/source/envs/dir>: path to your source venv directory

  **Note:**
    - We recommend running this program inside a protected path by `sudo chattr -R +i ../RenoteTest/`. To edit and unlock permission: `sudo chattr -R -i ../RenoteTest/`
    - We recommend these paths are outside of the program's directory to avoid unwanted behaviors caused by notebooks themselves.
    - The program can be executed in 2 modes (sequential or parallel). Thus, before running the script above, adjust the code to your preferred mode, simply by uncommenting the line you want to execute and commenting out the line you do not want to execute. If run in parallel, please specify the number of workers `num_envs` (should be equal to the number of venvs created).

2. If you want to view the results in CSV:
```bash
# In project main's directory
python main_pipeline/all_utils/convert_cache_to_csv.py --cache_path <path/to/results/cache/dir> --csv <path/to/result/csv/file>
```
