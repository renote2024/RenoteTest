import subprocess
import json
import argparse
import os
import shutil
import pandas as pd
from diskcache import Index
from joblib import Parallel, delayed

from main_pipeline.all_utils.print_format import print_msg
from main_pipeline.all_utils.requirement_file_process import convertAllRequirementsFiles

def split_list_into_n_parts(lst, num_parts):
    """
    Split a list into n parts with approximately equal sizes.
    The first few parts may be one item longer if the list can't be evenly divided.
    """
    k, m = divmod(len(lst), num_parts)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_parts)]


def split_dict_by_chunk_size(input_dict, chunk_size):
    # Convert dictionary items to a list of tuples
    items = list(input_dict.items())
    
    # Create a list of smaller dictionaries with specified chunk size
    return [dict(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]
    

def read_valid_notebooks_csv(csv_path, resume, results_cache_path, err_cache_path):
    df = pd.read_csv(csv_path)
    repo_dict = {}
    total_nbs = 0

    results_cache = err_cache = None
    if resume > 0:
        results_cache = Index(results_cache_path)
        err_cache = Index(err_cache_path)

    for _, row in df.iterrows():
        repo = row['repo_path']
        nb_paths_str = row['nb_paths']
        nb_list = [nb.strip() for nb in nb_paths_str.split(';')] if pd.notna(nb_paths_str) else []
        if resume > 0:
            # Filter out notebooks that are already evaluated
            nb_list = [nb for nb in nb_list if not (nb in results_cache or nb in err_cache)]
            if not nb_list:
                continue
        repo_dict[repo] = nb_list
        total_nbs += len(nb_list)

    total_repos = len(repo_dict)
    return repo_dict, total_repos, total_nbs


def reset_virtual_env(backup_venv_path, source_venv_path):
    """Reset the source virtual environment by copying from the backup virtual environment.
    If the source virtual environment already exists, it will be deleted first.
    Args:
        backup_venv_path (str): Path to the backup virtual environment.
        source_venv_path (str): Path to the source virtual environment to be reset.
    Raises:
        FileNotFoundError: If the backup virtual environment does not exist.
    """
    if not os.path.exists(backup_venv_path):
        raise FileNotFoundError(f"Backup venv not found: {backup_venv_path}")

    if os.path.exists(source_venv_path):
        print_msg(f"🗑️ Removing existing venv at: {source_venv_path}", 1)
        shutil.rmtree(source_venv_path)

    print_msg(f"♻️ Copying backup venv from {backup_venv_path} to {source_venv_path}", 1)
    shutil.copytree(backup_venv_path, source_venv_path, symlinks=True)  # symlinks=True => Preserve symlinks if any


def process_notebooks_in_shell_env(local_env, config):
    repo_path = config['repo_path']
    nb_paths = config['nb_paths']       # list
    json_paths = config['json_paths']
    results_cache_path = config['results_cache_path']
    err_cache_path = config['err_cache_path']
    resume = config['resume']           # 1 or 0
    backup_venv_path = os.path.join(config['backup_envs_path'], local_env)
    source_venv_path = os.path.join(config['source_envs_path'], local_env)
    total_repos = config['total_repos']
    repo_index = config['index']
    repo_name = '/'.join(repo_path.split('/')[-2:])

    print(f"\n{local_env} | \x1b[1m\x1b[38;5;201m{repo_index}/{total_repos}. {repo_name}\x1b[0m")

    reset_virtual_env(backup_venv_path, source_venv_path)

    # Activate the virtual environment
    activate_script = os.path.join(source_venv_path, 'bin', 'activate')
    command_activate = f'source {activate_script} &&'

    # Install requirements, if any
    tmp_env_dir = f'/home/chat/tmp/{local_env}'
    os.makedirs(tmp_env_dir, exist_ok=True)
    
    # Set up environment variables with security restrictions
    env = os.environ.copy()
    env['TMPDIR'] = tmp_env_dir
    env['PIP_TMPDIR'] = tmp_env_dir
    env['PYTHON_EGG_CACHE'] = tmp_env_dir
    env['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
    
    # Security: Restrict PATH to essentials and remove dangerous paths
    safe_paths = [
        "/usr/local/bin",
        "/usr/bin", 
        "/bin",
        os.path.join(source_venv_path, 'bin')  # Only allow venv binaries
    ]
    env['PATH'] = ":".join(safe_paths)
    
    # Remove potentially dangerous environment variables
    dangerous_vars = ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH']
    for var in dangerous_vars:
        env.pop(var, None)

    command_install_requirements = ''
    out_req_file = None
    try:
        out_req_file = convertAllRequirementsFiles(repo_path)
    except Exception as e:
        print(f"    === Error in converting requirement files for repo {repo_name}, skipping installation of requirements. Error: {e} ===")
    if out_req_file:
        # Add --ignore-installed to avoid conflicts and --no-deps to skip dependency resolution for problematic packages
        command_install_requirements = (
            f'pip install -r {out_req_file} --no-cache-dir --quiet '
            f'--upgrade --upgrade-strategy eager --ignore-installed --disable-pip-version-check || true && '
            'pip check || true &&'
        )
    
    has_req_file = True if out_req_file else False
    data = {
        'repo_path': repo_path,
        'nb_paths': nb_paths,
        'results_cache_path': results_cache_path,
        'err_cache_path': err_cache_path,
        'resume': resume,
        'tmp_env_dir': tmp_env_dir,
        'has_req_file': has_req_file
    }

    # Save the data to a json file
    json_path = os.path.join(json_paths, f'{os.path.basename(source_venv_path)}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    # Run the process_repo.py script
    command_run_process_repo = f'python -m main_pipeline.main101.main102.process_repo --json_path {json_path} &&'

    # Deactivate the virtual environment
    command_deactivate = 'deactivate'

    # Execute the full command
    try:    
        command = f'export TMPDIR={tmp_env_dir} PIP_TMPDIR={tmp_env_dir} PYTHON_EGG_CACHE={tmp_env_dir} ; {command_activate} {command_install_requirements} {command_run_process_repo} {command_deactivate}'
        subprocess.run(command, shell=True, env=env)
    except Exception as e:
        print(f"    === Error occurred while processing the repo {repo_name} === \n{e}")
    finally:
        subprocess.run(f'rm -rf {tmp_env_dir}/*', shell=True)
        if out_req_file and os.path.exists(out_req_file):
            os.remove(out_req_file)

    print(f"{local_env} | \x1b[1m\x1b[38;5;201m{repo_index}/{total_repos}. {repo_name}\x1b[0m \x1b[32mDONE\x1b[0m")


def create_config(index, total_repos, repo_path, nb_paths, results_cache_path, err_cache_path, resume, backup_envs_path, source_envs_path, json_paths):
    '''Create a configuration dictionary for processing a repository.'''
    return {
        'index': index,
        'total_repos': total_repos,
        'repo_path': repo_path,
        'nb_paths': nb_paths,
        'results_cache_path': results_cache_path,
        'err_cache_path': err_cache_path,
        'resume': resume,
        'backup_envs_path': backup_envs_path,
        'source_envs_path': source_envs_path,
        'json_paths': json_paths
    }


def process_notebooks_sequential(repo_nb_csv_path, 
                              json_paths, 
                              results_cache_path, 
                              err_cache_path, 
                              resume, 
                              backup_envs_path, 
                              source_envs_path, 
                              env='nb1_venv'):

    repo_dict, total_repos, total_nbs = read_valid_notebooks_csv(repo_nb_csv_path, resume, results_cache_path, err_cache_path)

    print(f"TOTAL {total_repos} REPOSITORIES & {total_nbs} NOTEBOOKS TO PROCESS")

    for repo_index, (repo_path, nb_paths) in enumerate(list(repo_dict.items())[:1], start=1):
        config = create_config(repo_index, len(repo_dict), repo_path, nb_paths, results_cache_path, err_cache_path, resume, backup_envs_path, source_envs_path, json_paths)
        try:
            process_notebooks_in_shell_env(env, config)
        except Exception as e:
            print(f"    Error in processing the repository {repo_path}, Error: {e}")
            print(f'    >>> EXITING THE PROCESSING OF THE REPOSITORY DUE TO ERROR <<<')
            continue

def execute_task(env, task_list):
    '''Execute a list of tasks (config dictionaries) in a specified virtual environment.

    Parameters:
        env: str, name of the virtual environment to use (e.g., 'nb1_venv')
        task_list: list of configuration dictionaries for processing repositories
    '''
    for config in task_list:
        try:
            process_notebooks_in_shell_env(env, config)
        except Exception as e:
            print(f"Error in processing the repository {config['repo_path']}, Error: {e}")


def process_notebooks_parallel(repo_nb_csv_path, 
                            json_paths, 
                            results_cache_path, 
                            err_cache_path, 
                            resume, 
                            backup_envs_path, 
                            source_envs_path,
                            num_envs):

    repo_dict, total_repos, total_nbs = read_valid_notebooks_csv(repo_nb_csv_path, resume, results_cache_path, err_cache_path)
    print(f"TOTAL {total_repos} REPOSITORIES, {total_nbs} NOTEBOOKS TO PROCESS")
    envs = [f'nb{i}_venv' for i in range(1, num_envs + 1)]

    list_of_all_repos = split_dict_by_chunk_size(repo_dict, chunk_size=len(envs) * 3)
    for repo_list in list_of_all_repos:
        all_repos_with_assign_ids = [
            create_config(i, len(repo_list), repo_path, nb_paths, results_cache_path, err_cache_path, resume, backup_envs_path, source_envs_path, json_paths)
            for i, (repo_path, nb_paths) in enumerate(repo_list.items(), start=1)
        ]
        li_of_li_tasks = split_list_into_n_parts(all_repos_with_assign_ids, len(envs))
        assert len(li_of_li_tasks) == len(envs)
        Parallel(backend='loky', n_jobs=len(envs), verbose=10)(
            delayed(execute_task)(env, task_l) for env, task_l in zip(envs, li_of_li_tasks)
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read all .ipynb files in a directory.')
    parser.add_argument('--repo_nb_csv_path', type=str, required=True, help='Path to the csv file containing repo and notebook paths')
    parser.add_argument('--json_paths', type=str, required=True, help='Path to the json files storing repo information for process_repo.py')
    parser.add_argument('--results_cache_path', type=str, required=True, help='Path to the results cache [DiskCache]')
    parser.add_argument('--resume', type=int,  help='Check the cache before processing the notebook')
    parser.add_argument('--err_cache_path', type=str, required=True, help='Path to the error cache [DiskCache]')
    parser.add_argument('--backup_envs_path', type=str, required=True, help='Path to the backup virtual environments')
    parser.add_argument('--source_envs_path', type=str, required=True, help='Path to the source virtual environments')

    args = parser.parse_args()

    os.makedirs(args.results_cache_path, exist_ok=True)
    os.makedirs(args.err_cache_path, exist_ok=True)
    os.makedirs(args.json_paths, exist_ok=True)

    env = 'nb1_venv'
    num_envs = 16

    # process_notebooks_sequential(repo_nb_csv_path=args.repo_nb_csv_path,
    #                        json_paths=args.json_paths,
    #                        results_cache_path=args.results_cache_path,
    #                        err_cache_path=args.err_cache_path,
    #                        resume=args.resume,
    #                        backup_envs_path=args.backup_envs_path,
    #                        source_envs_path=args.source_envs_path,
    #                        env=env)

    process_notebooks_parallel(repo_nb_csv_path=args.repo_nb_csv_path,
                             json_paths=args.json_paths,
                             results_cache_path=args.results_cache_path,
                             err_cache_path=args.err_cache_path,
                             resume=args.resume,
                             backup_envs_path=args.backup_envs_path,
                             source_envs_path=args.source_envs_path,
                             num_envs=num_envs)
