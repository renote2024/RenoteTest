import subprocess
import os
from tqdm import tqdm
import argparse

def create_and_setup_venv(base_path, venv_name):
    venv_path = os.path.join(base_path, venv_name)
    if not os.path.exists(venv_path):
        os.makedirs(venv_path)
    
    # Create the virtual environment
    subprocess.run(['python', '-m', 'venv', venv_path])
    
    # Activate the virtual environment
    activate_script = os.path.join(venv_path, 'bin', 'activate')
    
    # Install requirements
    requirements_file = "requirements.txt" 
    if os.path.isfile(requirements_file):
        subprocess.run(f'source {activate_script} && pip install --upgrade pip && pip install -r {requirements_file} && deactivate', shell=True)
    else:
        print(f"No requirements.txt found at {requirements_file}")

def copy_to_backup(venv_name, source_path, backup_path):
    source_env = os.path.join(source_path, venv_name)
    backup_env = os.path.join(backup_path, venv_name)
    if os.path.exists(source_env):
        subprocess.run(f"cp -r {source_env} {backup_env}", shell=True)
        print(f"Copied {venv_name} to backup at {backup_env}")
    else:
        print(f"Source virtual environment {source_env} does not exist.")

def main(source_path, backup_path, num_envs):
    os.makedirs(source_path, exist_ok=True)
    os.makedirs(backup_path, exist_ok=True)

    for i in tqdm(range(1, num_envs + 1)):
        venv_name = f'nb{i}_venv'
        create_and_setup_venv(source_path, venv_name)
        copy_to_backup(venv_name, source_path, backup_path)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Create and backup multiple virtual environments.")
    arg_parser.add_argument('--source_path', type=str, required=True, help="Path to create virtual environments")
    arg_parser.add_argument('--backup_path', type=str, required=True, help="Path to backup virtual environments")
    arg_parser.add_argument('--num_envs', type=int, default=31, required=True, help="Number of virtual environments to create")
    args = arg_parser.parse_args()

    source_path = args.source_path
    backup_path = args.backup_path
    num_envs = args.num_envs

    main(source_path, backup_path, num_envs)
