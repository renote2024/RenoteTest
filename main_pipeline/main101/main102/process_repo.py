import argparse
import json
import os
from diskcache import Index
from pathlib import Path
import shutil

from main_pipeline.all_utils.print_format import print_msg
from main_pipeline.main101.main102.process_nb import process_nb


def clean_up_temp_files(temp_files, path_name):
    """
    Clean up temporary files created during the notebook processing
    """
    if isinstance(temp_files, str):
        temp_files = [temp_files]

    if not temp_files:
        return

    print_msg(f"🧨 Cleaning up {len(temp_files)} path(s) from {path_name}", 1)
    for i, temp_file in enumerate(temp_files, start=1):
        temp_file = Path(temp_file).resolve()
        if temp_file.exists():
            print_msg(f"{i}. 🧹 Attempt to delete {temp_file}", 2)
            try:
                if temp_file.is_dir():
                    shutil.rmtree(temp_file)
                    print_msg("✅ Deleted a folder", 3)
                else:
                    temp_file.unlink()
                    print_msg("✅ Deleted a file", 3)
            except Exception as e:
                print_msg(f"❌ Failed to delete: {e}", 3)

def main(json_path):
    # Read the json file
    with open(json_path, "r", encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Get the data
    repo_path = data["repo_path"]
    nb_paths = data["nb_paths"]
    results_cache_path = data["results_cache_path"]
    err_cache_path = data["err_cache_path"]
    resume = data["resume"]
    tmp_env_dir = data['tmp_env_dir']
    has_req_file = data['has_req_file']

    # Initialize caches ahead of time
    results_cache = Index(results_cache_path)
    err_cache = Index(err_cache_path)

    missing_files_paths_to_remove = []
    name_fixed_paths = []
    
    # Process the notebooks
    try:
        
        for i, nb_path in enumerate(nb_paths, start=1):
            nb_name = os.path.basename(nb_path)
            if resume > 0:
                res = results_cache.get(nb_path, None)
                res_err = err_cache.get(nb_path, None)
                if res is not None or res_err is not None:
                    print(f"{nb_path} is already evaluated.")
                    continue

            print("\n    " + f"\x1b[1m\x1b[38;5;215m[{i}/{len(nb_paths)}] {nb_name}\x1b[0m")

            missing_files_paths_to_remove = []
            name_fixed_paths = []

            try:
                result = process_nb(nb_path, repo_path, tmp_env_dir)
                pre_exec_result = result['pre_test_result']
                post_exec_result = result['post_test_result']
                renote_result = result['renote_result']
                if result['missing_files_paths_to_remove']:
                    missing_files_paths_to_remove.extend(result['missing_files_paths_to_remove'])
                if result['name_fixed_paths']: 
                    name_fixed_paths.extend(result['name_fixed_paths'])
                
                paper_results = renote_result
                paper_results['repo_path'] = repo_path
                paper_results['has_req_file'] = has_req_file

                if "error" in pre_exec_result or "error" in post_exec_result:
                    status = pre_exec_result.get("error", post_exec_result.get("error", "Unknown error"))
                    err_cache[nb_path] = {"repo_path": repo_path, "status": status}
                    pass

                paper_results = {**paper_results, **pre_exec_result}
                paper_results = {**paper_results, **post_exec_result}

                results_cache[nb_path] = paper_results
            except Exception as e:
                err_cache[nb_path] = {"repo_path": repo_path, "status": str(e)}
                print(f"        {e}")
    finally:
        print()
        clean_up_temp_files(json_path, "JSON Configs")
        clean_up_temp_files(missing_files_paths_to_remove, "Missing Files")
        clean_up_temp_files(name_fixed_paths, "Name Fixed Paths")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysize a Single .ipynb file.')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the json file')
    args = parser.parse_args()

    main(json_path=args.json_path)
