import copy
from collections import defaultdict

from main_pipeline.test_utils.testFiles import TestFiles
from main_pipeline.test_utils.testExecution import TestExecution
from main_pipeline.test_utils.testModule import TestModule
from main_pipeline.all_utils.print_format import print_test_result, print_renote_results, print_msg

from main_pipeline.renote_utils.ExecuteNotebook import ExecuteNotebook
from main_pipeline.renote_utils.FixFileNotFound import FixFileNotFound
from main_pipeline.renote_utils.FixModuleNotFound import FixModuleNotFound
from main_pipeline.renote_utils.FixNameErrorLLM import FixNameErrorLLM
from main_pipeline.renote_utils.nb_utils import StaticAST, install_missing_module


def test_notebook_module_file_execution(nb_path, repo_path, post_fix):
    print_msg("[1/3] Test modules", 3)
    test_modules = TestModule(nb_path, repo_path)
    module_results = test_modules.analyze()
    print_msg(f"☑️ Found {len(module_results)} module(s)", 4)

    print_msg("[2/3] Test linear execution", 3)
    test_execution = TestExecution(nb_path)
    execution_results = test_execution.analyze()

    if execution_results['linear'] == "timeout":
        execution_results['linear'] = 'failed'
        raise TimeoutError


    try:
        print_msg("[3/3] Test file/folder paths", 3)
        test_files = TestFiles(nb_path)
        file_results = test_files.analyze()

        found_files = file_results.get('found_files', set())
        unfound_files = file_results.get('unfound_files', set())
        invalid_files = file_results.get('invalid_files', set())

        formatted_file_results = {}
        for f in found_files | unfound_files | invalid_files:
            if f in invalid_files:
                formatted_file_results[f] = "warn"
            elif f in unfound_files:
                formatted_file_results[f] = "failed"
            else:
                formatted_file_results[f] = "success"
        
        formatted_results = {
            "modules": module_results,
            "execution": execution_results,
            "files": formatted_file_results,
        }

        print()
        test_summary = print_test_result(formatted_results, 3)
        formatted_results['test_summary'] = test_summary
        formatted_results['files'] = file_results

        cache_result = generate_cache_result(formatted_results, post_fix)
        return cache_result
    except TimeoutError:
        raise TimeoutError
    except Exception as e:
        raise e

def group_modules_by_status(data):
    '''
    Group modules by their status.
    Args:
        data (dict): A dictionary where keys are module names and values are dictionaries containing module information
    Returns:
        dict: A dictionary where keys are statuses ('success', 'failed', 'warning', 'unknown') and values are lists of module names with that status.
    '''
    grouped = defaultdict(list)
    for module, info in data.items():
        status = info.get('status', 'unknown')
        grouped[status].append(module)
    return grouped


def generate_cache_result(result, post_fix):
    '''
    Generate the result of notebook analysis into the results cache.
    Args:
        result (dict): The result of the notebook analysis.
    '''
    aggregated_result = {}

    # Modules
    modules = result.get('modules', {})
    grouped_modules = group_modules_by_status(modules)

    for status, mods in grouped_modules.items():
        aggregated_result.setdefault(f'{status}_modules_{post_fix}', []).extend(mods)

    # Execution
    execution = result.get('execution', {})
    aggregated_result[f'linear_{post_fix}'] = execution.get('linear', None)
    aggregated_result[f'execution_patterns_{post_fix}'] = execution.get('patterns', [])

    # Files
    file_results = result.get('files', {})
    aggregated_result[f'found_files_{post_fix}'] = list(file_results.get('found_files', set()))
    aggregated_result[f'unfound_files_{post_fix}'] = list(file_results.get('unfound_files', set()))
    aggregated_result[f'invalid_files_{post_fix}'] = list(file_results.get('invalid_files', set()))

    # Test summary
    test_summary = result.get('test_summary', {})

    success = test_summary.get('success', 0)
    failed = test_summary.get('failed', 0)
    warning = test_summary.get('warning', 0)
    total = success + failed

    aggregated_result[f'total_test_{post_fix}'] = total
    aggregated_result[f'success_test_{post_fix}'] = success
    aggregated_result[f'failed_test_{post_fix}'] = failed
    aggregated_result[f'warning_test_{post_fix}'] = warning

    return aggregated_result

def summarize_fix_progress(all_exec_results):
    total_cell_ex_after_file_fix = 0
    total_cell_ex_after_module_fix = 0
    total_cell_ex_after_name_fix = 0

    total_module_not_found = 0
    total_file_not_found = 0
    total_name_error = 0

    last_name_error_found = None

    all_unique_errors_during_execution = list(set([d['status'] for d in all_exec_results]))
    for i in range(len(all_exec_results) - 1):
        d1 = all_exec_results[i]
        j = i
        d2 = all_exec_results[j]
        for j in range(i + 1, len(all_exec_results)):
            if d2['err_cell_num'] == d1['err_cell_num']:
                d2 = all_exec_results[j]
            else:
                break

        if d1['status'] == 'ModuleNotFoundError':
            total_cell_ex_after_module_fix += (d2['err_cell_num'] - d1['err_cell_num'])
            total_module_not_found += 1
        elif d1['status'] == 'FileNotFoundError':
            total_cell_ex_after_file_fix += (d2['err_cell_num'] - d1['err_cell_num'])
            total_file_not_found += 1
        elif d1['status'] == 'NameError':
            true_cell_count = d2['err_cell_num']
            if d1['NameError_type'] == 'undefined':
                true_cell_count = d2['err_cell_num'] - 1

            increase = true_cell_count - d1['err_cell_num']
            total_cell_ex_after_name_fix += increase
            if increase > 0:
                last_name_error_found = d1
            total_name_error += 1

    results = {
        'total_cell_ex_after_file_fix': total_cell_ex_after_file_fix,
        'total_cell_ex_after_module_fix': total_cell_ex_after_module_fix,
        'total_cell_ex_after_name_fix': total_cell_ex_after_name_fix,
        'all_unique_errors_during_execution': all_unique_errors_during_execution,
        'total_module_not_found': total_module_not_found,
        'total_file_not_found': total_file_not_found,
        'total_name_error': total_name_error,
    }

    if total_cell_ex_after_name_fix <= 0 and last_name_error_found is not None:
        results['last_name_error_found'] = last_name_error_found

    return results

def execute_with_module_file_name_fixes(nb_path, tmp_env_dir):
    all_exec_results = []
    missing_files_paths = set()
    missing_files_paths_to_remove = set()
    name_fixed_paths = set()
    missing_modules = set()
    successfully_installed_modules = set()
    err_in_file_creation = None
    total_module_fixing_llm = 0
    success_module_fixing_llm = 0
    ast_status = []
    name_error_count = 0
    name_err_exec = []

    # Initial Execution
    exec_r = ExecuteNotebook(nb_path).execute_notebook()
    all_exec_results.append(exec_r)

    break_loop = False

    while True:
        # Case 1: File not found, create the file, and re-run the notebook
        # if the file is already created, then break the loop
        if 'FileNotFoundError_path' in exec_r:
            missing_file_path = exec_r['FileNotFoundError_path']
            print_msg(f"🔍 Found FileNotFoundError path: {missing_file_path} in cell {exec_r['err_cell_num']}", 3)

            if not missing_file_path:
                err_in_file_creation = f'Fix it. Failed to find missing path'
                print_msg(f"❌ Failed to find missing path. Breaking the loop", 4)
                break

            print_msg(f'➡️ Fixing FileNotFoundError: {missing_file_path}', 4)

            if missing_file_path in missing_files_paths:
                err_in_file_creation = f'Fix it. File creation problem with {missing_file_path}'
                print_msg(f"❌ Failed to create file {missing_file_path}, already tried before, breaking the loop", 4)
                break

            missing_files_paths.add(missing_file_path)
            f = FixFileNotFound(nb_path, exec_r)
            create_status = f.create_missing_file()

            if f.generated_path is not None:
                missing_files_paths_to_remove.add(f.generated_path)

            if create_status:
                print_msg(f"🔄 File {missing_file_path} created successfully, re-running the notebook", 4)
                exec_r = ExecuteNotebook(nb_path).execute_notebook()
                all_exec_results.append(exec_r)
            else:
                err_in_file_creation = f'Fix it. File creation problem with {missing_file_path}'
                print_msg(f"❌ Failed to create file {missing_file_path}. Breaking the loop", 4)
                break

        # Case 2: Module not found, install the module and re-run the notebook
        # if the module is already installed OR can't be installed, then break the loop
        elif 'missing_module' in exec_r:
            m = exec_r['missing_module']
            print_msg(f"🔎 Found ModuleNotFoundError for {m} in cell {exec_r['err_cell_num']}", 3)

            if m == "unknown":
                print_msg(f"❌ Failed to find module name. Breaking the loop", 4)
                break

            print_msg(f"➡️ Fixing ModuleNotFoundError: {m}", 4)
            
            if m not in missing_modules:
                missing_modules.add(m)
                result_code = install_missing_module(m, tmp_env_dir)

                if result_code != 0:
                    fix_module = FixModuleNotFound(m)
                    correct_module = fix_module.find_correct_module()
                    total_module_fixing_llm += 1

                    if correct_module is not None:
                        missing_modules.add(correct_module)
                        returncode = install_missing_module(correct_module, tmp_env_dir)

                        if returncode == 0:
                            successfully_installed_modules.add(correct_module)
                            success_module_fixing_llm += 1
                        else:
                            print_msg(f'❌ Failed to install {correct_module}, breaking the loop', 4)
                            break
                else:
                    successfully_installed_modules.add(m)

                print_msg(f"🔄 Module {m} installed successfully, re-running the notebook", 4)
                exec_r = ExecuteNotebook(nb_path).execute_notebook()
                all_exec_results.append(exec_r)
            else:
                print_msg(f'❌ Failed to install {m}, already tried before, breaking the loop', 4)
                break

        # Case 3: NameError, fix the error
        elif 'undefined_var' in exec_r:
            undefined_var = exec_r['undefined_var']
            undefined_var_cell = exec_r['err_cell_num']
            print_msg(f"🔎 Found NameError for {undefined_var} in cell {undefined_var_cell}", 3)
            print_msg(f"➡️ Fixing NameError: {undefined_var} in cell {undefined_var_cell}", 4)

            if name_error_count > 0:
                last_err = name_err_exec[-1]
                last_err_cell = last_err['err_cell_num']
                last_err_var = last_err['undefined_var']

                # Case 1: The same variable is still undefined, or error is moving backward
                if undefined_var_cell < last_err_cell or undefined_var == last_err_var:
                    print_msg(f"❌ NameError is not improving: {undefined_var} in cell {undefined_var_cell}. Breaking the loop.", 4)
                    break

                # Case 2: Check for circular unresolved variable references
                for prev_err in reversed(name_err_exec):
                    if undefined_var == prev_err['undefined_var']:
                        print_msg(f"❌ Potential circular reference for variable '{undefined_var}'. Breaking the loop.", 4)
                        break_loop = True
                        break
                
            if break_loop:
                break

            # Static AST Analysis to locate definition of the variable
            staticAST = StaticAST(nb_path)
            result = staticAST.findOneVariableDefinition(undefined_var, undefined_var_cell)

            if result is None:
                print_msg(f"❌ Unable to resolve NameError for {undefined_var} in cell {undefined_var_cell}. Breaking the loop.", 4)
                break

            err_type, defined_cell = result
            exec_r['NameError_type'] = err_type
            ast_status.append(err_type)
            name_err_exec.append(exec_r)
            name_error_count += 1

            n = FixNameErrorLLM(nb_path, undefined_var, defined_cell, undefined_var_cell)

            # If the variable is undefined, then fix the NameError with LLM
            if err_type == "undefined" or defined_cell == undefined_var_cell:
                print_msg(f"➡️ Generating definition for {undefined_var} in cell {undefined_var_cell}", 4)
                nb_path = n.fix_name_error_and_save()
            # If the variable is defined after the cell, then reorder the cells
            elif err_type == "defined_after":
                print_msg(f"➡️ Found definition of {undefined_var} in {defined_cell}", 4)
                nb_path = n.get_reordered_notebook_path()

            print_msg(f"🆕 New path generated: {nb_path}", 4)
            name_fixed_paths.add(nb_path)

            # Rerun the notebook
            print_msg(f"🔄 Re-running the notebook after fixing NameError", 4)
            exec_r = ExecuteNotebook(nb_path).execute_notebook()
            all_exec_results.append(exec_r)

        # Case 4: No error or other ERR, break the loop
        else:
            if exec_r['status'] == 'executable':
                print_msg(f"✅ Notebook executed successfully", 3)
            elif exec_r['status'] == "TimeoutError":
                print_msg(f"🔎 Found TimeoutError", 3)
            else:
                err_cell_num = exec_r.get('err_cell_num', 'unknown')
                status = exec_r.get('status', 'unknown')
                print_msg(f"🔎 Found other error: {status} in cell {err_cell_num}", 3)
            break
        print()

    return_ast_status = ""
    if not ast_status:
        return_ast_status = "no_undefined"
    elif {"undefined", "defined_after"}.issubset(set(ast_status)):
        return_ast_status = "both"
    else:
        return_ast_status = ast_status[0]

    nb_exec_result = {
        'all_exec_results': all_exec_results,
        'err_in_file_creation': err_in_file_creation,
        'return_ast_status': return_ast_status,
        'total_module_fixing_llm': total_module_fixing_llm,
        'success_module_fixing_llm': success_module_fixing_llm,
        'installed_modules': successfully_installed_modules,
        'current_nb_path': nb_path,
        'missing_files_paths_to_remove': missing_files_paths_to_remove,
        'name_fixed_paths': name_fixed_paths,
    }

    return nb_exec_result

def process_nb(nb_path, repo_path, tmp_env_dir):
    """
    Process the notebook and return the results, including:
    1. Read the notebook and get the code cells
    2. AST analysis: parse the code cells
    3. If AST analysis is not successful, then do the execution
    4. Fix the import error and file error
    5. Aggregate the results
    6. Return the results
    """
    paper_results = {}

    # Step 1: Pre-execution test
    print_msg("1️⃣ Pre-execution test", 2)
    try:
        pre_test_result = test_notebook_module_file_execution(nb_path, repo_path, "pre")
    except TimeoutError:
        print_msg(f"❌ Timeout during pre-execution test", 3)
        pre_test_result = {"error": "Timeout"}
    except Exception as e:
        print_msg(f"❌ Error during pre-execution test: {e}", 3)
        pre_test_result = {"error": str(e)}

    # Step 2: Attempt execution + fixing
    print_msg(f"2️⃣ Initial execution and fixing of the notebook (if needed)", 2)
    result = execute_with_module_file_name_fixes(nb_path, tmp_env_dir)

    # Step 3: Unpack execution results
    all_fix_errors_results = result['all_exec_results']
    file_creation_error = result['err_in_file_creation']
    ast_status = result['return_ast_status']
    total_module_fixing_llm = result['total_module_fixing_llm']
    success_module_fixing_llm = result['success_module_fixing_llm']
    installed_modules = result['installed_modules']
    current_nb_path = result['current_nb_path']
    missing_files_paths_to_remove = result['missing_files_paths_to_remove']
    name_fixed_paths = result['name_fixed_paths']

    # Step 4: Summarize intermediate fix stats
    agg_results = summarize_fix_progress(all_fix_errors_results)

    # Step 5: Extract initial and final execution snapshots
    initial_exec_result = all_fix_errors_results[0]
    result_dict_after_all_fixes = copy.deepcopy(all_fix_errors_results[-1])
    final_exec_result = result_dict_after_all_fixes

    # Step 6: Prepare execution summary
    def get_max_cell_num(exec_result):
        if exec_result['err_cell_num'] > 0 and exec_result['status'] != "executable":
            return exec_result['err_cell_num'] - 1
        return exec_result['err_cell_num']

    renote_result = {
        'Initial Total Code Cells': initial_exec_result['total_code_cells'],
        'Initial_Status': initial_exec_result['status'],
        'Initial_max_execute_cells': get_max_cell_num(initial_exec_result),

        'Final Total Code Cells': final_exec_result['total_code_cells'],
        'Final_Status': final_exec_result['status'],
        'Final_max_execute_cells': get_max_cell_num(final_exec_result),

        'FileCreationError (Manual)': file_creation_error,
        'ast_status': ast_status,
        'total_module_fixing_using_llm': total_module_fixing_llm,
        'success_module_fixing_using_llm': success_module_fixing_llm,
        'missing_modules': installed_modules,
    }

    # Step 7: Add derived metrics
    renote_result['Increased_execution_cells'] = renote_result['Final_max_execute_cells'] - renote_result['Initial_max_execute_cells']
    renote_result['Increased_exection_percentage'] = (
        (renote_result['Final_max_execute_cells'] / renote_result['Final Total Code Cells']) * 100 - 
        (renote_result['Initial_max_execute_cells'] / renote_result['Initial Total Code Cells']) * 100
    )

    # Step 8: Combine with fix stats
    renote_result = {**renote_result, **agg_results}

    # Step 9: Post-execution test
    print()
    print_msg("3️⃣ Post-execution test", 2)
    if pre_test_result.get('error') == "Timeout":
        print_msg(f"❌ Skipping post-execution test due to pre-execution timeout", 2)
        post_test_result = pre_test_result
    else:
        try:
            post_test_result = test_notebook_module_file_execution(current_nb_path, repo_path, "post")
        except Exception as e:
            print_msg(f"❌ Error during post-execution test: {e}", 2)
            post_test_result = {"error": str(e)}

    paper_results = {
        "pre_test_result": pre_test_result,
        "renote_result": renote_result,
        "post_test_result": post_test_result,
        'missing_files_paths_to_remove': missing_files_paths_to_remove,
        'name_fixed_paths': name_fixed_paths
    }

    return paper_results
