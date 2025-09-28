import nbformat
import ast
import os
import subprocess
import json
import sys

from main_pipeline.renote_utils.ast_visit import ASTNodeVisitor
from main_pipeline.all_utils.print_format import print_msg


################################################################################
####### Utility functions for reading and checking validity of notebooks #######
################################################################################

def read_notebook(nb_path):
    """
    Reads a Jupyter notebook file and returns its content.
    Args:
        nb_path (str): Path to the Jupyter notebook file.
    Returns:
        dict: The content of the notebook as a dictionary.
    """
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            return nbformat.read(f, as_version=4)
    except Exception as e:
        print(f"CAN'T OPEN NOTEBOOK {nb_path}: {e}")
        return None


def get_code_cells_with_nb_path(nb_path):
    """
    Reads a Jupyter notebook and returns its code cells.
    Args:
        nb_path (str): Path to the Jupyter notebook file.
    Returns:
        list: A list of code cells in the notebook.
    """
    nb = read_notebook(nb_path)
    return extract_code_cells(nb)


def is_code_cell(cell):
    """
    Check if a cell is a code cell.
    Args:
        cell (dict): A cell from a Jupyter notebook.
    Returns:
        bool: True if the cell is a code cell, False otherwise.
    """
    return cell.get("cell_type") == "code"

def is_cell_empty(cell):
    """ 
    Check if a cell is empty.
    Args:
        cell (dict): A cell from a Jupyter notebook.
    Returns:
        bool: True if the cell is empty, False otherwise.
    """
    source = cell.get("source")
    if not source:
        return True
    if isinstance(source, str):
        return not source.strip().replace(" ", "")
    elif isinstance(source, (list, tuple, set)):
        return not ''.join(source).strip().replace(" ", "")
    return True

def extract_code_cells(nb):
    """ 
    Extract all code cells from a Jupyter notebook.
    Args:
        nb (dict): The content of a Jupyter notebook.
    Returns:
        list: A list of code cells, excluding empty ones.
    """
    return [cell for cell in nb.get("cells", []) if is_code_cell(cell) and not is_cell_empty(cell)]


def get_cell_source_code(cell):
    """
    Extract the source code from a code cell, excluding comments and magic commands.
    Args:
        cell (dict): A code cell from a Jupyter notebook.
    Returns:
        str: The source code of the cell, excluding comments and magic commands.
    """
    lines = cell.get('source', '').splitlines()
    return "\n".join(line for line in lines if not line.startswith(("!", "%", "#", "$", "-"))).rstrip()

    
def get_notebook_language(notebook_path):
    if os.path.getsize(notebook_path) == 0:
        print(f"Notebook file {notebook_path} is empty.")
        return None
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        try:
            notebook = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON from {notebook_path}: {e}")
            return None
    
    metadata = notebook.get('metadata', {})
    kernelspec = metadata.get('kernelspec', {})
    language_info = metadata.get('language_info', {})
    
    return (
        language_info.get('name', 'unknown'),
        language_info.get('version', 'unknown'),
        kernelspec.get('name', 'unknown')
    )

def check_notebook_validity(nb_path):  
    """
    Check if a Jupyter notebook is valid and contains executable Python code.
    Args:
        nb_path (str): Path to the Jupyter notebook file.
    Returns:
        tuple: A tuple containing a boolean indicating validity and a message.
        Returns True and "Success" if the notebook is valid, otherwise returns False and an error message.
    Raises:
        json.JSONDecodeError: If the notebook file is not a valid JSON.
    """
    nb = read_notebook(nb_path) 

    # 1. If we cannot read the notebook file, we will move it to the error directory
    if nb is None:
        print(f"Cannot read the notebook file {nb_path}")
        return False, "Cannot read"
    
    # 2. Check if the notebook has code cells
    code_cells = extract_code_cells(nb)
    if not code_cells:
        print(f"No code cells in the notebook {nb_path}")
        return False, "No code cells"

    # 3. Check the language of the notebook. If not Python, we will move it to the error directory
    language = get_notebook_language(nb_path)
    if language is None:
        return False, "Cannot read"
    
    language_name, version, kernel_name = language
    version = str(version)

    if (
        version == "unknown" and 'python3' not in kernel_name.lower()
        or not version.startswith("3")
        or 'python' not in language_name.lower()
    ):
        # print(f"Language is not Python 3 ({language_name} {version}) in the notebook {nb_path}")
        return False, "Non-Python"

    # 4. Check if the code cells are valid python code
    for cell in nb['cells']:
        if is_code_cell(cell) and cell["source"]:
            source_code = get_cell_source_code(cell)
            if source_code.strip():
                try:
                    ast.parse(source_code)
                except Exception as e:
                    print(f"AST Parsing Error during read_notebook in the notebook {nb_path}")
                    return False, "AST Parsing Error"

    return True, "Success"


################################################################################
#### Utility function for installing missing module to working environment #####
################################################################################

def install_missing_module(missing_module, tmp_env_dir):
    print_msg(f"▶️ Installing missing module: {missing_module}", 4)
    # if missing_module == "torch":
    #     print_msg("❗️ Skipping installation for torch", 4)
    #     return 1
    try:
        env = os.environ.copy()
        env['TMPDIR'] = tmp_env_dir
        env['PIP_TMPDIR'] = tmp_env_dir
        env['PYTHON_EGG_CACHE'] = tmp_env_dir  # needed for some wheel builds too
        env['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

        # r =  subprocess.run(
        #     f"pip install --no-cache-dir --quiet {missing_module}", 
        #     shell=True, 
        #     env=env,
        #     text=True
        # ) # Add capture_output=True to capture output and hide it from screen

        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--quiet", missing_module],
            env=env,
            text=True,
            capture_output=True
        )

        if r.returncode == 0:
            print_msg(f"✅ Successfully installed {missing_module}", 4)
            return 0
        else:
            print_msg(f"❌ Error installing {missing_module}", 4)
            return r.returncode
    except Exception as e:
        print_msg(f"❌ Exception during installation process: {e}", 4)
        return -1


################################################################################
######## Utility function for AST static analysis of Jupyter notebooks #########
################################################################################

class StaticAST:
    def __init__(self, nb_path):
        self.nb_path = nb_path
        # Store detailed variable usage information
        self.variable_uses = {}  # Format: {cell_number: {variable_name: [scope_ids]}}
        self.variable_defs = {}  # Format: {cell_number: {variable_name: [scope_ids]}}

    def _get_notebook_cells(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return nbformat.read(f, as_version=4)['cells']
        except Exception:
            print_msg(f"❌ Failed to open notebook: {path}", 4)
            return None

    def _analyze_cell_ast(self, cell_source, global_scope, cell_number):
        try:
            tree = ast.parse(cell_source)
            print_msg(f"📊 Parsed AST for cell {cell_number}", 5)
        except Exception:
            print_msg(f"❌ Failed to parse cell {cell_number} in notebook {self.nb_path}", 5)
            return None

        visitor = ASTNodeVisitor()
        visitor.scopes = [global_scope, {}]
        def_list, use_list = visitor.analyze(tree)
        
        # Update global scope with cell's definitions
        global_scope.update(visitor.scopes[-1])
        
        # Store detailed variable usage information
        self.variable_uses[cell_number] = {}
        self.variable_defs[cell_number] = {}
        
        # Process definitions - def_list is already {scope_id: [variables]} format
        for scope_id, vars_defined in def_list.items():
            for var in vars_defined:
                self.variable_defs[cell_number].setdefault(var, []).append(scope_id)

        # Process uses - use_list is already {scope_id: [variables]} format
        for scope_id, vars_used in use_list.items():
            for var in vars_used:
                self.variable_uses[cell_number].setdefault(var, []).append(scope_id)

        return def_list, use_list, global_scope

    def _is_scope_accessible(self, def_scope, use_scope):
        """
        Determine if a definition in def_scope is accessible from use_scope.
        """
        def_cell, def_scope_id = def_scope
        use_cell, use_scope_id = use_scope
        
        # Different cells - only global scope definitions are accessible
        if def_cell != use_cell:
            return def_scope_id == 0
            
        # Same cell - check scope hierarchy
        return def_scope_id <= use_scope_id

    def _find_variable_use_scopes(self, variable, cell_number):
        """
        Find all scopes where a variable is used in a specific cell.
        Returns list of scope IDs.
        """
        if cell_number in self.variable_uses:
            return self.variable_uses[cell_number].get(variable, [])
        return []

    def analyze_notebook(self):
        """
        Analyze the entire notebook to build the variable use and def maps.
        Should be called before finding variable definitions.
        """
        global_scope = {}
        cells = self._get_notebook_cells(self.nb_path)
        if cells is None:
            return False
            
        valid_cell_count = 0
        for cell in cells:
            if cell['cell_type'] == 'code':
                if cell["source"] is None:
                    continue
                    
                cell_content = ''.join(cell['source'].split())
                if cell_content:
                    valid_cell_count += 1
                    source_code = get_cell_source_code(cell)
                    result = self._analyze_cell_ast(source_code, global_scope, valid_cell_count)
                    
                    if result is None:
                        return False
        return True

    def findOneVariableDefinition(self, target_variable, use_cell):
        """
        Find the definition of a variable used in a cell, considering all scopes where it's used.
        
        Args:
            target_variable: the variable to find
            use_cell: the cell number where the variable is used
            
        Returns:
            tuple: (status, cell_number) where status is one of:
                - "defined_after" if the variable is defined after the use in an accessible scope
                - "undefined" if the variable is never defined in an accessible scope
        """
        if not self.analyze_notebook():  # Make sure we analyze the notebook first
            return None

        # Get all scopes where the variable is used in the specified cell
        use_scopes = self._find_variable_use_scopes(target_variable, use_cell)
        
        if not use_scopes:
            print(f"Warning: No uses found for variable '{target_variable}' in cell {use_cell}")
            return "undefined", -1
            
        # Check each use scope
        later_defs = []
        for use_scope_id in use_scopes:
            use_location = (use_cell, use_scope_id)
            
            # Check all cells for definitions
            for def_cell in self.variable_defs:
                if target_variable in self.variable_defs[def_cell]:
                    for def_scope_id in self.variable_defs[def_cell][target_variable]:
                        def_location = (def_cell, def_scope_id)
                        
                        # If definition is after use and in accessible scope
                        if def_cell > use_cell and self._is_scope_accessible(def_location, use_location):
                            later_defs.append((def_cell, def_scope_id))
        
        if later_defs:
            # Return the earliest accessible definition after use
            earliest_def = min(later_defs, key=lambda x: (x[0], x[1]))
            return "defined_after", earliest_def[0]
            
        return "undefined", -1


############################################################################################################

# class ReadNB:
#     def __init__(self, nb_path):
#         self.nb_path = nb_path
#         self.nb_content = None

#     def readNB(self):
#         """
#         Read a notebook file and return the code cells
#         :param nb_path: path to the notebook file
#         :return: list of code cells
#         """
#         try:
#             with open(self.nb_path, 'r', encoding='utf-8') as f:
#                 self.nb_content = nbformat.read(f, as_version=4)
#                 return self.nb_content
#         except Exception as e:
#             print(f"CAN'T OPEN NOTEBOOK: {e}")
#             # print("----> Error reading notebook:", self.nb_path)
#             return None

#     def readCodeCells(self):
#         code_cells = []
#         for cell in self.nb_content['cells']:
#             if 'cell_type' in cell:
#                 if cell['cell_type'] == 'code':
#                     if not self._is_empty(cell):
#                         code_cells.append(cell)

#         return code_cells
    
#     def _is_empty(self, cell):
#         if cell["source"] is None:
#             return True
#         else:
#             source_code = ''
#             if isinstance(cell['source'], (list, tuple, set)):
#                 source_code = ''.join(cell['source']).strip().replace(" ", "")
#             elif isinstance(cell['source'], str):
#                 source_code = cell['source'].strip().replace(" ", "")
            
#             return source_code == '' # return True if the cell is empty
    
#     def getKernelInfo(self):    
#         kernel_info = self.nb_content.get('metadata', {}).get('kernelspec', {})
#         if kernel_info:
#             kernel_name = kernel_info.get('name', 'Unknown').lower()
#             return kernel_name
#         else:
#             return 'Unknown'

#     def getTotalCodeCells(self):
#         nb_content = self.readNB()
       
#         # 1. If we cannot read the notebook file, we will move it to the error directory
#         if nb_content is None:
#             print(f"Cannot read the notebook file {nb_path}")
#             return None
        
#         # 2. Check if the notebook has code cells
#         code_cells = self.readCodeCells()
#         total_code_cells = len(code_cells)
#         if total_code_cells == 0:
#             print(f"No code cells in the notebook {nb_path}")
#             return None
#         return total_code_cells


# class ReOrderCellsTempNBForDefinedAfter:
#     def __init__(self, nb_path, defined_index, undefined_index):
#         self.nb_path = nb_path
#         self.defined_index = defined_index - 1
#         self.undefined_index = undefined_index - 1

#     def swapCells(self, notebook, cell1_index, cell2_index):
#         if cell1_index == 0:
#             cell_to_move = notebook.cells.pop(cell2_index)
#             notebook.cells.insert(0, cell_to_move)
#         else:
#             cell1_pred = cell1_index - 1
#             notebook.cells[cell1_pred], notebook.cells[cell2_index] = notebook.cells[cell2_index], notebook.cells[cell1_pred]

#         return notebook

#     def get_reordered_notebook_path(self):
#         content = nbformat.read(self.nb_path, as_version=4)
#         new_content = self.swapCells(content, self.undefined_index, self.defined_index)

#         nb_name = os.path.basename(self.nb_path)
#         output_nb_name = nb_name.replace(".ipynb", "_reordered_temp.ipynb")
#         new_notebook_path = os.path.join(os.path.dirname(self.nb_path), output_nb_name)
#         # new_ordered_nb = self.getReorderedNBFile()
#         with open(new_notebook_path, "w", encoding="utf-8") as f:
#             nbformat.write(new_content, f)
#         return new_notebook_path


# def getReOrderedNB(sorter, nb_path):
#     # 2. Get values of attributes of the sorter
#     defined_index = sorter.defined_index
#     older_cells = sorter.new_code_cells
#     undefined_index = sorter.undefined_index

#     # 3. Generate a file with reordered cells
#     re = ReOrderCellsTempNBForDefinedAfter(older_cells, defined_index, undefined_index)
#     new_ordered_nb = re.getReorderedNBFile()


#     # # 4. Sort it again to see if it still has undefined vars
#     # sub_sorter = RenoteAST(new_ordered_nb.cells)
#     # sub_sorter.run()
#     # new_exe = sub_sorter.undefined_index

#     new_notebook_path = nb_path.replace(".ipynb", "_reordered_temp.ipynb")

#     with open(new_notebook_path, "w", encoding="utf-8") as f:
#         nbformat.write(new_ordered_nb, f)

#     print(f"> Running for Rordered Cells is successful")
#     return new_notebook_path



############################################################################################################
# def assign_id(cell):
#     if 'id' in cell['metadata']:
#         return cell['metadata']['id']
#     elif 'id' in cell:
#         return cell['id']
#     else:
#         return str(uuid.uuid4())


# class Cell:
#     def __init__(self, cell):
#         self.successor = None  # Cell
#         self.cell_id = assign_id(cell)  # string
#         self.source = cell['source']  # string
#         self.spacial_order = 0
#         self.def_list = {}  # dict
#         self.use_list = {}  # dict

#     def set_successor(self, cell):
#         self.successor = []
#         if cell is not None:
#             self.successor = cell

#     def set_spacial_order(self, value):
#         self.spacial_order = value

############################################################################################################

# class RenoteAST:
#     def __init__(self, code_cells):
#         self.code_cells = code_cells
#         self.new_code_cells = []
#         self.undefined_vars_dict = None

#         self.destination = ""
#         self.undefined_var = ""
#         self.undefined_index = -1
#         self.defined_index = -1

    
#     def _create_cell(self, cell, def_list, use_list):
#         c = Cell(cell)
#         c.set_spacial_order(len(self.new_code_cells))
#         c.def_list = def_list
#         c.use_list = use_list
#         self.new_code_cells.append(c)
    
#     def _find_def_use(self, cell):
#         source_code = self._get_source_code(cell)
#         # try:
#         root = ast.parse(source_code)
#         # except:
#             # return None
#         visitor = ASTNodeVisitor()
#         visitor.visit(root, scope=0)

#         def_list = visitor.def_list
#         use_list = visitor.use_list
#         return def_list, use_list

#     def _get_source_code(self, cell):
#         source = ''''''
#         lines = cell['source'].splitlines()
#         for line in lines:
#             if not line.startswith(("!", "%", "#", "$", "-")):
#                 source += line + "\n"
#         return source
    
#     def _getPrevDefinedVars(self, index, scope):
#         total_def = []
#         cell = self.new_code_cells[index]  # current cell
#         for c in self.new_code_cells[:index]:
#             if 0 in c.def_list:
#                 total_def.extend(c.def_list[0])
#         i = 0
#         while i <= scope:
#             total_def.extend(cell.def_list[i])
#             i += 1

#         return total_def
    
#     def _setUndefinedVars(self):
#         undefined = {}
#         for index, cell in enumerate(self.new_code_cells):
#             use_vars_list = list(cell.use_list.items())
#             if not undefined.get(index):
#                 undefined[index] = set()
#             for scope, use_vars in use_vars_list:
#                 def_vars = self._getPrevDefinedVars(index, scope)
#                 for use_var in use_vars:
#                     if use_var not in def_vars:
#                         undefined[index].add(use_var)
#         self.undefined_vars_dict = undefined
    

#     def get_post_defined_vars(self, index):
#         defined = {}
#         for c in self.new_code_cells[index + 1:]:
#             c_index = self.new_code_cells.index(c)
#             if c_index not in defined:
#                 defined[c_index] = []
#                 if 0 in c.def_list:
#                     defined[c_index].extend(c.def_list[0])
#         return defined
    
#     def _getPostDefinedVarsArray(self, index):
#         total_def = []
#         defined = self.get_post_defined_vars(index)
#         for k in defined.keys():
#             total_def.extend(defined[k])
#         return total_def

#     def _getDefinedIndex(self, index):
#         defined = self.get_post_defined_vars(index)
#         for k, v in defined.items():
#             if self.undefined_var in v:
#                 return k
    
#     def _checkDefUse(self):
#         undefined_var_presence = False
#         for i in self.undefined_vars_dict.keys():
#             if len(self.undefined_vars_dict[i]) != 0:
#                 undefined_var_presence = True
        
#         if not undefined_var_presence:
#             self.destination = "no_undefined"
#             self.undefined_index = len(self.new_code_cells)
#             return
        

#         for i, undefined_vars in self.undefined_vars_dict.items():
#             undefined = False
#             def_after_use = False

#             def_list = self._getPostDefinedVarsArray(i)

#             for undefined_var in undefined_vars:
#                 if undefined_var in def_list:
#                     print("Def after", undefined_var)
#                     if self.undefined_var == "":
#                         self.undefined_var = undefined_var
#                         self.defined_index = self._getDefinedIndex(i)
#                     if self.undefined_index == -1:
#                         self.undefined_index = i
#                     def_after_use = True
#                 else:
#                     print("Undefined", undefined_var)
#                     if self.undefined_index == -1:
#                         self.undefined_index = i
#                     undefined = True

#             if undefined and def_after_use:
#                 self.destination = "both"
#                 return
#             elif undefined and not def_after_use:
#                 self.destination = "undefined"
#                 return
#             elif not undefined and def_after_use:
#                 self.destination = "defined_after"
#                 return
    
#     def _fooUndefinedVariable(self):
#         # check the undefined variable in the notebook
#         self._setUndefinedVars()
#         self._checkDefUse()

#     def run(self):
#         for cell in self.code_cells:
#             t = self._find_def_use(cell)    
#             def_list, use_list = t
#             self._create_cell(cell, def_list, use_list)        
#         self._fooUndefinedVariable()
        

############################################################################################################