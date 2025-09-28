import re
import os
import papermill as pm
from pydantic import BaseModel

from main_pipeline.renote_utils.nb_utils import get_code_cells_with_nb_path
from main_pipeline.renote_utils.localOllama import localchat
from main_pipeline.all_utils.print_format import print_msg


def papermill_execution(orignal_nb_path):
    notebook_dir = os.path.dirname(orignal_nb_path)
    print_msg("▶️ Executing notebook with papermill", 3)
    try:
        pm.execute_notebook(
            input_path = orignal_nb_path,
            output_path = None, #temp_nb_output_path,
            timeout=300,  # time out in second
            kernel_name="python3",
            progress_bar=False,
            cwd=notebook_dir
        )
    except Exception as e:
        print_msg('❌ Papermill execution failed', 3)
        raise e


class ErrCellNumber(BaseModel):
    err_cell_num: int
    err_type: str

class ExecuteNotebook:
    def __init__(self, nb_path):
        code_cells = get_code_cells_with_nb_path(nb_path)
        self.total_code_cells = len(code_cells)
        self.original_nb_path = nb_path

    def _get_error_cell_index_and_type_llm(self, err):
        print_msg(f"⁉️ Error Found:", 3)
        for line in err.splitlines():
            print_msg(f"{line}", 4)
        print()

        # First, try manual extraction
        manual_cell_num = self._get_error_cell_index(err)
        manual_err_type = self._extract_error_type(err)
        
        # Check if manual extraction was successful
        if manual_cell_num != -1 and manual_err_type is not None:
            print_msg(f"✅ Manual extraction successful: {manual_err_type} in cell {manual_cell_num}", 3)
            return manual_cell_num, manual_err_type
        
        # If manual extraction failed or incomplete, fall back to LLM
        print_msg(f"⚠️ Manual extraction incomplete (cell: {manual_cell_num}, type: {manual_err_type}), trying LLM...", 3)
        
        prompt = (
            f"Identify the error name (1 word, no quotes, for example: NameError) and cell index (for example, integer from Cell In[2]) where it happened from the error report below.\n\n{err}"
        )
        
        try:
            response = localchat(prompt, ErrCellNumber)
            
            llm_err_cell_num = getattr(response, 'err_cell_num', None)
            llm_err_type = getattr(response, 'err_type', None)
            
            if llm_err_cell_num is None or llm_err_type is None:
                print_msg(f"⚠️ LLM extraction also failed", 3)
                # Return best available results, preferring manual over None
                final_cell_num = manual_cell_num if manual_cell_num != -1 else 0
                final_err_type = manual_err_type if manual_err_type else "UnknownError"
                return final_cell_num, final_err_type
            
            llm_err_cell_num = int(llm_err_cell_num)
            
            # Use LLM results but validate against manual results if available
            final_cell_num = llm_err_cell_num
            final_err_type = llm_err_type.strip() if isinstance(llm_err_type, str) else "UnknownError"
            
            # Cross-validate with manual results if they were partially successful
            if manual_cell_num != -1 and manual_cell_num != llm_err_cell_num:
                print_msg(f"⚠️ Cell number mismatch: Manual={manual_cell_num}, LLM={llm_err_cell_num}, using manual", 3)
                final_cell_num = manual_cell_num
                
            if manual_err_type and manual_err_type != final_err_type:
                print_msg(f"⚠️ Error type mismatch: Manual={manual_err_type}, LLM={final_err_type}, using manual", 3)
                final_err_type = manual_err_type
            
            print_msg(f"✅ LLM extraction completed: {final_err_type} in cell {final_cell_num}", 3)
            return final_cell_num, final_err_type
            
        except Exception as e:
            print_msg(f"❌ LLM extraction failed: {e}", 3)
            # Return best available manual results
            final_cell_num = manual_cell_num if manual_cell_num != -1 else 0
            final_err_type = manual_err_type if manual_err_type else "UnknownError"
            return final_cell_num, final_err_type


    def _extract_error_type(self, err: str):
        """
        Manually extract error type from error string using multiple patterns.
        Returns the error type or None if not found.
        """
        # Only remove ANSI codes if they exist
        if "\x1b[" in err:
            err = re.sub(r'\x1b\[[0-9;]*m', '', err)

        # Common error patterns to try in order of specificity
        error_patterns = [
            # Specific error patterns first
            r'\b(ModuleNotFoundError)\b',
            r'\b(FileNotFoundError)\b', 
            r'\b(NameError)\b',
            r'\b(ImportError)\b',
            r'\b(AttributeError)\b',
            r'\b(TypeError)\b',
            r'\b(ValueError)\b',
            r'\b(KeyError)\b',
            r'\b(IndexError)\b',
            r'\b(RuntimeError)\b',
            r'\b(SyntaxError)\b',
            r'\b(IndentationError)\b',
            r'\b(TimeoutError)\b',
            r'\b(ConnectionError)\b',
            r'\b(PermissionError)\b',
            r'\b(OSError)\b',
            r'\b(IOError)\b',
            # Generic patterns last
            r'\b([A-Z][a-zA-Z]*Error)\b',
            r'\b([A-Z][a-zA-Z]*Exception)\b',
        ]
        
        for pattern in error_patterns:
            match = re.search(pattern, err)
            if match:
                error_type = match.group(1)
                print_msg(f"✅ Extracted error type manually: {error_type}", 3)
                return error_type
                
        # If no standard error found, try to extract from common error message formats
        # Look for patterns like "Exception: message" or "Error: message"
        fallback_patterns = [
            r'^([A-Z][a-zA-Z]*(?:Error|Exception)):',  # Start of line
            r'\n([A-Z][a-zA-Z]*(?:Error|Exception)):',  # After newline
            r'([A-Z][a-zA-Z]*(?:Error|Exception)): ',  # With colon and space
        ]
        
        for pattern in fallback_patterns:
            match = re.search(pattern, err)
            if match:
                error_type = match.group(1)
                print_msg(f"✅ Extracted error type (fallback): {error_type}", 3)
                return error_type
        
        print_msg(f"❌ Could not extract error type from: {err[:100]}...", 3)
        return None


    def _get_error_cell_index(self, err: str) -> int:
        """
        Extracts and returns the cell number from an error traceback string.
        If no cell number is found, returns -1.
        """

        # Common traceback patterns:
        # - In[5]
        # - In [5]
        # - Cell In[5], line ...
        # - File "<ipython-input-5-...>"
        
        # Try various patterns that may appear in tracebacks
        patterns = [
            r'In\[(\d+)\]',
            r'In \[(\d+)\]',
            r'Cell In\[(\d+)\]'
            r'<ipython-input-(\d+)-'
        ]
        
        for pat in patterns:
            match = re.search(pat, err)
            if match:
                cell_num = int(match.group(1))
                print_msg(f"✅ Extracted cell number manually: {cell_num}", 3)
                return cell_num

        return -1


    def execute_notebook(self):
        """
        Execute the notebook and return the result that include
        """
        try:
            papermill_execution(self.original_nb_path)
            return {
                'status': "executable", 
                'total_code_cells': self.total_code_cells,
                'err_cell_num': self.total_code_cells
            }
        except TimeoutError as e:
            return {
                'status': "TimeoutError", 
                'total_code_cells': self.total_code_cells,
                'err_cell_num': -1
            }
            # right now I cannot find the cell number where the TimeoutError occurred, so returning -1 fix it later 
            # neural network might be training, so we think if the notebook is taking longer than 5 minutes it can be 
            # executed
        except Exception as e:
            err_cell_num, err_type = self._get_error_cell_index_and_type_llm(str(e))

            # CASE 1: ModuleNotFoundError
            if ("ModuleNotFoundError" in str(e) and "No module named" in str(e)) or err_type == "ModuleNotFoundError":
                error_str = str(e)
                missing_module = None
                
                # Try to extract module name safely
                if "No module named" in error_str:
                    try:
                        # Handle both 'module' and "module" quote formats
                        parts = error_str.split("No module named ")
                        if len(parts) > 1:
                            module_part = parts[1]
                            # Remove quotes (both single and double)
                            missing_module = module_part.replace("'", "").replace('"', '').strip()
                            # Handle cases where there might be additional text after the module name
                            if '\n' in missing_module:
                                missing_module = missing_module.split('\n')[0]
                            # Get base module (before any dots)
                            missing_module = missing_module.split('.')[0]
                    except (IndexError, AttributeError) as parse_err:
                        print_msg(f"⚠️ Failed to parse module name from: {error_str[:100]}...", 3)
                        missing_module = "unknown"
                
                # Fallback if extraction failed
                if not missing_module:
                    missing_module = "unknown"
                
                return {
                    'status': "ModuleNotFoundError",
                    'total_code_cells': self.total_code_cells, 
                    'err_cell_num': err_cell_num,
                    'missing_module': missing_module
                }

            # CASE 2: FileNotFoundError
            elif "FileNotFoundError" in str(e) or "PATH_NOT_FOUND" in str(e) or err_type == "FileNotFoundError":
                extracted_path = None
                error_str = str(e)

                # Direct substring extraction
                if "No such file or directory: " in error_str:
                    extracted_path = error_str.split("No such file or directory: ")[1].replace("'", "").strip()
                else:
                    # List of regex patterns to try in order
                    patterns = [
                        r"FileNotFoundError: (.*?) not found\.",             # generic pattern
                        r"FileNotFoundError: File '(.*?)' does not exist",   # with quotes
                        r"FileNotFoundError: File (.*?) does not exist",     # without quotes
                        r"FileNotFoundError: The directory '(.*?)' does not exist",        # dir with quotes
                        r"FileNotFoundError: The directory (.*?) does not exist",          # dir no quotes
                        r"AnalysisException: \[PATH_NOT_FOUND\] Path does not exist: file:(.*?)\."
                    ]

                    for pattern in patterns:
                        match = re.search(pattern, error_str)
                        if match:
                            extracted_path = match.group(1)
                            if "\n" in extracted_path:
                                extracted_path = extracted_path.split("\n")[0]
                            break
        
                if extracted_path:
                    extracted_path = extracted_path.splitlines()[0]

                return {
                    'status': "FileNotFoundError",
                    'total_code_cells': self.total_code_cells, 
                    'err_cell_num': err_cell_num,
                    'FileNotFoundError_path': extracted_path
                }

            # CASE 3: NameError
            elif "NameError" in str(e) or err_type == "NameError":
                undefined_var = str(e).split("name '")[1].split("'")[0]
                return {
                    'status': "NameError", 
                    'total_code_cells': self.total_code_cells,
                    'err_cell_num': err_cell_num, 
                    'undefined_var': undefined_var
                }

            # CASE 4: Other Errors
            else:
                return {
                    'status': err_type, 
                    'total_code_cells': self.total_code_cells,
                    'err_cell_num': err_cell_num
                }