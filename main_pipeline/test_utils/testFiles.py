import nbformat
import os
from pathlib import Path
from main_pipeline.test_utils.fileOpsDetect import SafeFileIOMonitor
from main_pipeline.all_utils.print_format import print_msg

class TestFiles:
    def __init__(self, nb_path):
        self.nb_path = nb_path
        with open(nb_path, 'r', encoding='utf-8') as f:
            self.nb = nbformat.read(f, as_version=4)

    def _normalize_path(self, file):
        """Normalize a file path to an absolute path based on the notebook's directory.
        Handles relative paths and checks for permission issues.
        Args:
            file (str): The file path to normalize.

        Returns:
            str | None: The normalized absolute file path or None if permission denied.
        """
        try:
            file_norm = os.path.normpath(file)

            if os.path.isabs(file_norm):
                # Check if we can access this absolute path
                if not os.access(os.path.dirname(file_norm), os.R_OK):
                    return None  # Will be caught as permission denied
                return file_norm
            
            if self.nb_path:
                notebook_dir = os.path.dirname(self.nb_path)
                
                # Check if notebook directory is accessible
                if not os.access(notebook_dir, os.R_OK):
                    return None  # Will be caught as permission denied
                    
                abs_path = os.path.abspath(os.path.join(notebook_dir, file_norm))
                
                # Check if the resulting path's directory is accessible
                result_dir = os.path.dirname(abs_path)
                if not os.access(result_dir, os.R_OK):
                    return None  # Will be caught as permission denied
                    
                return abs_path
            
            final_path = os.path.abspath(file_norm)
            return final_path
        
        except PermissionError:
            return None  # Will be caught as permission denied
        except Exception:
            return None

    def _is_file_exist_in_filesystem(self, file):
        """
        Check if a file exists in the filesystem, handling permission issues.
        Args:
            file (str): The file path to check.
        Returns:
            bool: True if the file exists, False otherwise or if permission denied.
        """
        try:
            path_obj = Path(file)
            
            # Check if parent directory is readable first
            if not path_obj.parent.exists():
                return False
                
            if not os.access(path_obj.parent, os.R_OK):
                return False
                
            return path_obj.exists()
        except PermissionError:
            return False
        except Exception:
            return False
        

    def test_file_operations(self, result):
        """
        Test file operations for the given result.
        """
        existing_files = set()
        unfound_files = set()
        invalid_files = set()

        keys = list(result.keys())

        for i, id in enumerate(keys):
            if id == '_summary':
                continue

            log = result[id]
            operations = log.get('operations', [])

            for j, op_file in enumerate(operations):
                op = op_file.get('op')
                file = op_file.get('file')

                file_basename = os.path.basename(file) if file else ""
                if not file or\
                    ("<" in file_basename and ">" in file_basename) or \
                    "VirtualFile" in file or \
                    file_basename.isdigit():
                    continue

                try:
                    file_norm = self._normalize_path(file)
                    if file_norm is None:
                        continue

                    # Avoid checking env package files
                    if "site-packages" in file_norm or "dist-packages" in file_norm:
                        continue

                    # If normalization failed due to permissions, add to invalid files
                    if file_norm is None:
                        continue

                    if op in ['read', 'delete']:
                        found = False

                        # 1. Check if file exists (after write or create)
                        # 1.1. Within current cell
                        for k in range(j - 1, -1, -1):
                            prev_op = operations[k].get('op')
                            prev_file = operations[k].get('file')

                            if prev_file == file:
                                if prev_op in ['write', 'create']:
                                    if file_norm is not None:
                                        existing_files.add(file_norm)
                                    found = True
                                    break
                                elif prev_op == 'delete':
                                    if file_norm is not None:
                                        invalid_files.add(file_norm)
                                    found = True
                                    break
                        
                        if found:
                            continue

                        # 1.2. Within previous cells
                        for l in range(i - 1, -1, -1):
                            prev_cell_id = keys[l]
                            prev_ops = result[prev_cell_id].get('operations', [])

                            for m in range(len(prev_ops)-1, -1, -1):
                                prev_op_file = prev_ops[m]
                                prev_op = prev_op_file.get('op')
                                prev_file = prev_op_file.get('file')

                                if prev_file == file:
                                    if prev_op in ['write', 'create']:
                                        if file_norm is not None:
                                            existing_files.add(file_norm)
                                        found = True
                                        break
                                    elif prev_op == 'delete':
                                        if file_norm is not None:
                                            invalid_files.add(file_norm)
                                        found = True
                                        break
                            if found:
                                break

                        if found:
                            continue

                        # 2. Check if file exists in the filesystem
                        if self._is_file_exist_in_filesystem(file_norm):
                            if file_norm is not None:
                                existing_files.add(file_norm)
                            continue

                        try:
                            if self._is_file_exist_in_filesystem(file_norm):
                                if file_norm is not None:
                                    existing_files.add(file_norm)
                                continue
                            else:
                                # Check if it's a permission issue
                                if not os.access(os.path.dirname(file_norm), os.R_OK):
                                    if file_norm is not None:
                                        invalid_files.add(file_norm)
                                    continue
                        except PermissionError:
                            if file_norm is not None:
                                invalid_files.add(file_norm)
                            continue

                        if file_norm is not None:
                            unfound_files.add(file_norm)
                except PermissionError:
                    if file_norm is not None:
                        invalid_files.add(file_norm)
                    continue
                except Exception:
                    if file_norm is not None:
                        invalid_files.add(file_norm)
                    continue

        print_msg(f"- {len(existing_files)} valid path(s)", 5)
        print_msg(f"- {len(unfound_files)} missing path(s)", 5)
        print_msg(f"- {len(invalid_files)} invalid path(s)", 5)

        return existing_files, unfound_files, invalid_files

    def analyze(self):
        """
        Analyze the notebook for file operations.
        """
        monitor = SafeFileIOMonitor()
        results = monitor.analyze_cells(self.nb)
        found_files, unfound_files, invalid_files = self.test_file_operations(results)

        return {
            "found_files": found_files,
            "unfound_files": unfound_files,
            "invalid_files": invalid_files,
        }