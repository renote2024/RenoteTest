import importlib.util
import importlib.metadata
import sys
import os
import re
import sysconfig

class TestModule:
    def __init__(self, nb_path, repo_path):
        self.nb_path = nb_path
        self.repo_path = os.path.abspath(repo_path)
        std_paths = sysconfig.get_paths()
        self.stdlib_path = os.path.abspath(std_paths.get("stdlib", ""))
        with open(nb_path, "r", encoding="utf-8") as f:
            import nbformat
            self.nb = nbformat.read(f, as_version=4)

    def is_builtin_module(self, module_name: str) -> bool:
        """Check if a module is a built-in module."""
        return module_name in sys.builtin_module_names

    def is_standard_module(self, module_name: str) -> bool:
        """Check if a module is part of the standard library."""
        try:
            spec = importlib.util.find_spec(module_name)
            if not spec or not spec.origin:
                return False
            origin = os.path.abspath(spec.origin)
            return origin.startswith(self.stdlib_path)
        except Exception:
            return False

    def is_local_module(self, module_name: str) -> bool:
        """Check if a module is a local module within the repository."""
        try:
            spec = importlib.util.find_spec(module_name)
            if not spec or not spec.origin:
                return False
            origin = os.path.abspath(spec.origin)
            return origin.startswith(self.repo_path)
        except Exception:
            return False


    def analyze(self):
        """
        Analyze the notebook to extract module imports and their statuses.
        Returns:
            dict: A dictionary with module names as keys and their import status and version as values.
        Raises:
            Exception: If an error occurs during the analysis.
        """
        modules = set()
        import_pattern = re.compile(r"^\s*(?:import|from)\s+(\.?[\w\.]+)", re.MULTILINE)

        for cell in self.nb.cells:
            if cell.cell_type == 'code':
                code = cell.source
                matches = import_pattern.findall(code)
                for match in matches:
                    base_module = match.split('.')[0]
                    if base_module:
                        modules.add(base_module)

        result = {}
        for mod in modules:
            if self.is_builtin_module(mod) or self.is_standard_module(mod) or self.is_local_module(mod):
                continue

            KNOWN_STDLIB = set([
                "os", "sys", "re", "json", "math", "functools", "io", "time", "datetime",
                "random", "string", "hashlib", "hmac", "base64", "struct", "socket",
                "http", "urllib", "ssl", "email", "logging", "argparse", "configparser",
                "threading", "multiprocessing", "queue", "subprocess", "shutil", "tempfile",
                "itertools", "collections"
            ])
            if mod in KNOWN_STDLIB:
                continue

            mod_result = {}
            if importlib.util.find_spec(mod) is None:
                mod_result['status'] = 'failed'
                mod_result['version'] = None
            else:
                mod_result['status'] = 'success'
                try:
                    mod_result['version'] = importlib.metadata.version(mod)
                except importlib.metadata.PackageNotFoundError:
                    mod_result['version'] = 'unknown'
            
            result[mod] = mod_result

        return result