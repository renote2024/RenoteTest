import importlib.util
import importlib.metadata
import os
import re
import sysconfig
from importlib.machinery import PathFinder
import nbformat

class TestModule:
    def __init__(self, nb_path, repo_path):
        self.nb_path = nb_path
        self.nb_dir = os.path.dirname(os.path.abspath(nb_path))
        self.repo_path = os.path.abspath(repo_path)
        paths = sysconfig.get_paths()
        self.stdlib_path = os.path.abspath(paths.get("stdlib", ""))
        self.site_packages_path = os.path.abspath(paths.get("purelib", ""))
        with open(nb_path, "r", encoding="utf-8") as f:
            self.nb = nbformat.read(f, as_version=4)

    def is_standard_module(self, module_name: str) -> bool:
        """Return True if module is part of the stdlib (not site-packages)."""
        try:
            spec = importlib.util.find_spec(module_name)
            if not spec:
                return False
            origin = getattr(spec, "origin", None)

            # built-in or frozen modules (e.g., 'sys', 'builtins')
            if origin in ("built-in", "frozen"):
                return True

            if not origin:
                return False

            origin = os.path.abspath(origin)
            # stdlib but exclude site-packages / purelib inside stdlib
            if origin.startswith(self.stdlib_path) and not origin.startswith(self.site_packages_path):
                return True

            # some stdlib compiled extensions may live in lib-dynload
            lib_dynload = os.path.join(self.stdlib_path, "lib-dynload")
            if origin.startswith(lib_dynload):
                return True

            return False
        except Exception:
            return False

    def is_local_module(self, module_name: str) -> bool:
        """
        Check if a module is a local module within the repository or notebook folder.
        Returns True if found locally, otherwise False.
        """
        try:
            base = module_name.split(".")[0]

            # 1) direct file/package check
            candidates = [
                os.path.join(os.path.dirname(self.nb_path), base + ".py"),
                os.path.join(os.path.dirname(self.nb_path), base, "__init__.py"),
                os.path.join(self.repo_path, base + ".py"),
                os.path.join(self.repo_path, base, "__init__.py"),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    return True

            # 2) directory exists (namespace package)
            for root in (os.path.dirname(self.nb_path), self.repo_path):
                pdir = os.path.join(root, base)
                if os.path.isdir(pdir):
                    return True

            # 3) use PathFinder with explicit search path
            search_paths = [os.path.dirname(self.nb_path), self.repo_path]
            spec = PathFinder.find_spec(base, search_paths)
            if spec and getattr(spec, "origin", None) not in (None, "built-in", "frozen"):
                return True
            if spec and getattr(spec, "submodule_search_locations", None):
                return True

            return False
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

            print("Checking:", mod)
            # print("  builtin?", self.is_builtin_module(mod))
            print("  stdlib?", self.is_standard_module(mod))
            print("  local?", self.is_local_module(mod))
            if self.is_standard_module(mod) or self.is_local_module(mod):
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
