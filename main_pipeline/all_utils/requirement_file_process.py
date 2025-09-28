import os
import re
import yaml
import toml
import chardet
import sys
# import builtins
from collections import defaultdict
from main_pipeline.all_utils.print_format import print_msg

# List of Conda-specific packages that won't work in a venv environment
CONDA_SPECIFIC_PACKAGES = [
    'mkl', 'blas', 'intel-openmp', 'vc', 'vs2015_runtime', 'icc_rt',
    '_libgcc_mutex', '_openmp_mutex', 'ca-certificates', 'certifi',
    'libgcc-ng', 'libstdcxx-ng', 'ld_impl_linux-64', 'libffi',
    'ncurses', 'openssl', 'pip', 'python', 'readline', 'setuptools',
    'sqlite', 'tk', 'wheel', 'xz', 'zlib']

# Get Python built-in modules dynamically
def get_builtin_modules():
    """Get a set of Python built-in and standard library modules."""
    builtin_modules = set(sys.builtin_module_names)
    
    # Add common standard library modules that might not be in builtin_module_names
    stdlib_modules = {
        'os', 'sys', 'json', 'math', 'random', 'time', 'datetime', 'collections',
        'itertools', 'functools', 'operator', 're', 'string', 'io', 'pathlib',
        'urllib', 'http', 'html', 'xml', 'email', 'base64', 'binascii', 'struct',
        'pickle', 'csv', 'configparser', 'logging', 'unittest', 'doctest',
        'argparse', 'getopt', 'shlex', 'subprocess', 'threading', 'multiprocessing',
        'queue', 'socket', 'ssl', 'asyncio', 'concurrent', 'ctypes', 'mmap',
        'select', 'signal', 'errno', 'stat', 'glob', 'fnmatch', 'tempfile',
        'shutil', 'gzip', 'bz2', 'lzma', 'zipfile', 'tarfile', 'hashlib',
        'hmac', 'secrets', 'uuid', 'copy', 'pprint', 'textwrap', 'unicodedata',
        'stringprep', 'readline', 'rlcompleter', 'bisect', 'array', 'weakref',
        'types', 'copyreg', 'pickle', 'shelve', 'marshal', 'dbm', 'sqlite3',
        'zoneinfo', 'calendar', 'locale', 'gettext', 'codecs', 'encodings'
    }
    
    return builtin_modules.union(stdlib_modules)

# Cache the builtin modules set
BUILTIN_MODULES = get_builtin_modules()

# Common invalid patterns that shouldn't be treated as packages
INVALID_PATTERNS = [
    r'^\*\*.*\*\*',  # Markdown bold text
    r'^#',           # Comments
    r'^-',           # Markdown lists
    r'^\d+\.',       # Numbered lists
    r'github\.com',  # GitHub URLs
    r'http[s]?://',  # URLs
    r'\.git$',       # Git repositories
    r'\.md$',        # Markdown files
    r'\.txt$',       # Text files
    r'\.py$',        # Python files
    r'^[A-Z][A-Z_]+$',  # All caps (likely constants)
    r'\s+',          # Contains whitespace (invalid package names)
]

def is_conda_specific_package(package_line):
    """Check if a package is Conda-specific and should be ignored."""
    return any(package_line.startswith(pkg) for pkg in CONDA_SPECIFIC_PACKAGES)

def is_valid_package_name(package_line):
    """Check if a package line represents a valid installable package."""
    if not package_line or not package_line.strip():
        return False
    
    # Extract just the package name (before any version specifiers)
    package_name = extract_package_name(package_line).strip()
    
    # Check against built-in and standard library modules
    if package_name.lower() in BUILTIN_MODULES:
        return False
    
    # Check against invalid patterns
    for pattern in INVALID_PATTERNS:
        if re.search(pattern, package_line):
            return False
    
    # Must start with letter or underscore, contain only valid chars
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_\-\.]*$', package_name):
        return False
    
    # Skip conda-specific packages
    if is_conda_specific_package(package_name):
        return False
    
    # Skip very short names (likely invalid)
    if len(package_name) < 2:
        return False
    
    # Skip if it looks like a file path
    if '/' in package_name or '\\' in package_name:
        return False
    
    return True


def convert_conda_to_venv_line(package_line):
    """Convert Conda format to venv/pip format."""
    # Skip empty lines and comments
    if not package_line.strip() or package_line.strip().startswith('#'):
        return None
    
    # Handle conda build strings - skip any line containing build strings
    if re.search(r'py\d+h[a-f0-9]+_\d+', package_line):
        return None
    
    # Handle conda-specific format like "_libgcc_mutex==0.1=conda_forge"
    if '=conda_forge' in package_line or '=main' in package_line or '=defaults' in package_line:
        # Extract package name and version, ignore channel info
        parts = package_line.split('=')
        if len(parts) >= 3:
            package = parts[0]
            version = parts[1] if parts[1] else parts[2]
            # Skip if it's a conda-specific package
            if is_conda_specific_package(package):
                return None
            return f"{package}=={version}"
    
    # Handle standard conda format with single =
    match = re.match(r'^([a-zA-Z0-9_\-\.]+)=([0-9\.]+)', package_line)
    if match:
        package, version = match.groups()
        # Skip if it's a conda-specific package
        if is_conda_specific_package(package):
            return None
        return f"{package}=={version}"
    else:
        # Check if it's just a package name without version
        if not package_line.startswith("#") and not is_conda_specific_package(package_line):
            return package_line
    return None


def is_conda_env_file(file_path):
    """Check if the file is a Conda environment file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
        for line in infile:
            line = line.strip()
            if line.startswith("# platform:") or ("=" in line and not "==" in line):
                return True
    return False
def extract_conda_packages(conda_file):
    """Convert a Conda requirements.txt file to a venv-compatible requirements.txt."""
    venv_lines = []
    with open(conda_file, "rb") as rawdata:
        result = chardet.detect(rawdata.read(10000))
        encoding = result["encoding"]
    # Use detected encoding
    with open(conda_file, "r", encoding=encoding) as infile:
        for line in infile:
            line = line.strip()
            if line and not is_conda_specific_package(line):
                venv_line = convert_conda_to_venv_line(line)
                if venv_line:
                    venv_lines.append(venv_line)
    return venv_lines


def extract_yaml_packages(yaml_file):
    """Extract pip-compatible package list from a Conda environment YAML file, handling >=, <=, >, < operators."""
    with open(yaml_file, 'r') as infile:
        env_data = yaml.safe_load(infile)
    packages = []
    operators = ['>=', '<=', '>', '<', '=']
    if 'dependencies' in env_data:
        for dep in env_data['dependencies']:
            if isinstance(dep, str):
                # Skip conda-specific packages early
                if is_conda_specific_package(dep):
                    continue
                
                # Handle conda build strings - skip them entirely
                if re.search(r'py\d+h[a-f0-9]+_\d+', dep):
                    continue
                
                # Check if the dep string contains any of the operators except '=' (to avoid splitting on '=') 
                # We want to split only on '=' if it's a single '=' without < or >
                matched_op = None
                for op in operators:
                    if op in dep and op != '=':
                        matched_op = op
                        break
                
                if matched_op:
                    # Leave it as is, just strip whitespace
                    packages.append(dep.strip())
                else:
                    # No >=, <=, >, <, so split on single '=' if present
                    if '=' in dep:
                        parts = dep.split('=')
                        if len(parts) >= 2:
                            pkg = parts[0].strip()
                            ver = parts[1].strip()
                            # Skip if package name is conda-specific
                            if not is_conda_specific_package(pkg):
                                packages.append(f"{pkg}=={ver}")
                    else:
                        packages.append(dep.strip())
            elif isinstance(dep, dict) and 'pip' in dep:
                packages.extend([p.strip() for p in dep['pip']])
    # Remove duplicates while preserving order
    seen = set()
    unique_packages = []
    for p in packages:
        if p not in seen and p:  # Also check p is not empty
            unique_packages.append(p)
            seen.add(p)
    return unique_packages


def poetry_to_pip_versions(poetry_list):
    pip_list = []
    for entry in poetry_list:
        if '^' in entry:
            pkg, ver = entry.split('^')
            parts = ver.split('.')
            major = int(parts[0])
            upper_major = major + 1
            upper = f"<{upper_major}.0"
            pip_list.append(f"{pkg}>={ver},{upper}")
        elif '~' in entry:
            pkg, ver = entry.split('~')
            parts = ver.split('.')
            if len(parts) >= 2:
                major, minor = int(parts[0]), int(parts[1])
                upper_minor = minor + 1
                upper = f"<{major}.{upper_minor}"
                pip_list.append(f"{pkg}>={ver},{upper}")
            else:
                pip_list.append(f"{pkg}>={ver}")
        elif '*' in entry:
            pip_list.append(entry.replace('*', ''))  # just package name
        else:
            pip_list.append(entry)
    return pip_list


def extract_toml_packages(toml_path):
    with open(toml_path, "r", encoding="utf-8") as f:
        data = toml.load(f)
    packages = []
    # Poetry-style pyproject.toml
    if "tool" in data and "poetry" in data["tool"]:
        poetry = data["tool"]["poetry"]
        deps = poetry.get("dependencies", {})
        dev_deps = poetry.get("dev-dependencies", {})
        # Remove python version
        deps.pop("python", None)
        def format_pkg(pkg_dict):
            result = []
            for pkg, v in pkg_dict.items():
                if isinstance(v, str):
                    result.append(f"{pkg}{v}")
                elif isinstance(v, dict) and "version" in v:
                    result.append(f"{pkg}{v['version']}")
                else:
                    result.append(pkg)
            return result
        packages += format_pkg(deps)
        packages += format_pkg(dev_deps)
    # PEP 621-style pyproject.toml
    elif "project" in data:
        project = data["project"]
        packages += project.get("dependencies", [])
        optional_deps = project.get("optional-dependencies", {})
        for group in optional_deps.values():
            packages += group
    packages = poetry_to_pip_versions(packages)
    return sorted(set(packages))


def extract_packages_from_file(file_path):
    """Extract packages from various file formats."""
    ext = os.path.splitext(file_path)[1].lower()
    packages = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if ext in ['.txt', '.in', '.ci', '.tx']:
                    # Only add if it's a valid package name
                    if is_valid_package_name(line):
                        packages.add(line)
                elif ext == '.sh':
                    if line.startswith('pip install'):
                        pip_packages = line.split('pip install ')[1].strip().split()
                        for pkg in pip_packages:
                            if is_valid_package_name(pkg):
                                packages.add(pkg)
                elif ext == '.md':
                    if line.startswith('- ') or line.startswith('```'):
                        package_line = line.strip().strip('-').strip('```').strip()
                        if package_line and is_valid_package_name(package_line):
                            packages.add(package_line)
                elif ext == '.py':
                    matches = re.findall(r'import (\w+)|from (\w+) import', line)
                    for group in matches:
                        for m in group:
                            if m and is_valid_package_name(m):
                                packages.add(m)
                elif ext == '.go' and 'import' in line:
                    matches = re.findall(r'import\s*\((.*?)\)', line, re.DOTALL)
                    for block in matches:
                        go_packages = re.findall(r'"(.*?)"', block)
                        for pkg in go_packages:
                            if is_valid_package_name(pkg):
                                packages.add(pkg)
    except Exception as e:
        print_msg(f"❌ Error reading file {file_path}: {str(e)}", 1)
        
    return list(packages)


def findRequirementsFile(repo_path):
    """Find the most likely requirements file in the given repository."""
    # Common requirement file name patterns
    common_names = [
        'requirements.txt',
        'environment.yml',
        'environment.yaml',
        'Pipfile',
        'pyproject.toml',
        'setup.py',
    ]
    # Patterns to look for in filenames
    substrings = [
        'requirement', 'requirements', 'req', 'reqs',
        'env', 'environment',
        'deps', 'dependencies',
        'setup', 'install'
    ]
    valid_extensions = {
        '.txt', '.yml', '.yaml', '.in', '.ci', '.tx', '.sh', '.md', '.py', '.go', '.toml'
    }
    # Normalize search: collect candidates
    candidates = []
    for dirpath, _, filenames in os.walk(repo_path):
        for fname in filenames:
            lower_fname = fname.lower()
            full_path = os.path.join(dirpath, fname)
            # Direct common names
            if lower_fname in common_names:
                return full_path
            # Substring + valid extension
            if any(substr in lower_fname for substr in substrings):
                _, ext = os.path.splitext(fname)
                if ext.lower() in valid_extensions:
                    candidates.append(full_path)
    # If no exact match, return first candidate (if any)
    if candidates:
        return sorted(candidates)[0]
    return None

def findAllRequirementsFiles(repo_path):
    """Find all requirements files in the given repository."""
    
    # Common requirement file name patterns
    common_names = [
        'requirements.txt',
        'environment.yml',
        'environment.yaml',
        'Pipfile',
        'pyproject.toml',
        'setup.py',
    ]
    # Patterns to look for in filenames
    substrings = [
        'requirement', 'requirements', 'req', 'reqs',
        'env', 'environment',
        'deps', 'dependencies',
        'setup', 'install'
    ]
    valid_extensions = {
        '.txt', '.yml', '.yaml', '.in', '.ci', '.tx', '.sh', '.md', '.py', '.go', '.toml'
    }
    # Collect all candidates
    candidates = []
    for dirpath, _, filenames in os.walk(repo_path):
        for fname in filenames:
            lower_fname = fname.lower()
            full_path = os.path.join(dirpath, fname)
            # Direct common names
            if lower_fname in common_names:
                candidates.append(full_path)
            # Substring + valid extension
            elif any(substr in lower_fname for substr in substrings):
                _, ext = os.path.splitext(fname)
                if ext.lower() in valid_extensions:
                    candidates.append(full_path)
    return sorted(candidates)

def extract_package_name(package_line):
    """Extract package name from a package specification line."""
    # First normalize the package specification
    package_line = normalize_package_spec(package_line)
    
    # Handle various version specifiers
    version_patterns = [
        r'^([a-zA-Z0-9_\-]+)==.*',  # ==
        r'^([a-zA-Z0-9_\-]+)>=.*',  # >=
        r'^([a-zA-Z0-9_\-]+)<=.*',  # <=
        r'^([a-zA-Z0-9_\-]+)>.*',   # >
        r'^([a-zA-Z0-9_\-]+)<.*',   # <
        r'^([a-zA-Z0-9_\-]+)=.*',   # =
        r'^([a-zA-Z0-9_\-]+)\^.*',  # ^ (poetry)
        r'^([a-zA-Z0-9_\-]+)\*.*',  # * (poetry)
        r'^([a-zA-Z0-9_\-]+)~.*',   # ~ (poetry/pip)
        r'^([a-zA-Z0-9_\-]+)!.*',   # ! (poetry)
        r'^([a-zA-Z0-9_\-]+)$'      # no version
    ]
    
    for pattern in version_patterns:
        match = re.match(pattern, package_line)
        if match:
            return match.group(1)
    
    return package_line

def normalize_package_spec(package_line):
    """Normalize package specification to ensure proper formatting."""
    package_line = package_line.strip()
    
    # Remove any extra whitespace
    package_line = re.sub(r'\s+', ' ', package_line)
    
    # Fix common formatting issues
    # Handle cases like "accelerate0.33.0" -> "accelerate==0.33.0"
    # Look for package names followed immediately by version numbers
    package_pattern = r'^([a-zA-Z][a-zA-Z0-9_\-]*?)(\d+\.\d+.*)$'
    match = re.match(package_pattern, package_line)
    if match:
        pkg_name, version = match.groups()
        return f"{pkg_name}=={version}"
    
    # Handle cases with missing spaces around operators
    # Fix: "package>=1.0" -> "package >= 1.0" -> "package>=1.0"
    operators = ['>=', '<=', '==', '!=', '>', '<', '=', '~', '^']
    for op in operators:
        if op in package_line:
            # Ensure proper spacing around operators
            package_line = re.sub(rf'(\S){re.escape(op)}(\S)', rf'\1{op}\2', package_line)
            break
    
    return package_line

def convert_to_pip_format(package_line):
    """Convert package specification to pip-compatible format."""
    # First normalize the package specification
    package_line = normalize_package_spec(package_line)
    
    # Handle tilde operator (~) - convert to pip format
    if '~' in package_line and not package_line.startswith('~'):
        pkg, ver = package_line.split('~', 1)
        parts = ver.split('.')
        if len(parts) >= 2:
            major, minor = int(parts[0]), int(parts[1])
            upper_minor = minor + 1
            upper = f"<{major}.{upper_minor}"
            return f"{pkg}>={ver},{upper}"
        else:
            return f"{pkg}>={ver}"
    
    # Handle caret operator (^) - convert to pip format
    elif '^' in package_line and not package_line.startswith('^'):
        pkg, ver = package_line.split('^', 1)
        parts = ver.split('.')
        major = int(parts[0])
        upper_major = major + 1
        upper = f"<{upper_major}.0"
        return f"{pkg}>={ver},{upper}"
    
    # Handle single equals (=) - convert to double equals
    elif '=' in package_line and not '==' in package_line and not '>=' in package_line and not '<=' in package_line:
        pkg, ver = package_line.split('=', 1)
        return f"{pkg}=={ver}"
    
    # Handle asterisk (*) - remove it
    elif '*' in package_line and not package_line.startswith('*'):
        return package_line.replace('*', '')
    
    # Already in pip format or no version specifier
    return package_line

def merge_package_versions(packages_list):
    """Merge packages from multiple sources, handling version conflicts."""
    package_versions = defaultdict(list)
    
    # Group packages by name and collect all versions
    for package in packages_list:
        if package and is_valid_package_name(package):
            # Convert to pip format first
            pip_package = convert_to_pip_format(package)
            pkg_name = extract_package_name(pip_package)
            
            # Double-check the extracted package name is valid
            if is_valid_package_name(pkg_name):
                package_versions[pkg_name].append(pip_package)
    
    # Resolve conflicts and create final package list
    final_packages = []
    for pkg_name, versions in package_versions.items():
        if len(versions) == 1:
            # Single version, use as is
            final_packages.append(versions[0])
        else:
            # Multiple versions, prefer more specific constraints
            sorted_versions = sorted(versions, key=lambda x: (
                '==' in x,  # Exact version first
                '>=' in x or '<=' in x,  # Range constraints
                '>' in x or '<' in x,    # Open constraints
                '~' in x,  # Tilde constraints
                '^' in x,  # Caret constraints
                len(x)  # Shorter (less specific) last
            ), reverse=True)
            
            # Use the most specific version
            final_packages.append(sorted_versions[0])
    
    return sorted(final_packages)


def convertRequirementFile(requirements_file):
    """Convert requirements file to venv format."""
    # Determine the file format and convert to venv format
    file_ext = os.path.splitext(requirements_file)[1].lower()
    output_file = os.path.join(os.path.dirname(requirements_file), 'requirements_venv.txt')
    print(f"    Found {file_ext} file: {requirements_file}")
    if file_ext == '.txt' and is_conda_env_file(requirements_file):
        packages = extract_conda_packages(requirements_file)
    elif file_ext in ['.yml', '.yaml']:
        # print("Converting YAML to txt format...")
        packages = extract_yaml_packages(requirements_file)
    elif file_ext == '.toml':
        # print("Converting TOML to txt format...")
        packages = extract_toml_packages(requirements_file)
    else:
        # print(f"Converting {file_ext} format to txt...")
        packages = extract_packages_from_file(requirements_file)
    # Write packages to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for package in packages:
            if package and is_valid_package_name(package):
                outfile.write(f"{package}\n")
    # print(f"Converted requirements saved to: {output_file}")
    return os.path.abspath(output_file)

def clean_empty_version(package_line):
    """
    If package_line has no version, remove any trailing operators like ==, >=, <=, ~, ^.
    Otherwise, leave it as is.
    """
    # Match name + optional operator + optional version
    match = re.match(r'^([a-zA-Z0-9_\-]+)([=><!~^]*)(.*)$', package_line.strip())
    if match:
        name, op, version = match.groups()
        if version.strip():  # version exists → keep original line
            return package_line.strip()
        else:  # no version → just package name
            return name
    return package_line.strip()


def convertAllRequirementsFiles(repo_path):
    """Find all requirements files and merge them into a single venv-compatible file."""
    requirements_files = findAllRequirementsFiles(repo_path)
    
    if not requirements_files:
        print_msg(f"❗ No requirements files found in {repo_path}", 1)
        return None

    print_msg(f"👀 Found {len(requirements_files)} requirements files:", 1)
    for file_path in requirements_files:
        print_msg(f"- {os.path.basename(file_path)}", 2)
    
    all_packages = []
    
    # Extract packages from each file
    for requirements_file in requirements_files:
        file_ext = os.path.splitext(requirements_file)[1].lower()
        
        try:
            if file_ext == '.txt' and is_conda_env_file(requirements_file):
                packages = extract_conda_packages(requirements_file)
            elif file_ext in ['.yml', '.yaml']:
                packages = extract_yaml_packages(requirements_file)
            elif file_ext == '.toml':
                packages = extract_toml_packages(requirements_file)
            else:
                packages = extract_packages_from_file(requirements_file)
            
            all_packages.extend(packages)
            print_msg(f"🔎 Extracted {len(packages)} packages from {os.path.basename(requirements_file)}", 1)

        except Exception as e:
            print_msg(f"❌ Error processing {os.path.basename(requirements_file)}: {str(e)}", 1)
            continue
    
    # Merge and deduplicate packages
    merged_packages = merge_package_versions(all_packages)
    
    # Write merged packages to output file
    output_file = os.path.join(repo_path, 'requirements_merged_venv.txt')
    # with open(output_file, 'w', encoding='utf-8') as outfile:
    #     for package in merged_packages:
    #         outfile.write(f"{package}\n")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for package in merged_packages:
            cleaned_pkg = clean_empty_version(package)
            outfile.write(f"{cleaned_pkg}\n")

    print_msg(f"🖇 Merged {len(merged_packages)} unique packages into: {output_file}", 1)
    return os.path.abspath(output_file)