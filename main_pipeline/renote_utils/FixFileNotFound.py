import os
from pathlib import Path
import shutil
from pydantic import BaseModel
from contextlib import contextmanager

from main_pipeline.renote_utils.localOllama import localchat
from main_pipeline.all_utils.print_format import print_msg

class File(BaseModel):
    content: str

class FixFileNotFound:
    def __init__(self, nb_path, exec_results):
        self.nb_path = Path(nb_path).resolve()
        self.missing_path = Path(exec_results['FileNotFoundError_path'])
        self.generated_path = None

    @contextmanager
    def within_notebook_directory(self, path: Path):
        """
        Temporarily change the working directory.
        Args:
            path (Path): The directory to switch to.
        Yields:
            None
        """
        original_directory = Path.cwd()
        try:
            os.chdir(path)
            yield
        finally:
            os.chdir(original_directory)

    def _generate_abs_path(self, path: Path) -> Path | None:
        """
        Return absolute path inside notebook directory.
        If path is relative, it's considered relative to the notebook's directory.
        Args:
            path (Path): The input path (absolute or relative).
        Returns:
            Path | None: The absolute path or None if error occurs.
        """
        try:
            with self.within_notebook_directory(self.nb_path.parent):
                abs_path = path.resolve()
                print_msg(f"▶️ Generating absolute path for {abs_path}", 4)
                return abs_path
        except Exception as e:
            print_msg(f"❌ Error generating absolute path for {path}: {e}", 4)
            return None

    def _write_file_atomic(self, file_path: Path, content: str) -> bool:
        """
        Write content to file atomically using a temporary file.
        Args:
            file_path (Path): The target file path.
            content (str): The content to write.
        Returns:
            bool: True if successful, False otherwise.
        """
        tmp_path = None
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Create a temporary file in the same directory
            tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")

            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Atomic move (on the same filesystem)
            shutil.move(str(tmp_path), str(file_path))
            return True

        except Exception as e:
            print_msg(f"❌ Failed to write file {file_path}: {e}", 4)
            return False

        finally:
            # Clean up tmp file if it still exists
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    def _get_notebook_source(self) -> str:
        """
        Extract source code from notebook.
        Returns:
            str: The concatenated source code from all code cells.
        """
        print_msg(f"▶️ Getting source code from {self.nb_path}", 4)
        import nbformat
        with open(self.nb_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        source_lines = []
        for i, cell in enumerate(nb.cells, 1):
            if cell.cell_type == 'code' and cell.source.strip():
                source_lines.append(f"# In[{i}]:\n")
                source_lines.append(cell.source)
                source_lines.append("")
        return "\n".join(source_lines)

    def create_missing_file(self) -> bool:
        """
        Create missing file or directory, generating synthetic data if needed.
        Returns:
            bool: True if creation was successful, False otherwise.
        """
        self.generated_path = self._generate_abs_path(self.missing_path)
        if not self.generated_path:
            return False

        # Heuristic: directory if no suffix or ends with slash
        is_dir = self.generated_path.suffix == "" or str(self.generated_path).endswith("/")

        if is_dir:
            try:
                self.generated_path.mkdir(parents=True, exist_ok=True)
                return True
            except Exception as e:
                print_msg(f"❌ Failed to create directory {self.generated_path}: {e}", 4)
                # Ensure nothing exists if creation failed
                if self.generated_path.exists():
                    try:
                        self.generated_path.rmdir()
                    except Exception:
                        pass
                return False

        # It's a file → generate synthetic content
        notebook_source = self._get_notebook_source()
        file_ext = self.generated_path.suffix.lstrip(".")
        content = ""
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            print_msg(f"🕑 Attempt {attempt}", 4)
            prompt = (
                f"Generate raw sample data for the input file `{self.generated_path.name}` "
                f"in `.{file_ext}` format for the notebook below. "
                f"Output only the data, no explanations or extra text.\n\n"
                f"{notebook_source}"
            )
            response = localchat(prompt, File)
            content = getattr(response, 'content', "")
            if content and content.strip():
                print_msg(f"✅ Content generated for {self.generated_path}", 5)
                content = content.strip()
                print_msg("▶️ Generated Content:", 5)
                print_msg("-" * 40, 5)
                print("\n".join("    " * 5 + line for line in content.splitlines()))
                print_msg("-" * 40, 5)
                break
            else:
                print_msg("⚠️ Empty content received, retrying...", 5)

        if not content:
            print_msg(f"‼️ No valid content generated for {self.generated_path}", 4)
            return False

        # Write file atomically
        if self._write_file_atomic(self.generated_path, content):
            print_msg(f"✅ File created for {self.generated_path}", 4)
            return True
        else:
            print_msg(f"❌ Failed to write content to {self.generated_path}", 4)
            return False
