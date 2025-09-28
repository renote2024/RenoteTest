
import traceback
import os
import io
import builtins
import shutil
import pandas as pd
import numpy as np
from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr, contextmanager
from IPython.core.interactiveshell import InteractiveShell
from main_pipeline.all_utils.print_format import print_msg


# =========================
# Logging helper
# =========================
class MonkeyHooks:
    def __init__(self):
        self.logged_operations = []

    def should_ignore_path(self, filepath):
        """
        Determine if a file path should be ignored from logging
        Returns True if the path should be ignored
        """
        if isinstance(filepath, VirtualFileData) or isinstance(filepath, io.StringIO):
            return False

        filepath_str = str(filepath)

        current_wd = os.getcwd()

        ignore_patterns = [
            # System directories and files
            '/tmp/',
            '/var/',
            '/sys/',
            '/proc/',
            '/dev/',
            'tmp/',

            # Environment-specific paths
            'site-packages/',
            'dist-packages/',
            '/home/chat/.',
            '/home/chat/miniconda3/',
            
            # Metadata / config files
            '.lock',
            'fontlist-',
            'tex.cache',
            'matplotlibrc',
            '/.matplotlib/',

            # Python cache / compiled
            '__pycache__',
            '.pyc',
            '.pyo',
            '.pyd',
            '.pyx',

            # Downloaded / cached data for common libs
            'nltk_data/',
            'scikit_learn_data/',
            'scikit-learn-data/',
            'torch/hub/',
            'transformers/cache/',
            '.cache/',

            # Internal tools
            'fileOpsDetect.py',
            'print_format.py',
            '.git/',

            # Working files
            '.ipynb_checkpoints/',
            current_wd
        ]

        for pattern in ignore_patterns:
            if pattern in filepath_str:
                # print("IGNORING pattern:", pattern, "in", filepath_str)
                return True

        if filepath_str.startswith('/') and not filepath_str.startswith('/home/chat/'):
            if not any(allowed in filepath_str for allowed in ['/data/', '/workspace/', '/project/']):
                # print("IGNORING outside /home/chat/:", filepath_str)
                return True

        return False

    def normalize_path(self, path):
        if isinstance(path, VirtualFileData):
            return getattr(path, "path", str(path))  # use path attribute if available
        elif isinstance(path, (str, Path)):
            return str(Path(path).expanduser().resolve())
        return str(path)  # fallback

    def log_file_op(self, operation, filepath):
        if not self.should_ignore_path(filepath):
            new_op = {"op": operation, "file": filepath}
            if new_op not in self.logged_operations:
                self.logged_operations.append(new_op)



# =========================
# Virtual File primitives
# =========================
class VirtualFile:
    """
    A very small in-memory file object.
    - Text mode returns str from read()
    - Binary mode returns bytes from read()
    - Iteration yields lines (text mode only)
    """
    def __init__(self, content="", mode="r"):
        self.mode = mode
        self.closed = False
        # Internally keep both text and bytes representations available as needed
        if "w" in mode:
            self._text = "" if "b" not in mode else None
            self._bytes = b"" if "b" in mode else None
        else:
            if "b" in mode:
                if isinstance(content, str):
                    self._bytes = content.encode("utf-8")
                else:
                    self._bytes = content if isinstance(content, (bytes, bytearray)) else b""
                self._text = None
            else:
                if isinstance(content, bytes):
                    try:
                        self._text = content.decode("utf-8", errors="replace")
                    except Exception:
                        self._text = ""
                else:
                    self._text = content if isinstance(content, str) else ""
                self._bytes = None
        self._pos = 0

    # ---------- helpers ----------
    def _as_text(self):
        if self._text is not None:
            return self._text
        # convert bytes to text for read in text mode
        self._text = self._bytes.decode("utf-8", errors="replace") if self._bytes is not None else ""
        return self._text

    def _as_bytes(self):
        if self._bytes is not None:
            return self._bytes
        # convert text to bytes for read in binary mode
        self._bytes = (self._text or "").encode("utf-8")
        return self._bytes

    # ---------- core i/o ----------
    def read(self, size=-1):
        if self.closed:
            raise ValueError("I/O operation on closed file")

        if "b" in self.mode:
            data = self._as_bytes()
            if size is None or size < 0:
                out = data[self._pos:]
                self._pos = len(data)
                return out
            out = data[self._pos:self._pos + size]
            self._pos += len(out)
            return out
        else:
            data = self._as_text()
            if size is None or size < 0:
                out = data[self._pos:]
                self._pos = len(data)
                return out
            out = data[self._pos:self._pos + size]
            self._pos += len(out)
            return out

    def write(self, data):
        if self.closed:
            raise ValueError("I/O operation on closed file")

        if "b" in self.mode:
            b = data if isinstance(data, (bytes, bytearray)) else str(data).encode("utf-8")
            buf = bytearray(self._as_bytes())
            # emulate append vs write (we keep it simple: always append)
            buf[self._pos:self._pos] = b
            self._bytes = bytes(buf)
            self._pos += len(b)
            return len(b)
        else:
            s = data if isinstance(data, str) else str(data)
            txt = self._as_text()
            # emulate append (simpler): insert at current position
            self._text = txt[:self._pos] + s + txt[self._pos:]
            self._pos += len(s)
            return len(s)

    def seek(self, pos, whence=0):
        base = 0
        if whence == 0:  # from start
            base = 0
        elif whence == 1:  # from current
            base = self._pos
        elif whence == 2:  # from end
            if "b" in self.mode:
                base = len(self._as_bytes())
            else:
                base = len(self._as_text())
        self._pos = max(0, base + pos)
        return self._pos

    def tell(self):
        return self._pos

    def flush(self):
        pass

    def __iter__(self):
        if "b" in self.mode:
            self._iter_lines = self._as_bytes().splitlines(keepends=True)
        else:
            self._iter_lines = self._as_text().splitlines(keepends=True)
        self._iter_idx = 0
        return self

    def __next__(self):
        if "b" in self.mode:
            raise TypeError("iteration over binary mode virtual file is not supported")
        if self._iter_idx >= len(self._iter_lines):
            raise StopIteration
        line = self._iter_lines[self._iter_idx]
        self._iter_idx += 1
        return line

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()


class VirtualFileData:
    """Metadata wrapper for the Virtual FS."""
    def __init__(self, content='', file_type=None, metadata=None):
        self.content = content  # text (str) or a textual description
        self.file_type = file_type  # 'npy','npz','csv','json','pickle','dir', etc.
        self.metadata = metadata or {}

    def __str__(self):
        return self.content


class DataMockFactory:
    """Create mock objects based on file types for fallback loads."""
    @staticmethod
    def create_mock(vfd: VirtualFileData):
        if not vfd:
            return None
        ft = vfd.file_type
        md = vfd.metadata

        if ft == 'npy':
            shape = md.get('shape', (3,))
            return np.zeros(shape)

        if ft == 'npz':
            class MockNpzFile(dict):
                def __init__(self, arrays_metadata):
                    super().__init__()
                    for name, meta in arrays_metadata.items():
                        shape = meta.get('shape', (3,))
                        self[name] = np.zeros(shape)
                def close(self): pass
                def __enter__(self): return self
                def __exit__(self, exc_type, exc_val, exc_tb): self.close()
            return MockNpzFile(md.get('arrays', {}))

        if ft == 'csv':
            cols = md.get('columns', ['col1', 'col2'])
            rows = md.get('rows', 0)
            return pd.DataFrame({c: np.zeros(rows) for c in cols})

        if ft == 'pickle':
            return md.get('object', None)

        if ft == 'json':
            return md.get('object', {})

        return vfd.content


class EnhancedVirtualFS:
    def __init__(self):
        self._files = {}  # path -> VirtualFileData
        self.storage = self._files  # alias for external access

    def __contains__(self, filepath):
        return str(filepath) in self._files

    def get(self, filepath, default=None):
        return self._files.get(str(filepath), default)

    def get_mock_object(self, filepath):
        vfd = self.get(filepath)
        return DataMockFactory.create_mock(vfd) if vfd else None

    def put(self, filepath, content, file_type=None, metadata=None):
        self._files[str(filepath)] = VirtualFileData(content, file_type, metadata)

    def update(self, mapping):
        for fp, content in mapping.items():
            if isinstance(content, VirtualFileData):
                self._files[str(fp)] = content
            else:
                self._files[str(fp)] = VirtualFileData(content)

    def pop(self, filepath, default=None):
        return self._files.pop(str(filepath), default)

    def keys(self):
        return self._files.keys()

    def items(self):
        return self._files.items()

    def __iter__(self):
        return iter(self._files)


# =========================
# Main Monitor
# =========================
class SafeFileIOMonitor:
    def __init__(self):
        self.file_hooks = None
        self.cell_logs = {}
        self.shared_globals = {}
        self.shared_locals = {}
        self.virtual_fs = EnhancedVirtualFS()
        self._original = {}

    # ---------- originals ----------
    def _save_originals(self):
        self._original = {
            'open': builtins.open,
            # os
            'os_close': os.close,
            'os_remove': os.remove,
            'os_unlink': os.unlink,
            'os_rmdir': os.rmdir,
            'os_makedirs': os.makedirs,
            'os_mkdir': os.mkdir,
            'os_rename': os.rename,
            'os_replace': os.replace,
            'os_listdir': os.listdir,
            'os_path_exists': os.path.exists,
            'os_path_isfile': os.path.isfile,
            'os_path_isdir': os.path.isdir,
            # shutil
            'shutil_copy': shutil.copy,
            'shutil_copy2': shutil.copy2,
            'shutil_copytree': shutil.copytree,
            'shutil_move': shutil.move,
            'shutil_rmtree': shutil.rmtree,
            # pandas
            'pd_read_csv': pd.read_csv,
            'pd_to_csv': pd.DataFrame.to_csv,
            'pd_read_excel': pd.read_excel,
            # numpy
            'np_load': np.load,
            'np_save': np.save,
            'np_savez': np.savez,
            # pathlib
            'path_open': getattr(Path, 'open', None),
            'path_read_text': getattr(Path, 'read_text', None),
            'path_write_text': getattr(Path, 'write_text', None),
            'path_read_bytes': getattr(Path, 'read_bytes', None),
            'path_write_bytes': getattr(Path, 'write_bytes', None)
        }

    # ---------- virtual helpers ----------
    def _update_virtual_fs(self, filepath, content, file_type=None, metadata=None):
        self.virtual_fs.put(filepath, content, file_type, metadata)

    def _copy_virtual_file(self, src, dst):
        s, d = str(src), str(dst)
        if s in self.virtual_fs.keys():
            self.virtual_fs.update({d: self.virtual_fs.get(s)})
        else:
            self.virtual_fs.put(d, "", file_type='copy')
        return None

    def _rename_virtual_file(self, src, dst):
        s, d = str(src), str(dst)
        if s in self.virtual_fs.keys():
            v = self.virtual_fs.get(s)
            self.virtual_fs.update({d: v})
            self.virtual_fs.pop(s, None)
        else:
            self.virtual_fs.put(d, "", file_type='moved')
        return None

    # ---------- core patchers ----------
    def _virtual_open(self, file, mode='r', *args, **kwargs):
        if "r" in mode and all(m not in mode for m in ["w", "a", "+"]):
            op = "read"
        else:
            op = "write"
        self.file_hooks.log_file_op(op, file)

        fp = str(file)

        # Try real filesystem first; swallow any exception and fall back
        try:
            return self._original['open'](file, mode, *args, **kwargs)
        except Exception:
            pass

        # Fall back to virtual file
        vfd = self.virtual_fs.get(fp)
        if vfd is None:
            content = ""
        else:
            if isinstance(vfd, dict):
                # vfd could be {"content": "...", "mode": "r"} or similar
                content = vfd.get("content", "")
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                elif not isinstance(content, str):
                    content = str(content)
            elif isinstance(vfd, bytes):
                content = vfd.decode("utf-8", errors="replace")
            else:
                # For other types, make sure str() returns a string
                try:
                    content = str(vfd)
                except Exception:
                    content = repr(vfd)  # fallback representation

        if "b" in mode:
            content = content.encode("utf-8")

        vfile = VirtualFile(content, mode)

        # On write/append/update, persist content to VirtualFS at close
        if any(m in mode for m in ['w', 'a', '+']):
            def update_on_close(f):
                old_close = f.close
                def new_close():
                    # snapshot content (always as text)
                    if "b" in f.mode:
                        text = f._as_bytes().decode("utf-8", errors="replace")
                    else:
                        text = f._as_text()
                    self._update_virtual_fs(fp, text)
                    old_close()
                f.close = new_close
                return f
            vfile = update_on_close(vfile)

        return vfile

    # ---------------- OS ----------------
    def _patch_os(self):
        def _virtual_mkdir(path, *a, **kw):
            self.file_hooks.log_file_op("write", path)
            self.virtual_fs.put(path, content='', file_type='dir')
            try:
                return self._original['os_mkdir'](path, *a, **kw)
            except Exception:
                return None

        def _virtual_makedirs(path, *a, **kw):
            self.file_hooks.log_file_op("write", path)
            self.virtual_fs.put(path, content='', file_type='dir')
            try:
                return self._original['os_makedirs'](path, *a, **kw)
            except Exception:
                return None

        def _virtual_remove(path):
            self.file_hooks.log_file_op("delete", path)
            self.virtual_fs.pop(str(path), None)
            try:
                return self._original['os_remove'](path)
            except Exception:
                return None

        def _virtual_rmdir(path):
            self.file_hooks.log_file_op("delete", path)
            self.virtual_fs.pop(str(path), None)
            try:
                return self._original['os_rmdir'](path)
            except Exception:
                return None

        def _virtual_rename(src, dst):
            self.file_hooks.log_file_op("read", src)
            self.file_hooks.log_file_op("write", dst)
            self._rename_virtual_file(src, dst)
            try:
                return self._original['os_rename'](src, dst)
            except Exception:
                return None

        def _virtual_listdir(path):
            self.file_hooks.log_file_op("read", path)
            p = str(path)
            
            try:
                return self._original['os_listdir'](path)
            except Exception:
                prefix = p.rstrip("/") + "/"
                seen, out = set(), []
                for f in self.virtual_fs._files:
                    if f.startswith(prefix):
                        rest = str(f)[len(prefix):].split("/", 1)[0]
                        if rest and rest not in seen:
                            seen.add(rest)
                            out.append(rest)
                return out

        def _virtual_isfile(path):
            p = str(path)
            self.file_hooks.log_file_op("read", path)
            try:
                if self._original['os_path_isfile'](path):
                    return True
            except Exception:
                pass
            return p in self.virtual_fs._files

        def _virtual_isdir(path):
            p = str(path)
            self.file_hooks.log_file_op("read", path)
            try:
                if self._original['os_path_isdir'](path):
                    return True
            except Exception:
                pass
            prefix = p.rstrip("/") + "/"
            return any(str(f).startswith(prefix) for f in self.virtual_fs._files)

        def _virtual_exists(path):
            p = str(path)
            self.file_hooks.log_file_op("read", path)
            try:
                if self._original['os_path_exists'](path):
                    return True
            except Exception:
                pass
            if p in self.virtual_fs._files:
                return True
            prefix = p.rstrip("/") + "/"
            return any(str(f).startswith(prefix) for f in self.virtual_fs._files)

        # apply
        os.remove = _virtual_remove
        os.unlink = _virtual_remove
        os.rmdir = _virtual_rmdir
        os.mkdir = _virtual_mkdir
        os.makedirs = _virtual_makedirs
        os.rename = _virtual_rename
        os.replace = _virtual_rename
        os.listdir = _virtual_listdir
        os.path.isfile = _virtual_isfile
        os.path.isdir = _virtual_isdir
        os.path.exists = _virtual_exists

    # -------------- SHUTIL --------------
    def _patch_shutil(self):
        def _virtual_copy(src, dst, *a, **kw):
            self.file_hooks.log_file_op("read", src)
            self.file_hooks.log_file_op("write", dst)
            try:
                return self._original['shutil_copy'](src, dst, *a, **kw)
            except Exception:
                return self._copy_virtual_file(src, dst)

        def _virtual_copy2(src, dst, *a, **kw):
            self.file_hooks.log_file_op("read", src)
            self.file_hooks.log_file_op("write", dst)
            try:
                return self._original['shutil_copy2'](src, dst, *a, **kw)
            except Exception:
                return self._copy_virtual_file(src, dst)

        def _virtual_move(src, dst, *a, **kw):
            self.file_hooks.log_file_op("read", src)
            self.file_hooks.log_file_op("write", dst)
            try:
                return self._original['shutil_move'](src, dst, *a, **kw)
            except Exception:
                return self._rename_virtual_file(src, dst)

        def _virtual_rmtree(path, *a, **kw):
            self.file_hooks.log_file_op("delete", path)
            try:
                return self._original['shutil_rmtree'](path, *a, **kw)
            except Exception:
                prefix = str(path).rstrip("/") + "/"
                for k in list(self.virtual_fs.keys()):
                    if str(k).startswith(prefix):
                        self.virtual_fs.pop(k, None)
                self.virtual_fs.pop(str(path), None)
                return None

        def _virtual_copytree(src, dst, *a, **kw):
            s, d = str(src), str(dst)
            self.file_hooks.log_file_op("read", src)
            self.file_hooks.log_file_op("write", dst)
            try:
                return self._original['shutil_copytree'](src, dst, *a, **kw)
            except Exception:
                for f in [p for p in self.virtual_fs.keys() if str(p).startswith(s.rstrip("/") + "/")]:
                    rel = str(f)[len(s.rstrip("/")) + 1:]
                    dst_file = os.path.join(d, rel)
                    self.file_hooks.log_file_op("write", dst_file)
                    self.virtual_fs.put(dst_file, self.virtual_fs.get(f))
                return d

        shutil.copy = _virtual_copy
        shutil.copy2 = _virtual_copy2
        shutil.move = _virtual_move
        shutil.rmtree = _virtual_rmtree
        shutil.copytree = _virtual_copytree

    # -------------- PANDAS --------------
    def _patch_pandas(self):
        def _virtual_read_excel(io, *a, **kw):
            if isinstance(io, (str, Path)):
                self.file_hooks.log_file_op("read", io)
            try:
                return self._original['pd_read_excel'](io, *a, **kw)
            except Exception:
                return pd.DataFrame()

        def _virtual_read_csv(path, *a, **kw):
            self.file_hooks.log_file_op("read", path)
            # print("PATHHHHHH", path)

            file_data = self.virtual_fs.get(path)

            if file_data is not None:
                # Handle dict {"content": "..."}
                if isinstance(file_data, dict) and "content" in file_data:
                    csv_content = file_data["content"]
                # Handle raw string
                elif isinstance(file_data, str):
                    csv_content = str(file_data)
                else:
                    csv_content = None

                if csv_content and csv_content.strip():
                    self.file_hooks.log_file_op("read", path)
                    return self._original["pd_read_csv"](StringIO(csv_content), *a, **kw)
                # If content is empty → fall back to real FS
                self.file_hooks.log_file_op("read", path)

            # Fallback: real filesystem
            return self._original["pd_read_csv"](path, *a, **kw)

        def _virtual_to_csv(df_self, filepath_or_buf=None, *a, **kw):
            # Case 1: writing to a buffer (not a path)
            if filepath_or_buf is None or hasattr(filepath_or_buf, "write"):
                return self._original['pd_to_csv'](df_self, filepath_or_buf, *a, **kw)

            # Case 2: writing to a real path
            p = str(filepath_or_buf)
            self.file_hooks.log_file_op("write", filepath_or_buf)

            # Capture CSV content into a string
            buf = StringIO()
            self._original['pd_to_csv'](df_self, buf, *a, **kw)
            buf.seek(0)
            csv_text = buf.read()

            # Always update the virtual FS
            self._update_virtual_fs(
                p,
                {"content": csv_text},
                file_type="csv",
                metadata={"columns": df_self.columns.tolist(), "rows": len(df_self)}
            )

            # Try writing to real FS, but don't fail if it doesn't exist
            try:
                with self._original['open'](p, 'w') as f:
                    f.write(csv_text)
            except Exception:
                pass

            return None


        pd.read_excel = _virtual_read_excel
        pd.read_csv = _virtual_read_csv
        pd.DataFrame.to_csv = _virtual_to_csv

    # -------------- NUMPY --------------
    def _patch_numpy(self):
        def _virtual_load(file, *a, **kw):
            p = str(file)
            self.file_hooks.log_file_op("read", file)
            try:
                return self._original['np_load'](file, *a, **kw)
            except Exception:
                mock = self.virtual_fs.get_mock_object(p)
                return mock if mock is not None else np.array([])

        def _virtual_save(file, arr, *a, **kw):
            p = str(file)
            self.file_hooks.log_file_op("write", file)
            self._update_virtual_fs(
                p, f"NumPy array with shape {getattr(arr, 'shape', None)}",
                file_type='npy', metadata={'shape': getattr(arr, 'shape', None)}
            )
            try:
                return self._original['np_save'](file, arr, *a, **kw)
            except Exception:
                return None

        def _virtual_savez(file, *a, **kw):
            p = str(file)
            self.file_hooks.log_file_op("write", file)

            meta = {}
            for i, v in enumerate(a):
                if isinstance(v, np.ndarray):
                    meta[f'arr_{i}'] = {'shape': v.shape}
            for k, v in kw.items():
                if isinstance(v, np.ndarray):
                    meta[k] = {'shape': v.shape}

            self._update_virtual_fs(
                p,
                "NumPy compressed arrays: " + ', '.join(f"{k}:{v['shape']}" for k, v in meta.items()),
                file_type='npz',
                metadata={'arrays': meta}
            )
            try:
                return self._original['np_savez'](file, *a, **kw)
            except Exception:
                return None

        np.load = _virtual_load
        np.save = _virtual_save
        np.savez = _virtual_savez

    # ------------- PATHLIB -------------
    def _patch_pathlib(self):
        def _virtual_path_open(p_self, mode='r', *a, **kw):
            return self._virtual_open(p_self, mode, *a, **kw)

        def _virtual_read_text(p_self, *a, **kw):
            p = str(p_self)
            self.file_hooks.log_file_op("read", p_self)
            try:
                return self._original['path_read_text'](p_self, *a, **kw)
            except Exception:
                vfd = self.virtual_fs.get(p)
                return str(vfd) if vfd else ""

        def _virtual_write_text(p_self, content, *a, **kw):
            p = str(p_self)
            self.file_hooks.log_file_op("write", p_self)
            self._update_virtual_fs(p, content)
            try:
                return self._original['path_write_text'](p_self, content, *a, **kw)
            except Exception:
                return len(content)

        def _virtual_read_bytes(p_self, *a, **kw):
            p = str(p_self)
            self.file_hooks.log_file_op("read", p_self)
            try:
                return self._original['path_read_bytes'](p_self, *a, **kw)
            except Exception:
                vfd = self.virtual_fs.get(p)
                return (str(vfd) if vfd else "").encode("utf-8")

        def _virtual_write_bytes(p_self, data, *a, **kw):
            p = str(p_self)
            self.file_hooks.log_file_op("write", p_self)
            try:
                text = data.decode("utf-8", errors="replace")
            except Exception:
                text = f"Binary data ({len(data)} bytes)"
            self._update_virtual_fs(p, text)
            try:
                return self._original['path_write_bytes'](p_self, data, *a, **kw)
            except Exception:
                return len(data)

        if self._original['path_open'] is not None:
            Path.open = _virtual_path_open
        if self._original['path_read_text'] is not None:
            Path.read_text = _virtual_read_text
        if self._original['path_write_text'] is not None:
            Path.write_text = _virtual_write_text
        if self._original['path_read_bytes'] is not None:
            Path.read_bytes = _virtual_read_bytes
        if self._original['path_write_bytes'] is not None:
            Path.write_bytes = _virtual_write_bytes

    # ------------- MASTER PATCH ----------
    def _patch_all(self):
        builtins.open = self._virtual_open
        io.open = self._virtual_open
        self._patch_os()
        self._patch_shutil()
        self._patch_pandas()
        self._patch_numpy()
        self._patch_pathlib()

    # ------------- RESTORE ---------------
    def _restore_all(self):
        if not self._original:
            return
        builtins.open = self._original['open']

        os.remove = self._original['os_remove']
        os.unlink = self._original['os_unlink']
        os.rmdir = self._original['os_rmdir']
        os.makedirs = self._original['os_makedirs']
        os.mkdir = self._original['os_mkdir']
        os.rename = self._original['os_rename']
        os.replace = self._original['os_replace']
        os.listdir = self._original['os_listdir']
        os.path.exists = self._original['os_path_exists']
        os.path.isfile = self._original['os_path_isfile']
        os.path.isdir = self._original['os_path_isdir']

        shutil.copy = self._original['shutil_copy']
        shutil.copy2 = self._original['shutil_copy2']
        shutil.copytree = self._original['shutil_copytree']
        shutil.move = self._original['shutil_move']
        shutil.rmtree = self._original['shutil_rmtree']

        pd.read_csv = self._original['pd_read_csv']
        pd.DataFrame.to_csv = self._original['pd_to_csv']
        pd.read_excel = self._original['pd_read_excel']

        np.load = self._original['np_load']
        np.save = self._original['np_save']
        np.savez = self._original['np_savez']

        if self._original['path_open'] is not None:
            Path.open = self._original['path_open']
        if self._original['path_read_text'] is not None:
            Path.read_text = self._original['path_read_text']
        if self._original['path_write_text'] is not None:
            Path.write_text = self._original['path_write_text']
        if self._original['path_read_bytes'] is not None:
            Path.read_bytes = self._original['path_read_bytes']
        if self._original['path_write_bytes'] is not None:
            Path.write_bytes = self._original['path_write_bytes']

    # ---------- public context ----------
    @contextmanager
    def file_operation_context(self):
        """Patch everything, run, and restore — always."""
        self.file_hooks = MonkeyHooks()
        self._save_originals()
        try:
            self._patch_all()
            yield
        finally:
            self._restore_all()

    # ---------- notebook execution ----------
    def analyze_cells(self, nb):
        """
        Execute notebook cells one by one in an IPython shell.
        Supports magics, cumulative variables, and shell commands.
        """
        code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
        print_msg(f"🔍 Analyzing {len(code_cells)} code cells", 4)

        with self.file_operation_context():
            shell = InteractiveShell()
            for i, cell in enumerate(code_cells, start=1):
                cell_id = f'cell{i}'
                self.file_hooks.logged_operations.clear()
                stdout_capture, stderr_capture = StringIO(), StringIO()

                if not cell.source.strip():
                    self.cell_logs[cell_id] = {'operations': [], 'status': 'empty'}
                    continue

                try:
                    print_msg(f"▶️ Executing Cell {i}", 5)
                    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                        shell.run_cell(cell.source)

                    # Convert file paths to strings here, after cell execution
                    operations_serializable = [
                        {"op": op["op"], "file": str(op["file"])} 
                        for op in self.file_hooks.logged_operations
                    ]

                    self.cell_logs[cell_id] = {
                        'operations': operations_serializable,
                        'status': 'success',
                        'stdout': stdout_capture.getvalue(),
                        'stderr': stderr_capture.getvalue()
                    }
                except Exception as e:
                    operations_serializable = [
                        {"op": op["op"], "file": str(op["file"])}
                        for op in self.file_hooks.logged_operations
                    ]

                    self.cell_logs[cell_id] = {
                        'operations': operations_serializable,
                        'status': 'error',
                        'error': f"{type(e).__name__}: {str(e)}",
                        'traceback': traceback.format_exc(),
                        'stdout': stdout_capture.getvalue(),
                        'stderr': stderr_capture.getvalue()
                    }

        # summary
        total_ops = sum(len(log.get('operations', [])) for log in self.cell_logs.values() if isinstance(log, dict))
        success_cells = sum(1 for log in self.cell_logs.values() if isinstance(log, dict) and log.get('status') == 'success')
        error_cells = sum(1 for log in self.cell_logs.values() if isinstance(log, dict) and log.get('status') == 'error')

        self.cell_logs['_summary'] = {
            'total_cells': len(code_cells),
            'success_cells': success_cells,
            'error_cells': error_cells,
            'total_operations': total_ops,
            'virtual_files_created': len(list(self.virtual_fs.keys()))
        }

        print_msg("🔍 Analysis complete", 4)
        return self.cell_logs

    # ---------- Virtual FS utilities ----------
    def get_virtual_file_content(self, file_path):
        vfd = self.virtual_fs.get(str(file_path))
        return str(vfd) if vfd else ""

    def get_virtual_file_list(self):
        return list(self.virtual_fs.keys())

    def dump_virtual_filesystem(self, max_content_length=100):
        lines = []
        for fp in sorted(self.virtual_fs.keys()):
            vfd = self.virtual_fs.get(fp)
            text = str(vfd)
            file_type = vfd.file_type if vfd else None
            type_info = f" [{file_type}]" if file_type else ""
            if len(text) > max_content_length:
                display = text[:max_content_length] + "..."
            else:
                display = text
            lines.append(f"FILE: {fp}{type_info}\n{display}\n")
        return "\n".join(lines)