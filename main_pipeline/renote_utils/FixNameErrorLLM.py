import os
import uuid
import nbformat
from pydantic import BaseModel
import glob
import re
from main_pipeline.all_utils.print_format import print_msg
from main_pipeline.renote_utils.localOllama import localchat


class Name(BaseModel):
    code: str

class FixNameErrorLLM:
    def __init__(self, nb_path, undefined_var, defined_index, undefined_index):
        self.nb_path = nb_path
        self.undefined_var = undefined_var
        self.defined_index = defined_index
        self.undefined_index = undefined_index
        self.content = self._read_notebook()

    def _read_notebook(self):
        '''
        Read the notebook and return the notebook object
        Returns:
            nbformat.NotebookNode: The notebook object.
        '''
        with open(self.nb_path, 'r', encoding='utf-8') as f:
            return nbformat.read(f, as_version=4)

    def _get_notebook_source(self) -> str:
        """
        Extract source code from notebook.
        Args:
            nb_path (str): Path to the notebook file.
        Returns:
            str: Concatenated source code from all code cells.
        """
        print_msg(f"▶️ Getting source code from {self.nb_path}", 4)
        with open(self.nb_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        source_lines = []
        for i, cell in enumerate(nb.cells, 1):
            if cell.cell_type == 'code' and cell.source.strip():
                source_lines.append(f"# In[{i}]:\n")
                source_lines.append(cell.source)
                source_lines.append("")
        return "\n".join(source_lines)

    def _generate_definition_cell(self):
        '''
        Generate the code cell containing the definition of the undefined variable
        Returns:
            nbformat.NotebookNode: The new code cell with the variable definition.
    '''
        # Get the source code of the notebook
        source_code = self._get_notebook_source()

        # Generate the prompt
        prompt = (
            f"The variable `{self.undefined_var}` is undefined in cell {self.undefined_index} below.\n"
            f"Provide only the minimal Python code needed to define it. "
            f"Do NOT include any explanations, comments, markdown, or extra text. "
            f"Output only the Python code for the cell, ready to execute.\n\n"
            f"{source_code}"
        )

        attempts = 0
        code_in_text = ""

        while attempts < 3:
            print_msg(f"🕑 Attempt {attempts + 1} to generate definition for {self.undefined_var}", 4)
            try:
                response = localchat(prompt, Name)
                code_in_text = getattr(response, 'code', "")
                if code_in_text.strip():
                    print_msg(f"✅ Code generated for {self.undefined_var}", 5)
                    print_msg(f"▶️ Generated definition for {self.undefined_var}:", 5)
                    print_msg("-" * 40, 5)
                    for line in code_in_text.splitlines():
                        print_msg(line, 5)
                    print_msg("-" * 40, 5)
                    break
                else:
                    print_msg("⚠️ Empty or invalid code received, retrying...", 5)
                    code_in_text = ""
            except Exception as e:
                print_msg(f"❌ Error during LLM call: {e}", 5)
                code_in_text = ""
            attempts += 1
        else:
            print_msg(f"‼️ No valid code generated for {self.undefined_var} after 3 attempts", 4)
            code_in_text = f"{self.undefined_var} = None  # Auto-generated fallback definition"
            print_msg(f"▶️ Using fallback definition: {code_in_text}", 4)

        new_cell = nbformat.v4.new_code_cell(
            source=code_in_text,
            id=str(uuid.uuid4())
        )
            
        return new_cell

    def _get_next_namefixed_version(self, nb_path):
        """
        Generate the next versioned filename for the fixed notebook.
        E.g., if notebook is `analysis.ipynb`, it will generate `analysis_NameFixed_v1.ipynb`,
        and if that exists, `analysis_NameFixed_v2.ipynb`, etc.
        Args:
            nb_path (str): Original notebook path.
        Returns:
            str: New notebook path with incremented version.
        """
        dir_path = os.path.dirname(nb_path)
        filename = os.path.basename(nb_path)
        base_name = os.path.splitext(filename)[0]

        # Strip existing _vN if present
        base_name = re.sub(r"_v\d+$", "", base_name)

        # Final output prefix before adding version
        output_prefix = base_name if base_name.endswith("_NameFixed") else base_name + "_NameFixed"

        # Pattern for existing versions
        pattern = os.path.join(dir_path, f"{output_prefix}_v*.ipynb")
        existing_files = glob.glob(pattern)

        # Extract version numbers
        versions = []
        for f in existing_files:
            match = re.search(r"_v(\d+)\.ipynb$", f)
            if match:
                versions.append(int(match.group(1)))

        next_version = max(versions, default=0) + 1
        output_name = f"{output_prefix}_v{next_version}.ipynb"
        return os.path.join(dir_path, output_name)

    
    def fix_name_error_and_save(self):
        '''
        Fix the NameError in the notebook and return the path of the new notebook
        Returns:
            str: Path to the new notebook with the NameError fixed.
        '''
        # Generate the new cell containing the definition of the undefined variable
        new_cell = self._generate_definition_cell()

        # Insert the new cell at the correct position
        self.content.cells.insert(self.undefined_index - 1, new_cell)

        # Save the notebook with the new cell
        output_nb_path = self._get_next_namefixed_version(self.nb_path)

        with open(output_nb_path, "w", encoding="utf-8") as f:
            nbformat.write(self.content, f)

        return output_nb_path



    def _swap_cells(self, notebook, undefined_cell, defined_cell):
        """
        Swap the positions of the undefined variable cell and the defined variable cell.
        If the undefined cell is the first cell, move the defined cell to the top.
        Args:
            notebook (nbformat.NotebookNode): The notebook object.
            undefined_cell (int): Index of the cell with the undefined variable.
            defined_cell (int): Index of the cell with the variable definition.
        Returns:
            nbformat.NotebookNode: The notebook with swapped cells.
        """
        if undefined_cell == 0:
            cell_to_move = notebook.cells.pop(defined_cell)
            notebook.cells.insert(0, cell_to_move)
        else:
            cell1_pred = undefined_cell
            notebook.cells[cell1_pred], notebook.cells[defined_cell] = notebook.cells[defined_cell], notebook.cells[cell1_pred]

        return notebook


    def get_reordered_notebook_path(self):
        """
        Reorder the notebook by swapping the undefined and defined variable cells,
        and save the new notebook version.
        Returns:
            str: Path to the new reordered notebook.
        """
        print_msg(f"▶️ Trying to swap cell index {self.undefined_index} and {self.defined_index}", 4)
        true_undefined_index = self.undefined_index - 1
        true_defined_index = self.defined_index - 1
        
        new_content = self._swap_cells(self.content, true_undefined_index, true_defined_index)
        new_notebook_path = self._get_next_namefixed_version(self.nb_path)

        with open(new_notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(new_content, f)

        return new_notebook_path