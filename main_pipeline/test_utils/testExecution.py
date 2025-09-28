import nbformat
import os
import papermill as pm
from main_pipeline.all_utils.print_format import print_msg

class TestExecution:
    def __init__(self, nb_path):
        self.nb_path = nb_path
        with open(nb_path, 'r', encoding='utf-8') as f:
            self.nb = nbformat.read(f, as_version=4)

    def is_linear_execution(self, nb_path):
        """
        Test if a notebook can be executed linearly from start to finish without errors.
        Args:
            nb_path (str): Path to the notebook file.
        Returns:
            str: "success" if executed successfully, "timeout" if timed out, "failed"
                    if execution failed.
        """
        notebook_dir = os.path.dirname(nb_path)
        try:
            pm.execute_notebook(
                input_path=nb_path,
                output_path=None,
                timeout=300,
                kernel_name="python3",
                progress_bar=False,
                cwd=notebook_dir
            )
            print_msg(f"✅ Passed papermill linear test", 4)
            return "success"
        except TimeoutError:
            print_msg(f"❌ Timeout papermill linear test", 4)
            return "timeout"
        except Exception as e:
            print_msg(f"❌ Failed papermill linear test", 4)
            return "failed"

    def categorize_execution_order(self):
        """
        Categorizes a notebook based on its cells' execution order sequence
        1. Non-linear execution order
        2. Repeatedly executed cells
        3. Skipped cells
        4. Linear execution order
        5. No execution at all

        :param notebook: The notebook object to analyze
        :return: categorization result
        """

        # Extract execution counts
        execution_counts = [
            cell.get('execution_count')
            for cell in self.nb.cells if cell.cell_type == 'code'
        ]
        # Filter out None values (cells that were not executed)
        counts_non_null = [c for c in execution_counts if c is not None and c > 0]

        if not counts_non_null:
            return ["no-execution"]

        # Detect skipped cells
        has_skipped = len(counts_non_null) < len(execution_counts)

        # Detect non-linear execution
        has_non_linear = counts_non_null != sorted(counts_non_null)

        # Detect repeated execution by identifying missing counts in the range
        expected = set(range(min(counts_non_null), max(counts_non_null) + 1))
        actual = set(counts_non_null)
        has_missing_counts = expected != actual

        # Compile results
        patterns = []
        if has_skipped:
            patterns.append("skipped")
        if has_missing_counts:
            patterns.append("repeated")
        if has_non_linear:
            patterns.append("non-linear")

        return patterns
    
    def analyze(self):
        """
        Analyze the notebook for execution patterns.
        Returns:
            dict: A dictionary with execution analysis results.
        """

        # EXECUTION
        isLinear = self.is_linear_execution(self.nb_path)

        if isLinear == "success":
            patterns = ["linear"]
        else:
            patterns =  self.categorize_execution_order()

        return {
            "linear": isLinear,
            "patterns": patterns
        }