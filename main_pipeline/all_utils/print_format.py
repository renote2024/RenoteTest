# === Color helpers ===
def color_status(status):
    status = str(status)
    if "SUCCESS" in status.upper():
        return f"\x1b[32m✅ {status.lower()}\x1b[0m"
    elif "FAILED" in status.upper():
        return f"\x1b[31m❌ {status.lower()}\x1b[0m"
    elif "WARN" in status.upper() or "SKIPPED" in status.upper() or "NON-LINEAR" in status.upper() or "REPEATED" in status.upper() or "NO-EXECUTION" in status.upper():
        return f"\x1b[33m⚠️  {status.lower()}\x1b[0m"
    return status

# === Formatting helpers ===
def print_header(title, indent_num):  # 2 tabs (8 spaces)
    print("    " * indent_num + f"▶ {title}")

def print_status(label, result, indent_num):  # 3 tabs (12 spaces)
    print("    " * indent_num + f"- {label}: {color_status(result)}")

def print_info(msg, indent_num):  # 4 tabs (16 spaces)
    print("    " * indent_num + f">>> {msg}")

# === Notebook Summary Counter ===
class NotebookResultCounter:
    def __init__(self):
        self.success = 0
        self.failed = 0
        self.warning = 0

    def count(self, status):
        status = str(status).upper()
        if "SUCCESS" in status:
            self.success += 1
        elif "FAILED" in status:
            self.failed += 1
        elif "WARN" in status or "SKIPPED" in status or "NON-LINEAR" in status or "REPEATED" in status or "NO-EXECUTION" in status:
            self.warning += 1

    def summary_str(self):
        return (
            f"✅ {self.success} success | "
            f"❌ {self.failed} failed | "
            f"⚠️ {self.warning} warning"
        )
    
    def summary_dict(self):
        return {
            "success": self.success,
            "failed": self.failed,
            "warning": self.warning,
        }

# === Print one notebook result ===
def print_test_result(results, indent_num):
    counter = NotebookResultCounter()
    # print(f"    \x1b[1m\x1b[38;5;215m{repo_idx}.{nb_idx} {nb_name}\x1b[0m")

    if "modules" in results:
        print_header("Test Modules/Imports", indent_num)
        for mod, res in results["modules"].items():
            status = res.get('status')
            version = res.get('version')
            if version:
                status = f"{status} (v. {version})"
            else:
                status = str(status)
            print_status(mod, status, indent_num + 1)
            # print_status(f"{mod} (v. {version})", status)
            counter.count(status)
        # if results.get("req_generated"):
        #     print_info("REQ file generated")

    if "files" in results:
        print_header("Test Input Files", indent_num)
        for f, res in results["files"].items():
            print_status(f"{f}", res, indent_num + 1)
            counter.count(res)
            # if res.upper() == "FAILED" and f in results.get("generated_files", []):
            #     print_info(f"'{f}' is generated")

    if "execution" in results:
        print_header("Test Execution Order", indent_num)
        for key, res in results["execution"].items():
            print_status(key, res, indent_num + 1)
            counter.count(res)

    # === Bold summary line
    print("\n" + indent_num * "    " + f"\x1b[1m=== Summary: {counter.summary_str()} ===\x1b[0m\n")

    return counter.summary_dict()

# === Top-level function to iterate all repos ===
def format_print_results(repos_data):
    """
    Analyze a list of repositories and print the results for each notebook.
    :param repos_data: List of dictionaries containing repository data.
    Each dictionary should have a 'name' key for the repository name and a 'notebooks
    key for a list of notebooks.
    Each notebook should have a 'name' key and a 'results' key with the test results.
    """
   
    total_repos = len(repos_data)
    total_nbs = sum(len(repo["notebooks"]) for repo in repos_data)
    print(f"\nAnalyzing {total_repos} repositories. Total {total_nbs} notebooks\n")

    for i, repo in enumerate(repos_data, 1):
        # Print repository name
        print(f"\x1b[1m\x1b[38;5;201m{i}. {repo['name']}\x1b[0m")

        # Print notebook results
        for j, nb in enumerate(repo["notebooks"], 1):
            print_test_result(i, j, nb["name"], nb["results"])


def print_renote_results(nb_exec_result, indent_num):
    print("\n" + indent_num * "    " + f"\033[1mRenote Results:\033[0m")  # Bold title

    for k, v in nb_exec_result.items():
        print((indent_num + 1) * "    " + f"- {k}: {v}")  # 8-space indent


def print_msg(msg, indent_num):
    print("    " * indent_num, msg)

# === Example dummy data
# if __name__ == "__main__":
#     repos_data = [
#         {
#             "name": "Repo A",
#             "notebooks": [
#                 {
#                     "name": "Notebook A_1",
#                     "results": {
#                         "modules": {"x": "FAILED", "y": "SUCCESS"},
#                         "req_generated": True,
#                         "files": {"x": "SUCCESS", "y": "FAILED"},
#                         "generated_files": ["y"],
#                         "execution": {
#                             "linear": "FAILED",
#                             "patterns": "Non-linear/skipped/repeated"
#                         }
#                     }
#                 },
#                 {
#                     "name": "Notebook A_2",
#                     "results": {
#                         "modules": {"z": "SUCCESS"},
#                         "files": {"a.csv": "SUCCESS"},
#                         "execution": {
#                             "linear": "SUCCESS",
#                             "patterns": "Linear"
#                         }
#                     }
#                 }
#             ]
#         }
#     ]

#     format_print_results(repos_data)
