from main_pipeline.renote_utils.localOllama import localchat
from main_pipeline.all_utils.print_format import print_msg

from pydantic import BaseModel

class Module(BaseModel):
    name: str

class FixModuleNotFound:
    def __init__(self, missing_module):
        self.missing_module = missing_module

    def find_correct_module(self):
        prompt = (
            f"Find the correct pip-installable module name for `{self.missing_module}`. "
            f"If none, return None."
        )
        try:
            response = localchat(prompt, Module)
            name = getattr(response, 'name', None)
            if isinstance(name, str) and name:
                name = name.strip()
                if name and name.lower() != 'none':
                    print_msg(f"✅ Found correct name for {self.missing_module}: {name}", 4)
                    return name
            else:
                print_msg(f"⚠️ LLM returned None or invalid response for {self.missing_module}", 4)
        except Exception as e:
            print_msg(f"⚠️ Failed to get module name: {e}", 4)
        return None
