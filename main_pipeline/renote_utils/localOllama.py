from ollama import chat
from pydantic import BaseModel

def localchat(prompt: str, schema: type[BaseModel]) -> BaseModel:
    response = chat(
        messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ],
        model='gemma3:12b',
        format=schema.model_json_schema(),
    )
    return schema.model_validate_json(response['message']['content'])
