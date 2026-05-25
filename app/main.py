import argparse
import json
import os
import sys

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    chat = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=[{"role": "user", "content": args.p}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read and return the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to read",
                            }
                        },
                        "required": ["file_path"],
                    },
                },
            }
        ],
    )

    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")

    # You can use print statements as follows for debugging, they'll be visible when running tests.
    print("Logs from your program will appear here!", file=sys.stderr)

    if not chat.choices[0].message:
        raise RuntimeError("no message in first chat choice")

    tool_calls = chat.choices[0].message.tool_calls
    if not tool_calls or len(tool_calls) == 0:
        print(chat.choices[0].message.content)
        return

    first_tool_call = tool_calls[0]
    if not first_tool_call.function:
        raise RuntimeError("no function in first tool call")

    if first_tool_call.function.name == "Read":
        arguments = json.loads(first_tool_call.function.arguments)
        f = open(arguments["file_path"], "r")
        content = f.read()
        print(content)

        f.close()
    else:
        print("Something went wrong")


if __name__ == "__main__":
    main()
