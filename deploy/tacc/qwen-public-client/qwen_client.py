"""Call the course Qwen service using only the Python standard library."""
import argparse
import json
import os
from urllib.request import Request, urlopen


def chat(messages, base_url, max_tokens=256, temperature=0.7):
    """Return the full JSON response, including choices and token usage."""
    payload = {
        "model": "qwen3.8-27b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=300) as response:
        return json.load(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("QWEN_BASE_URL"))
    parser.add_argument("--prompt", default="37加58等于多少？只输出数字。")
    args = parser.parse_args()
    if not args.base_url:
        parser.error("set QWEN_BASE_URL=http://NODE:8000/v1 or pass --base-url")
    result = chat([{ "role": "user", "content": args.prompt }], args.base_url,
                  max_tokens=64, temperature=0)
    print(result["choices"][0]["message"]["content"])
    print(json.dumps(result.get("usage", {}), ensure_ascii=False))
