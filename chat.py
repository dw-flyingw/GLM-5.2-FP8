#!/usr/bin/env python3
"""Simple streaming chat CLI for the GLM-5.2-FP8 Dynamo (SGLang) server.

Talks to the OpenAI-compatible endpoint that the Dynamo frontend exposes
(./dynamo/serve.sh): host port :8000, served model "glm-5.2-fp8",
/v1/chat/completions, and a `reasoning_content` delta stream
(Dynamo's --dyn-reasoning-parser glm45).
Stdlib only — no pip install needed.

Usage:
    ./chat.py                       # connect to http://localhost:8000
    BASE_URL=http://host:8001 ./chat.py
    ./chat.py --no-think            # hide the model's reasoning stream

In-chat commands:
    /reset    clear the conversation history
    /think    toggle showing reasoning
    /exit     quit  (also Ctrl-D / Ctrl-C)
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
MODEL = os.environ.get("MODEL", "glm-5.2-fp8")

# GLM is bilingual and defaults to Chinese with no steering. Reply in the
# user's language so an English prompt gets an English answer. Override with
# --system, or disable with --no-system.
DEFAULT_SYSTEM = (
    "You are a helpful assistant. Respond in the same language the user "
    "writes in; default to English."
)

# ANSI styling (skipped if stdout is not a tty).
_TTY = sys.stdout.isatty()
DIM = "\033[2m" if _TTY else ""
CYAN = "\033[36m" if _TTY else ""
GREEN = "\033[32m" if _TTY else ""
RESET = "\033[0m" if _TTY else ""


def stream_completion(messages, show_think):
    """POST to /v1/chat/completions with stream=True, yield text as it arrives.

    Prints reasoning_content dimmed (when show_think) and returns the final
    assistant content so it can be appended to the history.
    """
    payload = json.dumps(
        {"model": MODEL, "messages": messages, "stream": True}
    ).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    answer = []
    in_think = False
    in_answer = False
    try:
        with urllib.request.urlopen(req) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

                reasoning = delta.get("reasoning_content")
                if reasoning and show_think:
                    if not in_think:
                        sys.stdout.write(f"{DIM}[thinking] ")
                        in_think = True
                    sys.stdout.write(reasoning)
                    sys.stdout.flush()

                content = delta.get("content")
                if content:
                    if in_think:
                        sys.stdout.write(f"{RESET}\n")
                        in_think = False
                    if not in_answer:
                        sys.stdout.write(f"{GREEN}")
                        in_answer = True
                    sys.stdout.write(content)
                    sys.stdout.flush()
                    answer.append(content)
    except urllib.error.URLError as e:
        sys.stdout.write(f"\n{RESET}[connection error: {e}. Is the server up? "
                         f"Try: curl {BASE_URL}/v1/models]\n")
        return None

    if in_think:
        sys.stdout.write(RESET)
    if in_answer:
        sys.stdout.write(RESET)
    sys.stdout.write("\n")
    return "".join(answer)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-think", action="store_true",
                        help="hide the model's reasoning stream")
    parser.add_argument("--system", default=None,
                        help="override the default system prompt")
    parser.add_argument("--no-system", action="store_true",
                        help="send no system prompt at all")
    args = parser.parse_args()

    show_think = not args.no_think
    messages = []
    system = None if args.no_system else (args.system or DEFAULT_SYSTEM)
    if system:
        messages.append({"role": "system", "content": system})

    print(f"{CYAN}GLM-5.2-FP8 chat{RESET}  ({BASE_URL}, model={MODEL})")
    print(f"{DIM}/reset  /think  /exit   (Ctrl-D to quit){RESET}")

    while True:
        try:
            user = input(f"{CYAN}you> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            messages = [m for m in messages if m["role"] == "system"]
            print(f"{DIM}(history cleared){RESET}")
            continue
        if user == "/think":
            show_think = not show_think
            print(f"{DIM}(reasoning {'shown' if show_think else 'hidden'}){RESET}")
            continue

        messages.append({"role": "user", "content": user})
        reply = stream_completion(messages, show_think)
        if reply is None:
            messages.pop()  # drop the user turn so a retry is clean
        else:
            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
