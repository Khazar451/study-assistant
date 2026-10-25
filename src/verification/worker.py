"""Helper process for TimeLimitedVerifier.

Reads one JSON request per line on stdin ({"answer": "..."}) and writes one JSON reply per line on
stdout ({"result": <verification or null>}). Started with `python -m src.verification.worker`.
"""

import json
import sys


def main() -> None:
    replies = sys.stdout
    sys.stdout = sys.stderr  # anything SymPy prints must not corrupt the reply stream

    from src.verification.symbolic_verifier import verify_math_solution

    verify_math_solution(r"$$\frac{d}{dx}x^2 = 2x$$")  # warm the LaTeX parser before reporting ready
    _reply(replies, {"ready": True})

    for line in sys.stdin:
        try:
            result = verify_math_solution(json.loads(line)["answer"])
        except Exception:
            result = None
        _reply(replies, {"result": result})


def _reply(stream, message: dict) -> None:
    stream.write(json.dumps(message) + "\n")
    stream.flush()


if __name__ == "__main__":
    main()
