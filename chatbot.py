import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path


def _load_dotenv(path):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        os.environ.setdefault(key, value)


_load_dotenv(Path(__file__).resolve().parents[1] / ".env")

CONFUSION_LEVELS = {
    "mild": (
        "Mild: add occasional subordinate clauses and a restrained parenthetical "
        "aside, keeping the answer close to ordinary speech."
    ),
    "medium": (
        "Medium: use layered clauses, a few well-placed parenthetical asides, "
        "and some passive constructions while remaining easy to follow."
    ),
    "extreme": (
        "Extreme: use noticeably nested clauses, purposeful asides, passive "
        "voice, and theatrical indirection, but remain grammatical and readable. "
        "Never sacrifice clarity or omit any part of the meaning."
    ),
}
FIDELITY_THRESHOLD = 90
CLARITY_THRESHOLD = 80


class ChatbotError(Exception):
    """Raised when the configured language model cannot complete a chat reply."""


def build_rewrite_prompt(answer, level):
    style = CONFUSION_LEVELS[level]
    return (
        "Rewrite the answer below in polished, deliberately tangled English. "
        "Preserve its meaning exactly, including facts, names, numbers, caveats, "
        "uncertainty, and intent. Do not add, remove, or correct claims. Use "
        "syntactically valid grammar and sound like a clever, slightly theatrical "
        "human speaker. Make it more difficult to parse through sentence structure, "
        "not obscure vocabulary. Return only the rewritten answer. "
        f"Style level: {style}\n\nAnswer to rewrite:\n{answer}"
    )


def _score_prompt(original, candidate):
    return (
        "Evaluate the rewritten answer against the original answer. Score meaning "
        "fidelity from 0 to 100 (100 means every claim, fact, number, caveat, "
        "uncertainty, and intent is preserved) and clarity from 0 to 100 (100 "
        "means precise, grammatical, and easy to understand; vague, confusing, "
        "or ungrammatical wording lowers this score). Return only a JSON object "
        'with numeric "fidelity" and "clarity" fields, each an integer from 0 to '
        '100, and a short string "reason".\n\n'
        f"Original answer:\n{original}\n\nRewritten answer:\n{candidate}"
    )


def _model_request(messages, expect_json=False):
    provider = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()
    base_url = os.environ.get(
        "LLM_BASE_URL",
        "http://127.0.0.1:11434" if provider == "ollama" else "https://api.openai.com/v1",
    ).strip().rstrip("/")
    default_model = "llama3.2" if provider == "ollama" else "gpt-4o-mini"
    model = os.environ.get("LLM_MODEL", default_model).strip()
    if not base_url or not model:
        raise ChatbotError("LLM_BASE_URL and LLM_MODEL must not be empty.")

    body = {"model": model, "messages": messages, "temperature": 0.7}
    headers = {"Content-Type": "application/json"}
    if provider == "ollama":
        endpoint = f"{base_url}/api/chat"
        body["stream"] = False
        if expect_json:
            body["format"] = "json"
    elif provider in ("openai", "openai-compatible"):
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ChatbotError(
                "Set OPENAI_API_KEY in .env or choose LLM_PROVIDER=ollama."
            )
        endpoint = f"{base_url}/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        if expect_json:
            body["response_format"] = {"type": "json_object"}
    else:
        raise ChatbotError(
            "LLM_PROVIDER must be 'ollama', 'openai', or 'openai-compatible'."
        )

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(detail).get("error", {}).get("message", detail)
        except (json.JSONDecodeError, AttributeError):
            pass
        raise ChatbotError(f"Language model API returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise ChatbotError(f"Could not reach the language model: {error.reason}") from error
    except (TimeoutError, json.JSONDecodeError) as error:
        raise ChatbotError(f"Could not read a valid language model response: {error}") from error

    try:
        if provider == "ollama":
            content = data["message"]["content"]
        else:
            content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ChatbotError("The language model returned an unexpected response.") from error

    if not isinstance(content, str) or not content.strip():
        raise ChatbotError("The language model returned an empty response.")
    return content.strip()


def _evaluate_answer(original, candidate):
    raw_score = _model_request(
        [{"role": "user", "content": _score_prompt(original, candidate)}],
        expect_json=True,
    )
    try:
        score = json.loads(raw_score)
        fidelity = score["fidelity"]
        clarity = score["clarity"]
        if (
            isinstance(fidelity, bool)
            or not isinstance(fidelity, (int, float))
            or isinstance(clarity, bool)
            or not isinstance(clarity, (int, float))
            or not 0 <= fidelity <= 100
            or not 0 <= clarity <= 100
        ):
            raise ValueError("scores must be numbers from 0 to 100")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ChatbotError(f"The model returned an invalid meaning/clarity score: {error}") from error

    return {
        "fidelity": round(fidelity),
        "clarity": round(clarity),
        "reason": score.get("reason", ""),
    }


def _build_guardrail_prompt(original, candidate, evaluation, level):
    problems = []
    if evaluation["fidelity"] < FIDELITY_THRESHOLD:
        problems.append(
            "restore every detail, caveat, uncertainty, and implication from the original"
        )
    if evaluation["clarity"] < CLARITY_THRESHOLD:
        problems.append(
            "simplify confusing or vague wording and make the sentences immediately understandable"
        )
    problems_text = "; ".join(problems)
    return (
        "Revise the candidate answer, using the original as the source of truth. "
        "Keep all original meaning and do not invent facts. Address these issues: "
        f"{problems_text}. Prefer clarity over theatrical complexity; if needed, "
        "use a milder style than requested. Return only the revised answer.\n"
        f"Requested style: {CONFUSION_LEVELS[level]}\n"
        f"Original answer:\n{original}\n\nCandidate to repair:\n{candidate}"
    )


def generate_reply(message, level, history):
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Please enter a message.")
    if len(message) > 4000:
        raise ValueError("Messages must be 4,000 characters or fewer.")
    if not isinstance(level, str) or level not in CONFUSION_LEVELS:
        raise ValueError("Choose a valid style level: mild, medium, or extreme.")

    messages = [{
        "role": "system",
        "content": "Answer the user helpfully and accurately in clear, ordinary English. "
        "Do not imitate a tangled style; that will be applied in a separate step.",
    }]
    for turn in history[-10:]:
        messages.extend((
            {"role": "user", "content": turn["user"]},
            {"role": "assistant", "content": turn["base"]},
        ))
    messages.append({"role": "user", "content": message.strip()})

    base_answer = _model_request(messages)
    candidate = _model_request([{
        "role": "user",
        "content": build_rewrite_prompt(base_answer, level),
    }])
    evaluation = _evaluate_answer(base_answer, candidate)
    adjusted = False
    fallback = False

    if (
        evaluation["fidelity"] < FIDELITY_THRESHOLD
        or evaluation["clarity"] < CLARITY_THRESHOLD
    ):
        adjusted = True
        repaired = _model_request([{
            "role": "user",
            "content": _build_guardrail_prompt(base_answer, candidate, evaluation, level),
        }])
        evaluation = _evaluate_answer(base_answer, repaired)
        if (
            evaluation["fidelity"] < FIDELITY_THRESHOLD
            or evaluation["clarity"] < CLARITY_THRESHOLD
        ):
            candidate = base_answer
            evaluation = {"fidelity": 100, "clarity": 100, "reason": "Used the clear original answer."}
            fallback = True
        else:
            candidate = repaired
    return base_answer, candidate, {
        "fidelity": evaluation["fidelity"],
        "clarity": evaluation["clarity"],
        "adjusted": adjusted,
        "fallback": fallback,
    }
