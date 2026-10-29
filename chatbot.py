import ast
import datetime
import json
import operator
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
    provider = os.environ.get("LLM_PROVIDER", "openai").strip().lower()
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
                "Set OPENAI_API_KEY in .env, or set a key for your configured OpenAI-compatible provider."
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
            parsed_detail = json.loads(detail)
            if isinstance(parsed_detail, dict):
                nested_error = parsed_detail.get("error")
                if isinstance(nested_error, dict):
                    detail = nested_error.get("message", detail)
                elif isinstance(nested_error, str):
                    detail = nested_error
        except json.JSONDecodeError:
            pass
        if provider == "ollama" and error.code == 404 and "model" in detail.lower() and "not found" in detail.lower():
            raise ChatbotError(
                f"Ollama model '{model}' is not installed. Run 'ollama pull {model}' and try again."
            ) from error
        raise ChatbotError(f"Language model API returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        reason_text = str(error.reason)
        winerror = getattr(error.reason, "winerror", None)
        connection_refused = (
            winerror == 10061
            or isinstance(error.reason, ConnectionRefusedError)
            or "actively refused" in reason_text.lower()
            or "connection refused" in reason_text.lower()
        )
        if provider == "ollama" and connection_refused:
            raise ChatbotError(
                f"Ollama is not running or is unavailable at {base_url}. "
                "Start Ollama, then try again."
            ) from error
        raise ChatbotError(f"Could not reach the language model: {reason_text}") from error
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


def _local_answer(message):
    text = message.strip()
    lower = text.lower()

    math_match = re.fullmatch(r"(?:what is|calculate)?\s*([0-9()+\-*/%. ]+)[?]?", lower)
    if math_match:
        expression = math_match.group(1).strip()
        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
        }

        def evaluate(node):
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in allowed_operators:
                return allowed_operators[type(node.op)](evaluate(node.left), evaluate(node.right))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = evaluate(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            raise ValueError("Unsupported expression")

        if expression and any(character.isdigit() for character in expression):
            try:
                result = evaluate(ast.parse(expression, mode="eval"))
                if isinstance(result, float) and result.is_integer():
                    result = int(result)
                return f"{expression} equals {result}."
            except (SyntaxError, ValueError, ZeroDivisionError, TypeError):
                pass

    if "what time" in lower:
        now = datetime.datetime.now().strftime("%I:%M %p")
        return f"The current local time is {now}."

    if "what date" in lower or "what day is it" in lower:
        today = datetime.datetime.now().strftime("%A, %B %d, %Y")
        return f"Today is {today}."

    if "infinite games" in lower and "bacon" in lower:
        return (
            "This sounds like a riddle or word puzzle, but literally it is a contradiction: infinity cannot contain game instances without including games. "
            "A coherent interpretation is that the phrase describes an endless supply of something whose contents are not currently playable: an archive could hold unlimited game names, or possible game configurations might exist without a single game being assembled. "
            "If the intended comparison is between abstract possibilities and actual objects, say it directly: there can be infinitely many possible games, but no game has yet been played or created."
        )

    facts = {
        "what is gravity": "Gravity is the attraction between objects with mass. It keeps planets in orbit and pulls objects toward Earth.",
        "what is photosynthesis": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to make sugars, releasing oxygen as a by-product.",
        "why is the sky blue": "The sky looks blue because molecules in the atmosphere scatter shorter blue wavelengths of sunlight more strongly than longer red wavelengths.",
        "what causes rain": "Rain begins when water vapor in clouds cools and condenses into droplets. When those droplets become heavy enough, gravity pulls them to the ground.",
        "what is python": "Python is a general-purpose programming language known for readable syntax. It is commonly used for web development, automation, data analysis, artificial intelligence, and scripts.",
        "what is dna": "DNA is the molecule that stores genetic instructions in living organisms. Its sequence helps cells build proteins and pass hereditary information to offspring.",
        "what is evolution": "Evolution is change in inherited traits of populations over generations, driven by mechanisms such as natural selection, mutation, and genetic drift.",
        "what is electricity": "Electricity is the movement or presence of electric charge. Electric current is the flow of charged particles through a conductor.",
        "what is the internet": "The Internet is a global network of networks that lets computers exchange information using shared protocols, especially TCP/IP.",
    }
    for question, answer in facts.items():
        if question in lower:
            return answer
    if re.fullmatch(r"(hi|hello|hey)[.! ]*", lower):
        return "Hello. Ask me a question and I can answer with the app's built-in local knowledge, or with a hosted AI model when one is configured."
    if "how are you" in lower:
        return "I am ready to help. I can answer several built-in questions directly, and hosted AI provides broader coverage when configured."
    if "thank" in lower:
        return "You are welcome."
    if "help" in lower:
        return "You can ask a general knowledge question, such as what gravity is, why the sky is blue, or how photosynthesis works. Hosted AI can answer a broader range of questions when configured."
    return (
        "I can handle built-in questions about greetings, simple calculations, dates and times, "
        "and several general topics such as gravity, electricity, DNA, Python, and photosynthesis. "
        "Please rephrase this question using one of those topics, or add a hosted model provider for broader answers."
    )


def _local_tangle(answer, level):
    if level == "mild":
        return f"Put simply, {answer[0].lower()}{answer[1:]}"

    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", answer) if sentence.strip()]
    if level == "medium":
        tangled_sentences = []
        for index, sentence in enumerate(sentences):
            sentence = sentence.rstrip(".!?")
            if index == 0:
                tangled_sentences.append(f"The answer, when approached with due qualification, is that {sentence.lower()}.")
            else:
                tangled_sentences.append(f"This is complemented, moreover, by the consideration that {sentence.lower()}.")
        return " ".join(tangled_sentences)

    tangled_sentences = []
    for index, sentence in enumerate(sentences):
        sentence = sentence.rstrip(".!?")
        opening = "Consider first" if index == 0 else "Having reached that point, consider"
        tangled_sentences.append(
            f"{opening} (though the reason may initially appear indirect) the proposition "
            f"that {sentence.lower()}, which, once its qualifications have been entertained, remains the point."
        )
    return " ".join(tangled_sentences)


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

    provider = os.environ.get("LLM_PROVIDER", "openai").strip().lower()
    if provider in ("openai", "openai-compatible") and not os.environ.get("OPENAI_API_KEY", "").strip():
        base_answer = _local_answer(message)
        candidate = _local_tangle(base_answer, level)
        return base_answer, candidate, {
            "fidelity": 100,
            "clarity": 100,
            "adjusted": False,
            "fallback": False,
            "local": True,
        }

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
