# Tangled Chat

A dependency-free Python chat app that answers in two stages: a language model
first produces a plain, helpful answer, then rewrites it in deliberately
layered, grammatical English. A second model check scores meaning fidelity and
clarity. If either score misses its threshold, the draft is repaired and scored
again; if it still misses, the app returns the clear original answer. Up to ten
previous turns are kept in memory per browser conversation.

## Requirements

- Python 3.10 or later
- Either [Ollama](https://ollama.com/) with a downloaded chat model, or an API
  key for an OpenAI-compatible chat-completions service

## Run locally with Ollama

1. Install Ollama, then download a model in PowerShell:

   ```powershell
   ollama pull llama3.2
   ```

2. Copy `.env.example` to `.env`. Its defaults select Ollama on
   `http://127.0.0.1:11434` with the `llama3.2` model.
3. Start the app from the project directory:

   ```powershell
   python backend\server.py
   ```

4. Open <http://127.0.0.1:8000>.

The app loads `.env` automatically at startup. Existing shell environment
variables take precedence over values in the file. `.env.example` is only a
template; do not put real API keys in files committed to source control.

## Use an OpenAI-compatible provider

Edit `.env` to select the provider and set its key, URL, and model:

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

`LLM_PROVIDER=openai-compatible` also uses the chat-completions API format and
the configured `LLM_BASE_URL`. Restart the server after changing environment
settings.

## Style and guardrails

The UI lets you choose **Mild**, **Medium**, or **Extreme** for each message.
These descriptions live in `CONFUSION_LEVELS` in `backend/chatbot.py`. The
meaning-fidelity threshold is `FIDELITY_THRESHOLD` (default 90/100), and the
clarity threshold is `CLARITY_THRESHOLD` (default 80/100). If the rewritten
answer falls below either threshold, one repair pass is made with clarity taking
priority over theatrical complexity. A failed second check returns the clear
original answer instead. Each result displays its fidelity and clarity scores
and whether the guardrail intervened.

The scores and rewrite are model-generated estimates, not formal proofs that
meaning is unchanged.

## Example

**User:** Why is the sky blue?

**Normal answer:** Sunlight contains many colors. Air molecules scatter shorter
blue wavelengths more than longer red wavelengths, so blue light reaches your
eyes from across the sky.

**Possible rewritten answer (medium):** The sky appears blue because sunlight,
which contains many colors, is scattered by air molecules more strongly at the
shorter blue wavelengths than at the longer red ones; and so it is blue light
that reaches your eyes from across the sky.

Actual wording and scores vary by model.

## API

`POST /api/chat`

```json
{
  "message": "Why is the sky blue?",
  "level": "medium",
  "sessionId": null
}
```

`level` is `mild`, `medium`, or `extreme` and defaults to `medium`. The response
includes the rewritten `reply`, a `sessionId` for follow-up messages, and an
`evaluation` object with `fidelity`, `clarity`, `adjusted`, and `fallback`
fields. `POST /api/reset` with `{"sessionId":"..."}` clears that conversation.
Sessions are in-memory and are lost when the server restarts.

## Tests

```powershell
python -m unittest discover -s tests -v
```
