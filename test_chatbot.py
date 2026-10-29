import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import chatbot


class ChatbotTests(unittest.TestCase):
    def test_rewrite_prompt_uses_requested_level_and_preserves_meaning(self):
        prompt = chatbot.build_rewrite_prompt("It may rain tomorrow.", "extreme")

        self.assertIn("Extreme:", prompt)
        self.assertIn("Preserve its meaning exactly", prompt)
        self.assertIn("It may rain tomorrow.", prompt)
        self.assertIn("Return only the rewritten answer.", prompt)

    def test_rejects_empty_message_before_calling_model(self):
        with patch.object(chatbot, "_model_request") as model_request:
            with self.assertRaisesRegex(ValueError, "enter a message"):
                chatbot.generate_reply("  ", "medium", [])
        model_request.assert_not_called()

    def test_rejects_unknown_style_before_calling_model(self):
        with patch.object(chatbot, "_model_request") as model_request:
            with self.assertRaisesRegex(ValueError, "valid style level"):
                chatbot.generate_reply("Hello", "unlisted", [])
        model_request.assert_not_called()

    def test_generates_and_scores_rewrite(self):
        score = json.dumps({"fidelity": 96, "clarity": 92, "reason": "Meaning preserved."})
        with patch.object(
            chatbot,
            "_model_request",
            side_effect=["A normal answer.", "A tangled answer.", score],
        ) as model_request:
            base, reply, evaluation = chatbot.generate_reply(
                "Hello", "mild", [{"user": "Earlier question", "base": "Earlier answer."}]
            )

        self.assertEqual((base, reply), ("A normal answer.", "A tangled answer."))
        self.assertEqual(evaluation, {
            "fidelity": 96, "clarity": 92, "adjusted": False, "fallback": False,
        })
        first_messages = model_request.call_args_list[0].args[0]
        self.assertEqual(first_messages[-3]["role"], "user")
        self.assertEqual(first_messages[-2]["content"], "Earlier answer.")
        rewrite_prompt = model_request.call_args_list[1].args[0][0]["content"]
        self.assertIn("A normal answer.", rewrite_prompt)
        self.assertIn("Mild:", rewrite_prompt)
        self.assertTrue(model_request.call_args_list[2].kwargs["expect_json"])

    def test_low_clarity_triggers_rewrite_and_rescore(self):
        with patch.object(
            chatbot,
            "_model_request",
            side_effect=[
                "A clear answer.", "An unclear answer.",
                '{"fidelity":95,"clarity":45,"reason":"Too vague."}',
                "A simpler, clear answer.",
                '{"fidelity":98,"clarity":91,"reason":"Clear now."}',
            ],
        ) as model_request:
            base, reply, evaluation = chatbot.generate_reply("Hello", "extreme", [])

        self.assertEqual((base, reply), ("A clear answer.", "A simpler, clear answer."))
        self.assertEqual(evaluation, {
            "fidelity": 98, "clarity": 91, "adjusted": True, "fallback": False,
        })
        self.assertIn("simplify confusing or vague wording", model_request.call_args_list[3].args[0][0]["content"])

    def test_persistent_guardrail_failure_returns_clear_base_answer(self):
        with patch.object(
            chatbot,
            "_model_request",
            side_effect=[
                "Clear source answer.", "Bad rewrite.",
                '{"fidelity":40,"clarity":30}',
                "Still bad.",
                '{"fidelity":50,"clarity":40}',
            ],
        ):
            base, reply, evaluation = chatbot.generate_reply("Hello", "medium", [])

        self.assertEqual((base, reply), ("Clear source answer.", "Clear source answer."))
        self.assertTrue(evaluation["adjusted"])
        self.assertTrue(evaluation["fallback"])
        self.assertEqual((evaluation["fidelity"], evaluation["clarity"]), (100, 100))

    def test_rejects_invalid_score(self):
        with patch.object(chatbot, "_model_request", return_value='{"fidelity":101,"clarity":90}'):
            with self.assertRaisesRegex(chatbot.ChatbotError, "invalid meaning/clarity score"):
                chatbot._evaluate_answer("Source.", "Rewrite.")

    def test_loads_dotenv_without_overwriting_existing_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                'CHATBOT_TEST_DOTENV="loaded value"\n'
                'CHATBOT_TEST_EXISTING=from-file\n'
                "# ignored comment\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"CHATBOT_TEST_EXISTING": "from-shell"}, clear=False):
                os.environ.pop("CHATBOT_TEST_DOTENV", None)
                chatbot._load_dotenv(env_path)
                self.assertEqual(os.environ["CHATBOT_TEST_DOTENV"], "loaded value")
                self.assertEqual(os.environ["CHATBOT_TEST_EXISTING"], "from-shell")

    def test_ollama_request_needs_no_api_key(self):
        response_data = {"message": {"content": "Local model reply."}}
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "ollama",
            "LLM_BASE_URL": "http://localhost:11434",
            "LLM_MODEL": "test-model",
        }, clear=False), patch.object(
            chatbot.urllib.request, "urlopen"
        ) as urlopen:
            response = urlopen.return_value.__enter__.return_value
            response.read.return_value = json.dumps(response_data).encode()
            reply = chatbot._model_request(
                [{"role": "user", "content": "Hello"}], expect_json=True
            )

        request = urlopen.call_args.args[0]
        request_body = json.loads(request.data)
        self.assertEqual(reply, "Local model reply.")
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        self.assertEqual(request_body["model"], "test-model")
        self.assertEqual(request_body["format"], "json")
        self.assertEqual(request_body["stream"], False)
        self.assertIsNone(request.get_header("Authorization"))

    def test_requires_api_key_for_openai_provider(self):
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "",
        }, clear=False):
            with self.assertRaisesRegex(chatbot.ChatbotError, "OPENAI_API_KEY"):
                chatbot._model_request([{"role": "user", "content": "Hello"}])


if __name__ == "__main__":
    unittest.main()
