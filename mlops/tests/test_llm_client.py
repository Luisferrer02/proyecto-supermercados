"""Tests for utils/llm_client.py"""

from unittest.mock import MagicMock, patch

from utils.llm_client import _looks_empty, chat_with_failover, resolve_models


class TestResolveModels:
    def test_defaults(self):
        models = resolve_models()
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)

    def test_explicit_list(self):
        assert resolve_models(["a", "b"]) == ["a", "b"]

    @patch.dict("os.environ", {"OPENROUTER_MODELS": "x/model-a, y/model-b"})
    def test_env_override(self):
        result = resolve_models(["ignored"])
        assert result == ["x/model-a", "y/model-b"]

    @patch.dict("os.environ", {"OPENROUTER_MODELS": ""})
    def test_empty_env_falls_through(self):
        result = resolve_models(["fallback"])
        assert result == ["fallback"]


class TestLooksEmpty:
    def test_empty_string(self):
        assert _looks_empty("")

    def test_whitespace(self):
        assert _looks_empty("   \n  ")

    def test_content(self):
        assert not _looks_empty("hello")


class TestChatWithFailover:
    def test_no_api_key_returns_none(self):
        result = chat_with_failover("prompt", api_key="", logger=MagicMock())
        assert result is None

    @patch("utils.llm_client.OpenAI")
    def test_success_first_model(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="result text"))]
        )

        result = chat_with_failover(
            "test prompt",
            api_key="key-123",
            models=["model-a"],
            logger=MagicMock(),
        )
        assert result == "result text"
        mock_client.chat.completions.create.assert_called_once()

    @patch("utils.llm_client.OpenAI")
    def test_empty_response_retries(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(content=""))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="ok"))]),
        ]

        result = chat_with_failover(
            "test", api_key="key", models=["m"], per_model_retries=2, logger=MagicMock()
        )
        assert result == "ok"

    @patch("utils.llm_client.time.sleep")
    @patch("utils.llm_client.OpenAI")
    def test_error_falls_through_to_next_model(self, MockOpenAI, mock_sleep):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            Exception("429 rate limited"),
            MagicMock(choices=[MagicMock(message=MagicMock(content="from model-b"))]),
        ]

        result = chat_with_failover(
            "test",
            api_key="key",
            models=["model-a", "model-b"],
            per_model_retries=1,
            logger=MagicMock(),
        )
        assert result == "from model-b"

    @patch("utils.llm_client.time.sleep")
    @patch("utils.llm_client.OpenAI")
    def test_all_models_fail_returns_none(self, MockOpenAI, mock_sleep):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("500 server error")

        result = chat_with_failover(
            "test",
            api_key="key",
            models=["m1", "m2"],
            per_model_retries=1,
            logger=MagicMock(),
        )
        assert result is None

    @patch("utils.llm_client.OpenAI")
    def test_strips_whitespace(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="  trimmed  \n"))]
        )

        result = chat_with_failover(
            "test", api_key="key", models=["m"], logger=MagicMock()
        )
        assert result == "trimmed"
