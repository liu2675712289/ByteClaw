import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from byteclaw.providers import openai_provider


class OpenAIProviderTests(unittest.TestCase):
    def test_create_model_falls_back_to_project_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "OPENAI_API_KEY=test-key\nOPENAI_MODEL=test-model\n",
                encoding="utf-8",
            )

            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(
                    openai_provider,
                    "_PROJECT_ENV_FILE",
                    env_file,
                ),
                patch.object(openai_provider, "find_dotenv", return_value=""),
                patch.object(openai_provider, "ChatOpenAI") as chat_openai,
            ):
                model = openai_provider.create_model()

        chat_openai.assert_called_once_with(
            api_key="test-key",
            model="test-model",
            temperature=0,
        )
        self.assertIs(model, chat_openai.return_value)


if __name__ == "__main__":
    unittest.main()
