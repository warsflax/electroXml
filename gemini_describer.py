"""
GeminiDescriber - Appel a l'API Gemini pour generer une description de schema.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from config import Config


class GeminiDescriber:
    """Envoie les donnees compactes a Gemini et retourne une description."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or self._load_api_key()
        self.client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _load_api_key() -> str:
        load_dotenv()
        api_key = os.getenv("API_KEY_IA")
        if not api_key:
            raise RuntimeError(
                "La variable API_KEY_IA est absente.\n"
                "Ajoutez-la dans le fichier .env :\n"
                "API_KEY_IA=VOTRE_CLE_GEMINI"
            )
        return api_key

    def describe(self, compact_data: str) -> str:
        """Retourne une description generee par Gemini a partir des donnees compactes."""
        try:
            response = self.client.models.generate_content(
                model=Config.MODEL_NAME,
                contents=compact_data,
                config=types.GenerateContentConfig(
                    system_instruction=Config.SYSTEM_PROMPT,
                    temperature=Config.TEMPERATURE,
                    max_output_tokens=Config.MAX_OUTPUT_TOKENS,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=Config.THINKING_LEVEL
                    ),
                ),
            )

            if response.candidates and response.candidates[0].finish_reason == "MAX_TOKENS":
                raise RuntimeError(
                    "Reponse tronquee : max_output_tokens atteint (reflexion + texte). "
                    "Augmente Config.MAX_OUTPUT_TOKENS."
                )

            if not response.text:
                raise RuntimeError("Gemini a retourne une reponse vide.")

            return response.text

        except RuntimeError:
            raise
        except Exception as exc:
            message = str(exc)

            if "RESOURCE_EXHAUSTED" in message or "429" in message:
                raise RuntimeError(
                    "Quota Gemini epuise (HTTP 429).\n"
                    "Les credits du projet sont insuffisants."
                ) from exc

            if "NOT_FOUND" in message or "404" in message:
                raise RuntimeError(
                    f"Modele Gemini indisponible : {Config.MODEL_NAME}"
                ) from exc

            raise RuntimeError(f"Erreur lors de l'appel a Gemini : {message}") from exc
