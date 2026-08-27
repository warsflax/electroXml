"""
Config - Parametres de configuration pour l'analyse de schema XML avec Gemini.
"""

from __future__ import annotations


class Config:
    """Parametres de configuration du script."""

    MODEL_NAME = "gemini-3.5-flash"

    # Limite de securite : on ne transmet jamais un XML gigantesque.
    MAX_INPUT_CHARS = 8000

    # Limite de sortie. Une description tient largement dans ce budget,
    # y compris avec la reflexion interne du modele (thinking_level="low").
    MAX_OUTPUT_TOKENS = 1000

    # Temperature basse = reponse plus deterministe et moins bavarde.
    TEMPERATURE = 0.1

    THINKING_LEVEL = "low"

    SYSTEM_PROMPT = """
Tu es un ingenieur electronique senior specialise en logique numerique.

Redige uniquement une description technique du schema, a partir des
donnees fournies dans le message utilisateur.

REGLES STRICTES :
1. N'invente aucune information absente des donnees.
2. Si une information essentielle est manquante, ecris "Non disponible".
3. Sois factuel, precis et concis - aucune explication generale hors sujet.

CONTENU ATTENDU (en un paragraphe continu, sans titres) :
- role et fonctionnement global du circuit ;
- entrees et sorties principales et leur role ;
- deroulement logique resume : entree -> traitement -> sortie.

Reponse maximale : 120 mots.
"""
