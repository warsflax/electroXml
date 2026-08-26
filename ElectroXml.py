"""
ElectroXml - Analyseur de schéma électronique XML avec Gemini.

Version optimisée pour réduire la consommation de tokens/crédits.

Principes :
- Le prompt d'analyse est STATIC : il n'est jamais reconstruit à partir du XML.
- Le XML brut complet n'est JAMAIS envoyé à Gemini.
- Seules les informations techniques utiles sont extraites.
- Les doublons sont supprimés.
- Le nombre maximal de caractères envoyés au modèle est limité.
- La réponse Gemini est limitée avec max_output_tokens.
- Aucun appel client.models.list() à chaque exécution.
- Gestion spécifique des erreurs 429 / quota épuisé.

Installation :
    pip install -U google-genai python-dotenv

Utilisation :
    python ElectroXml_optimized.py schemas/schema_irrigation_logique.xml
"""

from __future__ import annotations

import os
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

# load_dotenv()
# client = genai.Client(api_key=os.getenv("API_KEY_IA"))

# print("Modèles disponibles :")
# for model in client.models.list():
#     print(model.name)
# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-3.5-flash"

# Limite de sécurité : on ne transmet jamais un XML gigantesque.
# Augmente uniquement si ton XML contient réellement beaucoup
# d'informations indispensables.
MAX_INPUT_CHARS = 18000

# Limite de sortie pour éviter qu'une analyse très longue
# consomme inutilement des crédits.
MAX_OUTPUT_TOKENS = 3000

# Température basse = réponse plus déterministe et moins bavarde.
TEMPERATURE = 0.1

SYSTEM_PROMPT = """
Tu es un ingénieur électronique senior spécialisé en logique numérique,
analyse de schémas, vérification fonctionnelle et optimisation de circuits.

Analyse uniquement les données techniques du schéma fournies dans le
message utilisateur.

RÈGLES STRICTES :
1. N'invente aucune information absente des données.
2. Si une information est inconnue, écris "Non disponible".
3. Distingue clairement les faits des hypothèses.
4. Ne prétends pas certifier la sécurité du système.
5. Sois technique, précis et concis.
6. Évite les explications générales inutiles.
7. Concentre-toi sur le circuit réellement fourni.

FORMAT DE RÉPONSE :

## 1. Fonctionnement global
Explique en quelques mots le rôle et le fonctionnement général.

## 2. Entrées / sorties
Présente les entrées et sorties importantes et leur rôle.

## 3. Chaîne logique
Explique la chaîne logique étape par étape :
entrée → porte/composant → signal intermédiaire → sortie.

## 4. Portes et composants
Liste les portes/composants importants et leur fonction.
Donne une expression booléenne uniquement si les connexions permettent
de la déduire avec certitude.

## 5. Cohérence
Identifie les problèmes :
- connexions incohérentes ;
- entrées inutilisées ;
- sorties non connectées ;
- composants orphelins ;
- signaux manquants ;
- contradictions ;
- boucles potentielles.

Classe chaque problème : CRITIQUE / ÉLEVÉ / MOYEN / FAIBLE.

## 6. Sécurité fonctionnelle
Identifie les risques visibles dans les données :
états indéfinis, absence de fail-safe, point unique de défaillance,
sortie dangereuse ou signal critique non surveillé.

Cette analyse n'est pas une certification de sécurité.

## 7. Tests
Analyse les vecteurs de test disponibles.
Signale les cas importants non couverts si cela peut être déterminé.

## 8. Optimisations
Propose uniquement des optimisations justifiées par les données :
simplification logique, réduction de portes/connexions, robustesse,
testabilité ou maintenabilité.

## 9. Conclusion
Donne :
- points forts ;
- problèmes prioritaires ;
- recommandations prioritaires ;
- niveau de confiance : ÉLEVÉ / MOYEN / FAIBLE.

Réponse maximale : environ 1200 mots.
"""


# ============================================================
# CHARGEMENT API
# ============================================================

def load_api_key() -> str:
    """Charge API_KEY_IA depuis .env."""

    load_dotenv()

    api_key = os.getenv("API_KEY_IA")

    if not api_key:
        raise RuntimeError(
            "La variable API_KEY_IA est absente.\n"
            "Ajoutez-la dans le fichier .env :\n"
            "API_KEY_IA=VOTRE_CLE_GEMINI"
        )

    return api_key


# ============================================================
# XML
# ============================================================

def load_xml(xml_path: str | Path) -> ET.Element:
    """Charge et parse le fichier XML."""

    path = Path(xml_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier XML introuvable : {path}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"Le chemin n'est pas un fichier : {path}"
        )

    try:
        return ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ET.ParseError(
            f"XML invalide : {exc}"
        ) from exc


def clean_text(value: str | None) -> str:
    """Nettoie un texte XML."""

    if not value:
        return ""

    return " ".join(value.split())


def tag_name(element: ET.Element) -> str:
    """Retourne le nom de balise sans namespace."""

    return element.tag.split("}")[-1].lower()


def element_summary(element: ET.Element) -> str:
    """
    Transforme un élément XML en représentation compacte.

    Contrairement à l'ancienne version, on ne convertit plus
    récursivement tout le XML en énorme dictionnaire JSON.
    """

    parts: list[str] = []

    # Attributs utiles
    for key, value in element.attrib.items():
        value = clean_text(value)

        if value:
            parts.append(f"{key}={value}")

    # Texte direct
    text = clean_text(element.text)

    if text:
        parts.append(f"text={text}")

    # Informations des enfants directs seulement.
    # Cela conserve le contexte sans exploser le nombre de tokens.
    for child in list(element):
        child_tag = tag_name(child)
        child_text = clean_text(child.text)

        if child_text:
            parts.append(f"{child_tag}={child_text}")

        for key, value in child.attrib.items():
            value = clean_text(value)
            if value:
                parts.append(f"{child_tag}.{key}={value}")

    return "; ".join(parts)


# ============================================================
# CLASSIFICATION DES ELEMENTS
# ============================================================

INPUT_NAMES = {
    "input", "inputs", "entrée", "entree", "entrees",
    "inputport", "input_port", "in"
}

OUTPUT_NAMES = {
    "output", "outputs", "sortie", "sorties",
    "outputport", "output_port", "out"
}

GATE_NAMES = {
    "and", "or", "not", "nand", "nor", "xor", "xnor",
    "gate", "logicgate", "logic_gate"
}

COMPONENT_NAMES = {
    "component", "components", "composant", "composants",
    "cell", "module"
}

CONNECTION_NAMES = {
    "connection", "connections", "connexion", "connexions",
    "wire", "wires", "net", "nets"
}

TEST_NAMES = {
    "testvector", "testvectors", "test_vector", "test_vectors",
    "test", "tests", "vecteur", "vecteurs",
    "vecteurtest", "vecteursdetest"
}


def extract_group(
    root: ET.Element,
    accepted_names: set[str],
    max_items: int = 100
) -> list[str]:
    """
    Extrait une liste compacte d'éléments.

    Les doublons exacts sont supprimés.
    """

    result: list[str] = []
    seen: set[str] = set()

    for element in root.iter():
        name = tag_name(element)

        if name not in accepted_names:
            continue

        summary = element_summary(element)

        if not summary:
            summary = "(élément sans détails)"

        line = f"{name}: {summary}"

        if line not in seen:
            seen.add(line)
            result.append(line)

        if len(result) >= max_items:
            break

    return result


# ============================================================
# EXTRACTION DES METADONNEES
# ============================================================

def find_metadata(
    root: ET.Element,
    accepted_names: set[str],
    max_items: int = 10
) -> list[str]:
    """Extrait quelques métadonnées sans parcourir/serialiser tout le XML."""

    values: list[str] = []
    seen: set[str] = set()

    for element in root.iter():
        name = tag_name(element)

        if name not in accepted_names:
            continue

        text = clean_text(element.text)

        if text and text not in seen:
            seen.add(text)
            values.append(text)

        for key, value in element.attrib.items():
            value = clean_text(value)

            if value and value not in seen:
                seen.add(value)
                values.append(f"{key}={value}")

        if len(values) >= max_items:
            break

    return values


def extract_schema(xml_path: str | Path) -> dict[str, Any]:
    """
    Extrait uniquement les informations utiles à l'analyse.

    IMPORTANT :
    On ne conserve plus la structure XML complète.
    C'est l'optimisation principale pour réduire les tokens.
    """

    root = load_xml(xml_path)

    metadata = {
        "titre": find_metadata(
            root,
            {"title", "titre", "name", "nom"}
        ),
        "auteur": find_metadata(
            root,
            {"author", "auteur", "creator", "createdby"}
        ),
        "date": find_metadata(
            root,
            {"date", "created", "creationdate", "modifieddate"}
        ),
        "description": find_metadata(
            root,
            {"description", "desc"}
        ),
        "specifications": find_metadata(
            root,
            {"specification", "specifications", "spec", "property"}
        ),
    }

    technical = {
        "entrees": extract_group(root, INPUT_NAMES),
        "sorties": extract_group(root, OUTPUT_NAMES),
        "portes_logiques": extract_group(root, GATE_NAMES),
        "composants": extract_group(root, COMPONENT_NAMES),
        "connexions": extract_group(root, CONNECTION_NAMES),
        "vecteurs_de_test": extract_group(root, TEST_NAMES),
    }

    return {
        "metadata": metadata,
        "technical_data": technical,
    }


# ============================================================
# DONNEES COMPACTES
# ============================================================

def build_compact_data(schema: dict[str, Any]) -> str:
    """
    Prépare les données à envoyer à Gemini.

    Ici, contrairement à l'ancienne version, il n'y a PAS de prompt
    dynamique. On ne fait que sérialiser les données utiles.
    """

    metadata = schema["metadata"]
    technical = schema["technical_data"]

    lines: list[str] = []

    lines.append("SCHEMA ELECTRONIQUE")
    lines.append("")

    # Métadonnées : uniquement si présentes
    for key, values in metadata.items():
        if values:
            lines.append(
                f"{key.upper()}: " + " | ".join(values)
            )

    lines.append("")
    lines.append("DONNEES TECHNIQUES")

    sections = [
        ("ENTREES", technical["entrees"]),
        ("SORTIES", technical["sorties"]),
        ("PORTES_LOGIQUES", technical["portes_logiques"]),
        ("COMPOSANTS", technical["composants"]),
        ("CONNEXIONS", technical["connexions"]),
        ("VECTEURS_DE_TEST", technical["vecteurs_de_test"]),
    ]

    for title, items in sections:
        lines.append("")
        lines.append(f"[{title}]")

        if not items:
            lines.append("Aucune donnée disponible.")
            continue

        for item in items:
            lines.append(f"- {item}")

    data = "\n".join(lines)

    # Sécurité supplémentaire contre les XML énormes.
    if len(data) > MAX_INPUT_CHARS:
        data = (
            data[:MAX_INPUT_CHARS]
            + "\n\n[DONNEES TRONQUEES POUR LIMITER LES TOKENS]"
        )

    return data


# ============================================================
# GEMINI
# ============================================================

def analyze_with_gemini(
    compact_data: str,
    api_key: str
) -> str:
    """
    Envoie uniquement les données compactes à Gemini.

    Le prompt d'instructions est statique via system_instruction.
    """

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=compact_data,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )

        if not response.text:
            raise RuntimeError(
                "Gemini a retourné une réponse vide."
            )

        return response.text

    except Exception as exc:
        message = str(exc)

        if "RESOURCE_EXHAUSTED" in message or "429" in message:
            raise RuntimeError(
                "Quota Gemini épuisé (HTTP 429).\n"
                "Les crédits du projet sont insuffisants."
            ) from exc

        if "NOT_FOUND" in message or "404" in message:
            raise RuntimeError(
                f"Modèle Gemini indisponible : {MODEL_NAME}"
            ) from exc

        raise RuntimeError(
            f"Erreur lors de l'appel à Gemini : {message}"
        ) from exc


# ============================================================
# AFFICHAGE
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print()


def print_analysis(xml_path: str | Path, analysis: str) -> None:
    print_header(
        f"ANALYSE : {Path(xml_path).name}"
    )
    print(analysis)
    print()
    print("=" * 80)
    print("FIN DE L'ANALYSE")
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    print_header(
        "ANALYSEUR DE SCHEMA ELECTRONIQUE XML + GEMINI"
    )

    if len(sys.argv) != 2:
        print(
            "Usage :\n"
            "    python ElectroXml_optimized.py "
            "schemas/schema_irrigation_logique.xml"
        )
        return 1

    xml_path = sys.argv[1]

    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    try:
        api_key = load_api_key()
    except RuntimeError as exc:
        print(f"[ERREUR CONFIGURATION] {exc}")
        return 1

    # --------------------------------------------------------
    # XML
    # --------------------------------------------------------

    print("[1/3] Lecture et extraction compacte du XML...")

    try:
        schema = extract_schema(xml_path)
    except FileNotFoundError as exc:
        print(f"[ERREUR FICHIER] {exc}")
        return 1
    except ET.ParseError as exc:
        print(f"[ERREUR XML] {exc}")
        return 1
    except Exception as exc:
        print(f"[ERREUR EXTRACTION] {exc}")
        return 1

    compact_data = build_compact_data(schema)

    print("      XML correctement analysé.")
    print(
        f"      Données envoyées à Gemini : "
        f"{len(compact_data):,} caractères."
    )

    # --------------------------------------------------------
    # PROMPT STATIQUE
    # --------------------------------------------------------

    print("[2/3] Préparation des données compactes...")
    print("      Prompt système statique utilisé.")
    print(
        f"      Limite entrée : {MAX_INPUT_CHARS:,} caractères."
    )
    print(
        f"      Limite sortie : {MAX_OUTPUT_TOKENS:,} tokens."
    )

    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    print(f"[3/3] Analyse par {MODEL_NAME}...")

    try:
        analysis = analyze_with_gemini(
            compact_data,
            api_key
        )
    except RuntimeError as exc:
        print(f"[ERREUR GEMINI] {exc}")
        return 1

    print_analysis(xml_path, analysis)

    return 0


if __name__ == "__main__":
    sys.exit(main())
