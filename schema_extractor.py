"""
SchemaExtractor - Extraction compacte d'un schema electronique XML.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from config import Config


class SchemaExtractor:
    """
    Charge un fichier XML de schema electronique et en extrait une
    representation compacte, sans jamais serialiser le XML complet.
    """

    INPUT_NAMES = {
        "input", "inputs", "entree", "entrees",
        "inputport", "input_port", "in",
    }
    OUTPUT_NAMES = {
        "output", "outputs", "sortie", "sorties",
        "outputport", "output_port", "out",
    }
    GATE_NAMES = {
        "and", "or", "not", "nand", "nor", "xor", "xnor",
        "gate", "logicgate", "logic_gate",
    }
    COMPONENT_NAMES = {
        "component", "components", "composant", "composants",
        "cell", "module",
    }
    CONNECTION_NAMES = {
        "connection", "connections", "connexion", "connexions",
        "wire", "wires", "net", "nets",
    }

    def __init__(
        self,
        xml_path: str | Path,
        max_input_chars: int = Config.MAX_INPUT_CHARS,
    ) -> None:
        self.xml_path = Path(xml_path)
        self.max_input_chars = max_input_chars

    # ---- chargement / helpers ----

    def _load_xml(self) -> ET.Element:
        if not self.xml_path.exists():
            raise FileNotFoundError(f"Fichier XML introuvable : {self.xml_path}")
        if not self.xml_path.is_file():
            raise FileNotFoundError(f"Le chemin n'est pas un fichier : {self.xml_path}")
        try:
            return ET.parse(self.xml_path).getroot()
        except ET.ParseError as exc:
            raise ET.ParseError(f"XML invalide : {exc}") from exc

    @staticmethod
    def _clean_text(value: str | None) -> str:
        return " ".join(value.split()) if value else ""

    @staticmethod
    def _tag_name(element: ET.Element) -> str:
        return element.tag.split("}")[-1].lower()

    def _element_summary(self, element: ET.Element) -> str:
        parts: list[str] = []

        for key, value in element.attrib.items():
            value = self._clean_text(value)
            if value:
                parts.append(f"{key}={value}")

        text = self._clean_text(element.text)
        if text:
            parts.append(f"text={text}")

        # Infos des enfants directs seulement : garde le contexte sans
        # exploser le nombre de tokens.
        for child in list(element):
            child_tag = self._tag_name(child)
            child_text = self._clean_text(child.text)
            if child_text:
                parts.append(f"{child_tag}={child_text}")
            for key, value in child.attrib.items():
                value = self._clean_text(value)
                if value:
                    parts.append(f"{child_tag}.{key}={value}")

        return "; ".join(parts)

    def _extract_group(
        self,
        root: ET.Element,
        accepted_names: set[str],
        max_items: int = 100,
    ) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        for element in root.iter():
            name = self._tag_name(element)
            if name not in accepted_names:
                continue

            summary = self._element_summary(element) or "(element sans details)"
            line = f"{name}: {summary}"

            if line not in seen:
                seen.add(line)
                result.append(line)
            if len(result) >= max_items:
                break

        return result

    def _find_metadata(
        self,
        root: ET.Element,
        accepted_names: set[str],
        max_items: int = 10,
    ) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()

        for element in root.iter():
            if self._tag_name(element) not in accepted_names:
                continue

            text = self._clean_text(element.text)
            if text and text not in seen:
                seen.add(text)
                values.append(text)

            for key, value in element.attrib.items():
                value = self._clean_text(value)
                if value and value not in seen:
                    seen.add(value)
                    values.append(f"{key}={value}")

            if len(values) >= max_items:
                break

        return values

    # ---- API publique ----

    def extract(self) -> dict[str, Any]:
        """Extrait uniquement les informations utiles a la description."""
        root = self._load_xml()

        metadata = {
            "titre": self._find_metadata(root, {"title", "titre", "name", "nom"}),
            "description": self._find_metadata(root, {"description", "desc"}),
            "specifications": self._find_metadata(
                root, {"specification", "specifications", "spec", "property"}
            ),
        }

        technical = {
            "entrees": self._extract_group(root, self.INPUT_NAMES),
            "sorties": self._extract_group(root, self.OUTPUT_NAMES),
            "portes_logiques": self._extract_group(root, self.GATE_NAMES),
            "composants": self._extract_group(root, self.COMPONENT_NAMES),
            "connexions": self._extract_group(root, self.CONNECTION_NAMES),
        }

        return {"metadata": metadata, "technical_data": technical}

    def build_compact_data(self) -> str:
        """Serialise les donnees extraites en texte compact pour Gemini."""
        schema = self.extract()
        metadata = schema["metadata"]
        technical = schema["technical_data"]

        lines: list[str] = ["SCHEMA ELECTRONIQUE", ""]

        for key, values in metadata.items():
            if values:
                lines.append(f"{key.upper()}: " + " | ".join(values))

        lines.append("")
        lines.append("DONNEES TECHNIQUES")

        sections = [
            ("ENTREES", technical["entrees"]),
            ("SORTIES", technical["sorties"]),
            ("PORTES_LOGIQUES", technical["portes_logiques"]),
            ("COMPOSANTS", technical["composants"]),
            ("CONNEXIONS", technical["connexions"]),
        ]

        for title, items in sections:
            lines.append("")
            lines.append(f"[{title}]")
            if not items:
                lines.append("Aucune donnee disponible.")
                continue
            for item in items:
                lines.append(f"- {item}")

        data = "\n".join(lines)

        if len(data) > self.max_input_chars:
            data = (
                data[: self.max_input_chars]
                + "\n\n[DONNEES TRONQUEES POUR LIMITER LES TOKENS]"
            )

        return data
