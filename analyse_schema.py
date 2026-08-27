
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

from config import Config
from console_reporter import ConsoleReporter
from gemini_describer import GeminiDescriber
from schema_extractor import SchemaExtractor


def main() -> int:
    ConsoleReporter.header("DESCRIPTION DE SCHEMA ELECTRONIQUE XML + GEMINI")

    if len(sys.argv) != 2:
        print("Usage :\n    python analyse_schema.py schema.xml")
        return 1

    xml_path = sys.argv[1]

    print("[1/2] Lecture et extraction compacte du XML...")
    try:
        extractor = SchemaExtractor(xml_path)
        compact_data = extractor.build_compact_data()
    except FileNotFoundError as exc:
        print(f"[ERREUR FICHIER] {exc}")
        return 1
    except ET.ParseError as exc:
        print(f"[ERREUR XML] {exc}")
        return 1
    except Exception as exc:
        print(f"[ERREUR EXTRACTION] {exc}")
        return 1

    print(f"      Donnees envoyees a Gemini : {len(compact_data):,} caracteres.")

    print(f"[2/2] Description par {Config.MODEL_NAME}...")
    try:
        describer = GeminiDescriber()
        description = describer.describe(compact_data)
    except RuntimeError as exc:
        print(f"[ERREUR] {exc}")
        return 1

    ConsoleReporter.result(xml_path, description)
    return 0


if __name__ == "__main__":
    sys.exit(main())
