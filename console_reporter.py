"""
ConsoleReporter - Affichage terminal des etapes et du resultat.
"""

from __future__ import annotations

from pathlib import Path


class ConsoleReporter:
    """Affiche les etapes et le resultat dans le terminal."""

    @staticmethod
    def header(title: str) -> None:
        print()
        print("=" * 80)
        print(title)
        print("=" * 80)
        print()

    @staticmethod
    def result(xml_path: str | Path, description: str) -> None:
        ConsoleReporter.header(f"DESCRIPTION : {Path(xml_path).name}")
        print(description)
        print()
        print("=" * 80)
        print("FIN")
        print("=" * 80)
