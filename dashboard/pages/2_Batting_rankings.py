"""Ranking page, layout shared with the other role."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rankings_page import render

render("bat")
