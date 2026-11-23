"""Hook PostToolUse: formata arquivos .py com ruff depois de editados pelo Claude.

Usa o ruff do .venv do projeto (macOS/Linux: .venv/bin, Windows: .venv\\Scripts);
se não houver, tenta o ruff do PATH. Sem ruff, não faz nada (nunca deve bloquear o trabalho).
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    data = json.load(sys.stdin)
    path = data.get("tool_input", {}).get("file_path", "")
    if path.endswith(".py"):
        venv = Path(os.environ.get("CLAUDE_PROJECT_DIR", ".")) / ".venv"
        candidates = [venv / "bin" / "ruff", venv / "Scripts" / "ruff.exe"]
        ruff = next((str(c) for c in candidates if c.exists()), shutil.which("ruff"))
        if ruff:
            subprocess.run([ruff, "format", path], check=False, capture_output=True)
except Exception:
    pass

sys.exit(0)
