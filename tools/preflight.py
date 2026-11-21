"""Read-only pre-commit scan; no files are modified."""
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[1]
tracked = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=False)
paths = [root / b.decode("utf-8", "replace") for b in tracked.stdout.split(b"\0") if b]
secret = [re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), re.compile(r"\bgh[puosr]_[A-Za-z0-9_]{20,}\b"), re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")]
banned = {".env", ".env.local", ".env.production"}
suffixes = {".sqlite", ".sqlite3", ".db", ".pem", ".key", ".pdf", ".docx", ".pt", ".pth", ".safetensors"}
problems = []
for path in paths:
    if path.name in banned or path.suffix.lower() in suffixes:
        problems.append(f"Review tracked file: {path.relative_to(root)}")
    if path.is_file() and path.stat().st_size < 2_000_000:
        content = path.read_text("utf-8", errors="ignore")
        if any(p.search(content) for p in secret):
            problems.append(f"Possible secret: {path.relative_to(root)}")
print(f"Checked {len(paths)} tracked files")
print("\n".join(problems) if problems else "No obvious sensitive files found; review manually too.")
raise SystemExit(bool(problems))
