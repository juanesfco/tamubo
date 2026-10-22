from pathlib import Path


def test_public_source_has_no_prohibited_downstream_token():
    root = Path(__file__).resolve().parents[1]
    target = ("cer" + "topt").encode("ascii")
    text_exts = {".py", ".md", ".txt", ".tex", ".toml", ".cfg", ".ini", ".json", ".rst"}
    hits = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if target in rel.lower().encode("utf-8", errors="ignore"):
            hits.append(rel + " [name]")
        elif path.suffix.lower() in text_exts and target in path.read_bytes().lower():
            hits.append(rel)
    assert hits == []
