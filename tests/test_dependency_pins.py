from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _requirement_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_all_direct_runtime_dependencies_are_exactly_pinned():
    lines = _requirement_lines(ROOT / "requirements.txt")
    specifications = [line.split(";", 1)[0].strip() for line in lines]

    assert lines
    assert all("==" in specification for specification in specifications)
    assert not any(
        ">=" in specification or "~=" in specification
        for specification in specifications
    )


def test_development_dependencies_are_exactly_pinned():
    lines = _requirement_lines(ROOT / "requirements-dev.txt")
    direct = [line for line in lines if not line.startswith("-r ")]

    assert "-r requirements.txt" in lines
    assert direct
    assert all("==" in line for line in direct)
