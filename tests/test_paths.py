from pathlib import Path

from paddle_pyfi.paths import safe_child_path


def test_safe_child_path_strips_traversal(tmp_path: Path) -> None:
    result = safe_child_path(tmp_path, "../../evil.png")
    assert tmp_path.resolve() in result.resolve().parents
    assert result.name == "evil.png"


def test_safe_child_path_handles_nested_relative_path(tmp_path: Path) -> None:
    result = safe_child_path(tmp_path, "figures/chart 1.png")
    assert result == tmp_path.resolve() / "figures" / "chart-1.png"
