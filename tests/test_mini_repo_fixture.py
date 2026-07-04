from pathlib import Path

MINI_REPO_SOURCE = Path(__file__).parent / "fixtures" / "mini_repo"


def _relative_files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def _relative_dirs(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()}


def test_mini_repo_source__matches_file_manifest(mini_repo_manifest) -> None:
    assert _relative_files(MINI_REPO_SOURCE) == mini_repo_manifest.files


def test_mini_repo_source__matches_dir_manifest(mini_repo_manifest) -> None:
    assert _relative_dirs(MINI_REPO_SOURCE) == mini_repo_manifest.dirs


def test_long_file__has_pinned_fixture_lines(mini_repo_manifest) -> None:
    lines = (MINI_REPO_SOURCE / "data" / "long_file.txt").read_text(encoding="utf-8").splitlines()

    assert len(lines) == mini_repo_manifest.long_file_lines
    assert lines[0] == "line 001"
    assert lines[-1] == "line 450"


def test_text_files__use_lf_without_cr_bytes(mini_repo_manifest) -> None:
    text_files = mini_repo_manifest.files - {mini_repo_manifest.binary_file}

    for relative_path in text_files:
        assert b"\r" not in (MINI_REPO_SOURCE / relative_path).read_bytes()


def test_logo_bin__is_small_binary_with_null_byte(mini_repo_manifest) -> None:
    content = (MINI_REPO_SOURCE / mini_repo_manifest.binary_file).read_bytes()

    assert len(content) < 64
    assert b"\x00" in content[:1024]


def test_parse_date_definition__is_unique_and_pinned(mini_repo_manifest) -> None:
    matches: list[tuple[str, int]] = []
    text_files = mini_repo_manifest.files - {mini_repo_manifest.binary_file}

    for relative_path in sorted(text_files):
        text = (MINI_REPO_SOURCE / relative_path).read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if "def parse_date(" in line:
                matches.append((relative_path, line_number))

    assert matches == [(mini_repo_manifest.parse_date_file, mini_repo_manifest.parse_date_line)]


def test_parse_date_mentions__stay_in_pinned_files(mini_repo_manifest) -> None:
    text_files = mini_repo_manifest.files - {mini_repo_manifest.binary_file}
    mention_files = frozenset(
        relative_path
        for relative_path in text_files
        if "parse_date" in (MINI_REPO_SOURCE / relative_path).read_text(encoding="utf-8")
    )

    assert mention_files == mini_repo_manifest.parse_date_mention_files


def test_hidden_entries__exist(mini_repo_manifest) -> None:
    for relative_path in mini_repo_manifest.hidden_entries:
        assert (MINI_REPO_SOURCE / relative_path).exists()


def test_unicode_probe__decodes_as_utf8(mini_repo_manifest) -> None:
    text = (MINI_REPO_SOURCE / mini_repo_manifest.unicode_file).read_text(encoding="utf-8")

    assert mini_repo_manifest.unicode_probe in text


def test_mini_repo_fixture__returns_tmp_path_copy(mini_repo: Path, tmp_path: Path) -> None:
    assert mini_repo == tmp_path / "mini_repo"
    assert mini_repo.is_dir()
    assert mini_repo != MINI_REPO_SOURCE
    assert mini_repo.is_relative_to(tmp_path)
    assert _relative_files(mini_repo) == _relative_files(MINI_REPO_SOURCE)
