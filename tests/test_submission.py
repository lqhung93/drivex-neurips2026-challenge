import numpy as np
import pytest

from drivex26.submission import COORD_COLUMNS, mean_per_scene, validate_submission, write_submission

FORM_NAME = "drivex26_CSV_version0.csv"  # upload form rule: <TEAM>_CSV_version<N>


def straight(n):
    pred = np.zeros((n, 12, 2))
    pred[:, :, 0] = 2.5 * np.arange(1, 13)
    return pred


@pytest.fixture
def valid(tmp_path):
    path = tmp_path / FORM_NAME
    write_submission(path, ["tokA", "tokB"], ["turn left, then stop", "go"], straight(2))
    return path


def test_written_file_validates_against_expected_tokens(valid):
    assert validate_submission(valid, expected_tokens={"tokA", "tokB"}) == ([], [])


def test_writer_uses_header_with_instruction(valid):
    assert valid.read_text().splitlines()[0] == ",".join(["sample_token", "instruction", *COORD_COLUMNS])


def test_header_without_instruction_is_accepted(tmp_path):
    path = tmp_path / FORM_NAME
    row = ",".join(["tokA", *map(str, straight(1).reshape(-1))])
    path.write_text(",".join(["sample_token", *COORD_COLUMNS]) + "\n" + row + "\n")
    assert validate_submission(path) == ([], [])


def test_wrong_filename_is_error(tmp_path):
    path = tmp_path / "preds.csv"
    write_submission(path, ["tokA"], ["go"], straight(1))
    errors, _ = validate_submission(path)
    assert any("file name" in e for e in errors)


def test_wrong_header_is_error(tmp_path):
    path = tmp_path / FORM_NAME
    path.write_text("token," + ",".join(COORD_COLUMNS) + "\n")
    errors, _ = validate_submission(path)
    assert any("header" in e for e in errors)


def test_duplicate_token_is_error(tmp_path):
    path = tmp_path / FORM_NAME
    write_submission(path, ["tokA"], ["go"], straight(1))
    path.write_text(path.read_text() + path.read_text().splitlines()[1] + "\n")
    errors, _ = validate_submission(path)
    assert any("duplicate" in e for e in errors)


def test_non_finite_value_is_error(tmp_path):
    path = tmp_path / FORM_NAME
    write_submission(path, ["tokA"], ["go"], straight(1))
    path.write_text(path.read_text().replace(",30.0", ",nan"))
    errors, _ = validate_submission(path)
    assert any("finite" in e for e in errors)


def test_missing_and_unexpected_tokens_are_errors(valid):
    errors, _ = validate_submission(valid, expected_tokens={"tokA", "tokC"})
    assert any("missing" in e and "tokC" in e for e in errors)
    assert any("unexpected" in e and "tokB" in e for e in errors)


def test_backward_trajectories_warn(tmp_path):
    path = tmp_path / FORM_NAME
    write_submission(path, ["tokA", "tokB"], ["go", "go"], -straight(2))
    errors, warnings = validate_submission(path)
    assert errors == [] and any("forward" in w for w in warnings)


def test_physically_impossible_step_warns(tmp_path):
    path = tmp_path / FORM_NAME
    pred = straight(1)
    pred[0, 6:, 0] += 100.0
    write_submission(path, ["tokA"], ["go"], pred)
    _, warnings = validate_submission(path)
    assert any("step" in w for w in warnings)


def test_writer_rejects_duplicate_tokens(tmp_path):
    with pytest.raises(ValueError):
        write_submission(tmp_path / FORM_NAME, ["tokA", "tokA"], ["a", "b"], straight(2))


def test_mean_per_scene_averages_predictions_of_same_scene():
    preds = np.stack([np.zeros((12, 2)), np.full((12, 2), 2.0), np.full((12, 2), 5.0)])
    tokens, instructions, out = mean_per_scene(
        np.array(["s1", "s1", "s2"]), np.array(["go", "stop", "left"]), preds)
    assert tokens.tolist() == ["s1", "s2"]
    assert instructions.tolist() == ["go", "left"]
    np.testing.assert_allclose(out[0], 1.0)
    np.testing.assert_allclose(out[1], 5.0)


@pytest.mark.parametrize("name", ["drivex26_CSV_version0.csv", "Team-A_CSV_version12.csv"])
def test_form_filenames_are_accepted(tmp_path, name):
    path = tmp_path / name
    write_submission(path, ["tokA"], ["go"], straight(1))
    assert validate_submission(path) == ([], [])


@pytest.mark.parametrize("name", ["submission.csv", "drivex26_CSV_versionX.csv", "drivex26_CSV_v1.csv",
                                  "_CSV_version0.csv", "drivex26_CSV_version0.txt"])
def test_other_filenames_are_rejected(tmp_path, name):
    # A wrongly named upload is not a valid entry, so the page's submission.csv is rejected too.
    path = tmp_path / name
    write_submission(path, ["tokA"], ["go"], straight(1))
    errors, _ = validate_submission(path)
    assert any("file name" in e for e in errors)


def test_team_name_must_match_when_given(tmp_path):
    path = tmp_path / "OtherTeam_CSV_version0.csv"
    write_submission(path, ["tokA"], ["go"], straight(1))
    assert validate_submission(path, team="drivex26")[0]
    assert validate_submission(path, team="OtherTeam") == ([], [])


def test_form_filename_builder():
    from drivex26.submission import form_csv_name
    assert form_csv_name("drivex26", 3) == "drivex26_CSV_version3.csv"
    with pytest.raises(ValueError):
        form_csv_name("drivex26", -1)
    with pytest.raises(ValueError):
        form_csv_name("bad team", 0)


def test_first_per_scene_keeps_first_row_of_each_scene():
    from drivex26.submission import first_per_scene
    preds = np.stack([np.zeros((12, 2)), np.full((12, 2), 2.0), np.full((12, 2), 5.0)])
    tokens, instructions, out = first_per_scene(
        np.array(["s2", "s1", "s1"]), np.array(["left", "go", "stop"]), preds)
    assert tokens.tolist() == ["s1", "s2"]
    assert instructions.tolist() == ["go", "left"]
    np.testing.assert_allclose(out[0], 2.0)
    np.testing.assert_allclose(out[1], 0.0)
