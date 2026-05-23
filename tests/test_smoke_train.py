from pathlib import Path


def test_entrypoint_placeholders_exist() -> None:
    scripts = [
        "prepare_data.py",
        "make_splits.py",
        "extract_features.py",
        "train.py",
        "evaluate.py",
        "run_experiment_matrix.py",
        "generate_report_tables.py",
    ]
    for script in scripts:
        assert Path("scripts", script).exists()
