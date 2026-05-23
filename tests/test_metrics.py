from src.evaluation.statistical import bootstrap_ci, mcnemar_test


def test_statistical_functions_import() -> None:
    assert callable(mcnemar_test)
    assert callable(bootstrap_ci)
