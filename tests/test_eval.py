from voiceforge.eval.metrics import accuracy, f1_macro, rmse, r2


def test_accuracy():
    assert accuracy([1, 2, 3], [1, 2, 3]) == 1.0
    assert 0.0 <= accuracy([1, 2, 3], [1, 2, 9]) <= 1.0


def test_f1_macro():
    v = f1_macro(["a", "b", "a"], ["a", "b", "a"])
    assert abs(v - 1.0) < 1e-9
    assert 0.0 <= f1_macro(["a", "b"], ["a", "a"]) <= 1.0


def test_rmse():
    assert rmse([0, 0, 0], [0, 0, 0]) == 0.0
    assert abs(rmse([0, 0], [1, 1]) - 1.0) < 1e-9


def test_r2():
    assert abs(r2([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-6
