import titanic


def test_package_importable():
    assert titanic.__version__ == "0.1.0"
