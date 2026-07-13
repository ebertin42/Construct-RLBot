def test_engine_importable():
    import construct
    assert construct.version() == "0.1.0"
