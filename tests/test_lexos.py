from lexos import __version__


def test_version():
    assert __version__ == '0.0.1'

def test_initials():
    from lexos.initials import initials
    assert initials('Guide van Rossum') == 'GvR'
