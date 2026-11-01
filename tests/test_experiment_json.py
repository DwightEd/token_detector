"""Experiment metadata preserves Unicode across platform default encodings."""

from experiments import data


def test_json_metadata_round_trip_preserves_unicode(tmp_path):
    path = tmp_path / 'metadata.json'
    expected = {'\u7ed3\u679c': '\u7814\u7a76', 'status': '\u2713'}
    data.write_json(path, expected)
    assert data.read_json(path) == expected
