import json
import stat

from kernelthing import opencode_client


def test_build_env_seeds_auth_into_isolated_data_dir(tmp_path, monkeypatch):
    src_data = tmp_path / "xdg-data"
    auth = src_data / "opencode" / "auth.json"
    auth.parent.mkdir(parents=True)
    auth.write_text('{"openrouter":{"type":"api","key":"test"}}', encoding="utf-8")
    auth.chmod(0o600)
    monkeypatch.setenv("XDG_DATA_HOME", str(src_data))

    data_dir = tmp_path / "candidate-oc"
    env, oc_state = opencode_client.build_opencode_env(data_dir=data_dir)

    dst = data_dir / "share" / "opencode" / "auth.json"
    assert env["XDG_DATA_HOME"] == str(data_dir / "share")
    assert oc_state == [data_dir / "share", data_dir / "state", data_dir / "cache"]
    assert dst.read_text(encoding="utf-8") == auth.read_text(encoding="utf-8")
    assert stat.S_IMODE(dst.stat().st_mode) == 0o600


def test_parse_ndjson_surfaces_error_event():
    line = json.dumps(
        {
            "type": "error",
            "sessionID": "ses_123",
            "error": {
                "name": "UnknownError",
                "data": {
                    "message": "Unexpected server error. Check server logs for details.",
                    "ref": "err_abc",
                },
            },
        }
    )

    text, sid, cost, tokens, tool_calls, error = opencode_client.parse_ndjson(line)

    assert text == ""
    assert sid == "ses_123"
    assert cost == 0.0
    assert tokens == {}
    assert tool_calls == 0
    assert error == "UnknownError: Unexpected server error. Check server logs for details. (ref err_abc)"
