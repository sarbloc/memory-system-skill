"""Tests for config resolution in client.load_config (decoupling from OpenClaw)."""

import json

import pytest

from entity_memory.client import load_config


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point HOME at a temp dir and clear config-related env vars.

    Stops the test from picking up a real ``~/.openclaw/memory.json`` or
    ``~/.config/entity-memory/config.json`` on the host running the suite.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("ENTITY_MEMORY_CONFIG", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    return tmp_path


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class TestLoadConfig:
    def test_defaults_when_nothing_set(self, isolated_home):
        cfg = load_config()
        assert cfg["qdrant"]["url"] == "http://127.0.0.1:6333"
        assert cfg["qdrant"]["api_key_env"] == "QDRANT_API_KEY"

    def test_qdrant_url_env_overrides_everything(self, isolated_home, monkeypatch):
        # Even with a config file present, QDRANT_URL wins — a deployment can
        # repoint Qdrant with zero file edits.
        _write(isolated_home / ".config" / "entity-memory" / "config.json",
               {"qdrant": {"url": "http://fromfile:6333"}})
        monkeypatch.setenv("QDRANT_URL", "http://fromenv:9999")
        assert load_config()["qdrant"]["url"] == "http://fromenv:9999"

    def test_xdg_config_path(self, isolated_home):
        _write(isolated_home / ".config" / "entity-memory" / "config.json",
               {"qdrant": {"url": "http://xdg:6333", "api_key_env": "QDRANT_API_KEY"}})
        assert load_config()["qdrant"]["url"] == "http://xdg:6333"

    def test_explicit_config_env_path(self, isolated_home, tmp_path, monkeypatch):
        cfg_file = tmp_path / "custom.json"
        cfg_file.write_text(json.dumps({"qdrant": {"url": "http://explicit:6333"}}))
        monkeypatch.setenv("ENTITY_MEMORY_CONFIG", str(cfg_file))
        assert load_config()["qdrant"]["url"] == "http://explicit:6333"

    def test_explicit_config_missing_raises(self, isolated_home, monkeypatch):
        # A set-but-missing ENTITY_MEMORY_CONFIG must error, not silently fall
        # back to XDG/legacy and connect to a Qdrant the operator didn't pick.
        _write(isolated_home / ".openclaw" / "memory.json",
               {"qdrant": {"url": "http://legacy:6333"}})
        monkeypatch.setenv("ENTITY_MEMORY_CONFIG", str(isolated_home / "nope.json"))
        with pytest.raises(FileNotFoundError):
            load_config()

    def test_qdrant_url_rescues_missing_explicit_config(self, isolated_home, monkeypatch):
        # QDRANT_URL is highest precedence: a missing ENTITY_MEMORY_CONFIG must
        # NOT abort when the env URL already supplies what we need.
        monkeypatch.setenv("ENTITY_MEMORY_CONFIG", str(isolated_home / "nope.json"))
        monkeypatch.setenv("QDRANT_URL", "http://fromenv:9999")
        assert load_config()["qdrant"]["url"] == "http://fromenv:9999"

    def test_legacy_openclaw_path_still_read(self, isolated_home):
        # Back-compat: existing OpenClaw/Endurance installs keep resolving.
        _write(isolated_home / ".openclaw" / "memory.json",
               {"qdrant": {"url": "http://legacy:6333"}})
        assert load_config()["qdrant"]["url"] == "http://legacy:6333"

    def test_xdg_takes_precedence_over_legacy(self, isolated_home):
        _write(isolated_home / ".config" / "entity-memory" / "config.json",
               {"qdrant": {"url": "http://xdg:6333"}})
        _write(isolated_home / ".openclaw" / "memory.json",
               {"qdrant": {"url": "http://legacy:6333"}})
        assert load_config()["qdrant"]["url"] == "http://xdg:6333"

    def test_explicit_env_path_precedes_xdg(self, isolated_home, tmp_path, monkeypatch):
        cfg_file = tmp_path / "custom.json"
        cfg_file.write_text(json.dumps({"qdrant": {"url": "http://explicit:6333"}}))
        monkeypatch.setenv("ENTITY_MEMORY_CONFIG", str(cfg_file))
        _write(isolated_home / ".config" / "entity-memory" / "config.json",
               {"qdrant": {"url": "http://xdg:6333"}})
        assert load_config()["qdrant"]["url"] == "http://explicit:6333"

    def test_xdg_config_home_is_honored(self, isolated_home, tmp_path, monkeypatch):
        # When XDG_CONFIG_HOME points elsewhere, the config is read from there,
        # not from ~/.config.
        xdg_dir = tmp_path / "custom_xdg"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_dir))
        _write(xdg_dir / "entity-memory" / "config.json",
               {"qdrant": {"url": "http://xdghome:6333"}})
        # A file at the default ~/.config location must be ignored in favor of XDG.
        _write(isolated_home / ".config" / "entity-memory" / "config.json",
               {"qdrant": {"url": "http://default:6333"}})
        assert load_config()["qdrant"]["url"] == "http://xdghome:6333"

    def test_relative_xdg_config_home_ignored(self, isolated_home, monkeypatch):
        # The XDG spec says a relative $XDG_CONFIG_HOME is invalid and must be
        # ignored — we fall back to ~/.config rather than resolving it against cwd.
        monkeypatch.setenv("XDG_CONFIG_HOME", "relative/path")
        _write(isolated_home / ".config" / "entity-memory" / "config.json",
               {"qdrant": {"url": "http://default:6333"}})
        assert load_config()["qdrant"]["url"] == "http://default:6333"
