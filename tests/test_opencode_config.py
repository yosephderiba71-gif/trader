import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "opencode.json"


def test_opencode_config_is_valid_json():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["$schema"] == "https://opencode.ai/config.json"


def test_fin_data_mcp_is_local_and_enabled():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    server = config["mcp"]["fin-data"]
    assert server["type"] == "local"
    assert server["enabled"] is True
    assert "github:CohenD/fin-data-mcp" in server["command"]
    assert server["timeout"] >= 30_000


def test_mcp_config_has_no_secrets():
    text = CONFIG_PATH.read_text(encoding="utf-8").lower()
    for forbidden in ("api_key", "api_secret", "alpaca", "bearer", "password", "token"):
        assert forbidden not in text


def test_mcp_config_does_not_add_broker_tools():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert set(config["mcp"]) == {"fin-data"}
