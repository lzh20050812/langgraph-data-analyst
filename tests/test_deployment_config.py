from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _compose_services() -> dict:
    compose_path = PROJECT_ROOT / "docker" / "docker-compose.yml"
    return yaml.safe_load(compose_path.read_text(encoding="utf-8"))["services"]


def test_compose_injects_env_without_mounting_secret_file():
    services = _compose_services()

    for name in ("db-init", "app", "worker"):
        service = services[name]
        assert "../.env" in service["env_file"]
        assert all("/app/.env" not in str(volume) for volume in service.get("volumes", []))


def test_compose_api_and_worker_share_linux_task_volume():
    services = _compose_services()
    task_mount = "task_runtime_data:/app/data/runtime"

    assert task_mount in services["app"]["volumes"]
    assert task_mount in services["worker"]["volumes"]
    assert task_mount not in services["db-init"].get("volumes", [])
