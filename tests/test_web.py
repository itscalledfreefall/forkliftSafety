"""Tests for SafetyVision Web UI API."""

import pytest
import yaml

# These tests require fastapi + httpx
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from httpx import ASGITransport, AsyncClient
from safetyvision.web.app import app, SESSION_TOKENS
import safetyvision.web.app as web_app


@pytest.fixture(autouse=True)
def _use_tmp_config(tmp_path, monkeypatch):
    """Point web app at a temp config for each test."""
    cfg = {
        "input": {
            "cameras": [
                {
                    "id": "back",
                    "rtsp_url": "rtsp://cam:554/sub",
                    "rtsp_url_main": "rtsp://cam:554/main",
                    "mode": "zone",
                    "zone": {
                        "yellow_start_y": 0.34,
                        "red_start_y": 0.68,
                    },
                    "distance": {
                        "warning_distance_m": 2.0,
                        "danger_distance_m": 1.0,
                        "calibration_path": "config/calibration/back.yaml",
                    },
                }
            ],
            "width": 640,
            "height": 480,
            "target_fps": 15,
        },
        "model": {
            "runtime": "hailo",
            "path_hef": "/usr/share/hailo-models/yolov6n_h8l.hef",
            "conf_threshold": 0.45,
            "iou_threshold": 0.50,
        },
        "alert": {
            "yellow_start_y": 0.33,
            "red_start_y": 0.66,
            "min_alert_confidence": 0.55,
            "repeat_interval_sec": 1.5,
            "min_clear_sec": 3.0,
        },
        "perf": {"max_queue_size": 1},
    }
    p = tmp_path / "test_config.yaml"
    p.write_text(yaml.dump(cfg))
    monkeypatch.setattr(web_app, "CONFIG_PATH", str(p))
    SESSION_TOKENS.clear()
    yield


@pytest.fixture
def auth_cookies():
    """Create a valid session token and return cookies dict."""
    import secrets, time
    token = secrets.token_urlsafe(32)
    SESSION_TOKENS[token] = time.time() + 3600
    return {"sv_session": token}


@pytest.mark.asyncio
async def test_login_success():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/auth/login", json={
            "username": "admin",
            "password": "safetyvision",
        })
        assert res.status_code == 200
        assert "sv_session" in res.cookies


@pytest.mark.asyncio
async def test_login_failure():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/auth/login", json={
            "username": "admin",
            "password": "wrong",
        })
        assert res.status_code == 401


@pytest.mark.asyncio
async def test_auth_required():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.get("/api/config")
        assert res.status_code == 401


@pytest.mark.asyncio
async def test_get_config(auth_cookies):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", cookies=auth_cookies) as ac:
        res = await ac.get("/api/config")
        assert res.status_code == 200
        data = res.json()
        assert data["alert"]["yellow_start_y"] == 0.33
        assert data["alert"]["red_start_y"] == 0.66


@pytest.mark.asyncio
async def test_get_camera_configs(auth_cookies):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", cookies=auth_cookies) as ac:
        res = await ac.get("/api/config/cameras")
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 1
        assert data[0]["id"] == "back"
        assert data[0]["mode"] == "zone"
        assert data[0]["effective_zone"]["yellow_start_y"] == 0.34
        assert data[0]["distance"]["calibration_path"] == "config/calibration/back.yaml"


@pytest.mark.asyncio
async def test_save_zones_valid(auth_cookies):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", cookies=auth_cookies) as ac:
        res = await ac.post("/api/config/zones", json={
            "yellow_start_y": 0.40,
            "red_start_y": 0.75,
        })
        assert res.status_code == 200
        assert res.json()["ok"]

        # Verify persisted
        res2 = await ac.get("/api/config")
        assert res2.json()["alert"]["yellow_start_y"] == 0.40
        assert res2.json()["alert"]["red_start_y"] == 0.75


@pytest.mark.asyncio
async def test_save_zones_invalid_yellow_gte_red(auth_cookies):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", cookies=auth_cookies) as ac:
        res = await ac.post("/api/config/zones", json={
            "yellow_start_y": 0.80,
            "red_start_y": 0.50,
        })
        assert res.status_code == 400


@pytest.mark.asyncio
async def test_save_zones_out_of_range(auth_cookies):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", cookies=auth_cookies) as ac:
        res = await ac.post("/api/config/zones", json={
            "yellow_start_y": 0.0,
            "red_start_y": 1.0,
        })
        assert res.status_code == 400


@pytest.mark.asyncio
async def test_save_timing_valid(auth_cookies):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", cookies=auth_cookies) as ac:
        res = await ac.post("/api/config/timing", json={
            "repeat_interval_sec": 2.0,
            "min_clear_sec": 5.0,
            "min_alert_confidence": 0.65,
        })
        assert res.status_code == 200


@pytest.mark.asyncio
async def test_save_timing_invalid(auth_cookies):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", cookies=auth_cookies) as ac:
        res = await ac.post("/api/config/timing", json={
            "repeat_interval_sec": 0,
            "min_clear_sec": 3.0,
            "min_alert_confidence": 0.55,
        })
        assert res.status_code == 400


@pytest.mark.asyncio
async def test_save_camera_config(auth_cookies):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", cookies=auth_cookies) as ac:
        res = await ac.post("/api/config/cameras/back", json={
            "mode": "distance",
            "zone": {
                "yellow_start_y": 0.22,
                "red_start_y": 0.60,
            },
            "distance": {
                "warning_distance_m": 3.0,
                "danger_distance_m": 1.5,
                "calibration_path": "config/calibration/rear.yaml",
            },
        })
        assert res.status_code == 200
        payload = res.json()
        assert payload["ok"] is True
        assert payload["camera"]["mode"] == "distance"

        res2 = await ac.get("/api/config")
        saved = res2.json()["input"]["cameras"][0]
        assert saved["mode"] == "distance"
        assert saved["zone"]["yellow_start_y"] == 0.22
        assert saved["distance"]["warning_distance_m"] == 3.0
        assert saved["distance"]["calibration_path"] == "config/calibration/rear.yaml"


@pytest.mark.asyncio
async def test_save_camera_config_invalid_distance_thresholds(auth_cookies):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", cookies=auth_cookies) as ac:
        res = await ac.post("/api/config/cameras/back", json={
            "mode": "distance",
            "distance": {
                "warning_distance_m": 1.0,
                "danger_distance_m": 1.0,
                "calibration_path": "config/calibration/rear.yaml",
            },
        })
        assert res.status_code == 400


@pytest.mark.asyncio
async def test_validate_config(auth_cookies):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", cookies=auth_cookies) as ac:
        res = await ac.post("/api/config/validate")
        assert res.status_code == 200
        assert res.json()["valid"]


@pytest.mark.asyncio
async def test_restore_no_backup(auth_cookies):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", cookies=auth_cookies) as ac:
        res = await ac.post("/api/config/restore")
        assert res.status_code == 404


@pytest.mark.asyncio
async def test_logout(auth_cookies):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", cookies=auth_cookies) as ac:
        res = await ac.post("/api/auth/logout")
        assert res.status_code == 200

        # Should be unauthenticated now
        res2 = await ac.get("/api/config")
        assert res2.status_code == 401
