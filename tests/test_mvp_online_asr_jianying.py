import requests

from integrations import online_asr


def test_jianying_sign_service_failure_falls_back_to_legacy_static_sign(monkeypatch) -> None:
    def fail_post(*_args, **_kwargs):
        raise requests.RequestException("sign service down")

    monkeypatch.setattr(online_asr.requests, "post", fail_post)

    asr = online_asr.JianYingASR(b"fake mp3")
    sign, device_time = asr._get_sign("/lv/v1/upload_sign", appvr="1.4.4")
    headers = asr._build_headers(device_time, sign)

    assert sign == online_asr.JianYingASR.LEGACY_STATIC_SIGN
    assert device_time == online_asr.JianYingASR.LEGACY_STATIC_DEVICE_TIME
    assert asr._tdid == online_asr.JianYingASR.LEGACY_STATIC_TDID
    assert headers["appvr"] == "1.4.4"
    assert headers["tdid"] == online_asr.JianYingASR.LEGACY_STATIC_TDID


def test_jianying_upload_auth_uses_cn_north_1_region(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_signature(_secret_key, _params, _headers, region="cn", service="vod"):
        captured["region"] = region
        captured["service"] = service
        return "signature"

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "Result": {
                    "UploadAddress": {
                        "StoreInfos": [
                            {
                                "StoreUri": "tos-cn-v-22574f/test",
                                "Auth": "auth",
                                "UploadID": "upload-id",
                            }
                        ],
                        "SessionKey": "session-key",
                        "UploadHosts": ["tos-lf-x.snssdk.com"],
                    }
                }
            }

    def fake_get(url, headers, timeout=None):
        captured["url"] = url
        captured["authorization"] = headers["authorization"]
        return FakeResponse()

    monkeypatch.setattr(online_asr, "_aws_signature", fake_signature)
    monkeypatch.setattr(online_asr.requests, "get", fake_get)

    asr = online_asr.JianYingASR(b"fake mp3")
    asr._access_key = "ak"
    asr._secret_key = "sk"
    asr._session_token = "token"
    asr._upload_auth()

    assert captured["region"] == "cn-north-1"
    assert captured["service"] == "vod"
    assert "/cn-north-1/vod/aws4_request" in captured["authorization"]
    assert asr._store_uri == "tos-cn-v-22574f/test"


def test_jianying_upload_commit_keeps_store_uri_when_tos_commit_fails(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self):
            raise requests.HTTPError("400 MismatchChecksum")

    def fake_put(*_args, **_kwargs):
        return FakeResponse()

    monkeypatch.setattr(online_asr.requests, "put", fake_put)

    asr = online_asr.JianYingASR(b"fake mp3")
    asr._upload_hosts = "tos-lf-x.snssdk.com"
    asr._store_uri = "tos-cn-v-22574f/test"
    asr._upload_id = "upload-id"
    asr._session_token = "token"
    asr._auth = "auth"

    assert asr._upload_commit() == "tos-cn-v-22574f/test"
