"""
============================================================
在线 ASR 接口 — 独立版（零项目依赖，可直接拷贝使用）
============================================================

提供多个免费的在线语音识别接口：

  接口          │ 类名          │ 提供商       │ 是否需要外部服务
  ──────────────┼───────────────┼──────────────┼─────────────────
  B 接口 (必剪) │ BcutASR       │ Bilibili     │ 不需要，直接可用
  J 接口 (剪映) │ JianYingASR   │ 字节跳动     │ 需签名服务（见下方说明）
  K 接口 (快手) │ KuaiShouASR   │ 快手         │ 不需要，但当前接口返回禁用

依赖（仅一个）:
    pip install requests

============================
快速开始
============================

  from online_asr import BcutASR, JianYingASR

  # ----- 必剪 ASR（推荐，完全独立）-----
  asr = BcutASR("audio.mp3")
  result = asr.run()                # -> ASRData
  print(result.to_srt())            # 导出 SRT 字符串
  result.save("output.srt")         # 保存到文件

  # ----- 剪映 ASR -----
  asr = JianYingASR("audio.mp3")
  result = asr.run()
  result.save("output.json", fmt="json")

============================
完整使用示例
============================

1. 基本识别 + 保存 SRT 文件:
    from online_asr import bcut_transcribe
    result = bcut_transcribe("audio.mp3")
    result.save("output.srt")

2. 遍历每个字幕片段:
    asr = BcutASR("audio.mp3")
    result = asr.run()
    for seg in result:
        print(f"[{seg.to_srt_ts()}] {seg.text}")
        # seg.start_time / seg.end_time 单位是毫秒

3. 词级时间戳（每个词一个时间戳，而非每句）:
    asr = BcutASR("audio.mp3", need_word_time_stamp=True)
    result = asr.run()

4. 进度回调（显示上传/识别进度）:
    def on_progress(percent, msg):
        print(f"[{percent}%] {msg}")

    asr = BcutASR("audio.mp3")
    result = asr.run(callback=on_progress)

5. 直接传音频 bytes（不从文件读取）:
    with open("audio.mp3", "rb") as f:
        audio_bytes = f.read()
    result = BcutASR(audio_bytes).run()

6. 导出其他格式:
    result.to_srt()     # SRT 字幕字符串
    result.to_json()    # dict
    result.to_txt()     # 纯文本，一行一句
    result.save("out.srt", fmt="srt")
    result.save("out.json", fmt="json")
    result.save("out.txt", fmt="txt")

============================
两个接口的区别
============================

  BcutASR (必剪):
    - 优点: 完全独立，不需要任何外部签名服务，开箱即用
    - 缺点: 偶尔不稳定，B站可能改接口
    - 速度: 中等，需轮询等待（通常几秒到几十秒）

  JianYingASR (剪映):
    - 优点: 识别准确率通常更高，字节跳动内部服务
    - 缺点: 依赖第三方签名服务 asrtools-update.bkfeng.top
            如果该服务挂了则不可用，需要自建签名服务替换
    - 注意: SIGNSERVICE_URL 为类属性，可自行替换:
            JianYingASR.SIGN_SERVICE_URL = "https://你的签名服务/sign"
    - 速度: 较快，提交后直接查询结果

============================
命令行
============================

  python online_asr.py audio.mp3            # 默认必剪
  python online_asr.py audio.mp3 bcut       # 必剪
  python online_asr.py audio.mp3 jianying   # 剪映
  python online_asr.py audio.mp3 kuaishou   # 快手
  运行后在同目录下生成 audio.{引擎名}.srt
"""

import datetime
import hashlib
import hmac
import json
import os
import time
import uuid
import zlib
from enum import Enum
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import requests


# ============================================================================
# ASRStatus
# ============================================================================

class ASRStatus(Enum):
    INITIALIZING = ("initializing", 0)
    UPLOADING = ("uploading", 10)
    CREATING_TASK = ("creating_task", 40)
    SUBMITTING = ("submitting", 50)
    TRANSCRIBING = ("transcribing", 60)
    QUERYING_RESULT = ("querying_result", 80)
    COMPLETED = ("completed", 100)

    @property
    def message(self) -> str:
        return self.value[0]

    @property
    def progress(self) -> int:
        return self.value[1]

    def callback_tuple(self) -> Tuple[int, str]:
        return (self.progress, self.message)


# ============================================================================
# ASRDataSeg - 字幕片段
# ============================================================================

class ASRDataSeg:
    """单个字幕片段

    Attributes:
        text:       字幕文本
        start_time: 开始时间，单位毫秒 (int)
        end_time:   结束时间，单位毫秒 (int)

    Example:
        seg = ASRDataSeg("你好", 0, 1500)
        print(seg.text)          # "你好"
        print(seg.start_time)    # 0
        print(seg.end_time)      # 1500
        print(seg.to_srt_ts())   # "00:00:00,000 --> 00:00:01,500"
        print(seg.to_dict())     # {"text": "你好", "start_time": 0, "end_time": 1500}
    """
    def __init__(self, text: str, start_time: int, end_time: int):
        self.text = text
        self.start_time = start_time  # 毫秒
        self.end_time = end_time      # 毫秒

    def __repr__(self):
        return f"ASRDataSeg(text={self.text!r}, start={self.start_time}, end={self.end_time})"

    def to_srt_ts(self) -> str:
        """转为 SRT 时间戳格式 HH:MM:SS,mmm"""
        return f"{self._ms_to_srt(self.start_time)} --> {self._ms_to_srt(self.end_time)}"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    @staticmethod
    def _ms_to_srt(ms: int) -> str:
        total_sec, millis = divmod(ms, 1000)
        minutes, seconds = divmod(total_sec, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


# ============================================================================
# ASRData - 字幕数据集合
# ============================================================================

class ASRData:
    """字幕数据集合，asr.run() 的返回值。

    可迭代、可索引、可取 len()。

    Example:
        result = BcutASR("audio.mp3").run()

        # 遍历
        for seg in result:
            print(seg.text, seg.start_time, seg.end_time)

        # 导出
        srt_str = result.to_srt()                    # SRT 格式字符串
        json_dict = result.to_json()                 # dict
        txt_str = result.to_txt()                    # 纯文本，一行一句

        # 保存
        result.save("output.srt")                    # 默认 SRT
        result.save("output.json", fmt="json")
        result.save("output.txt", fmt="txt")

        # 基本属性
        len(result)          # 片段数量
        result.has_data()    # 是否有内容 (bool)
        result[0]            # 第一个 ASRDataSeg
    """
    def __init__(self, segments: List[ASRDataSeg]):
        self.segments = [s for s in segments if s.text and s.text.strip()]
        self.segments.sort(key=lambda x: x.start_time)

    def __iter__(self):
        return iter(self.segments)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        return self.segments[index]

    def has_data(self) -> bool:
        return len(self.segments) > 0

    def to_srt(self) -> str:
        """导出为 SRT 格式"""
        lines = []
        for i, seg in enumerate(self.segments, 1):
            lines.append(f"{i}\n{seg.to_srt_ts()}\n{seg.text}\n")
        return "\n".join(lines)

    def to_json(self) -> dict:
        """导出为 JSON 格式"""
        return {
            str(i): seg.to_dict()
            for i, seg in enumerate(self.segments, 1)
        }

    def to_txt(self) -> str:
        """导出为纯文本，一行一句"""
        return "\n".join(seg.text for seg in self.segments)

    def save(self, path: str, fmt: str = "srt"):
        """保存到文件

        Args:
            path: 输出路径
            fmt:  格式 — "srt" / "json" / "txt"
        """
        if fmt == "srt":
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.to_srt())
        elif fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_json(), f, ensure_ascii=False, indent=2)
        elif fmt == "txt":
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(seg.text for seg in self.segments))
        else:
            raise ValueError(f"不支持的格式: {fmt}")


# ============================================================================
# BaseASR - 基类
# ============================================================================

class BaseASR:
    """ASR 基类，提供音频加载和流程模板"""

    SUPPORTED_FORMATS = ["flac", "m4a", "mp3", "wav"]

    def __init__(self, audio_input: Union[str, bytes]):
        self.audio_input = audio_input
        self.file_binary: Optional[bytes] = None
        self._load_audio()
        crc32_value = zlib.crc32(self.file_binary) & 0xFFFFFFFF
        self.crc32_hex = format(crc32_value, "08x")

    def _load_audio(self):
        if isinstance(self.audio_input, bytes):
            self.file_binary = self.audio_input
        elif isinstance(self.audio_input, str):
            ext = self.audio_input.rsplit(".", 1)[-1].lower()
            if ext not in self.SUPPORTED_FORMATS:
                raise ValueError(f"不支持的音频格式: {ext}")
            if not os.path.exists(self.audio_input):
                raise FileNotFoundError(f"文件不存在: {self.audio_input}")
            with open(self.audio_input, "rb") as f:
                self.file_binary = f.read()
        else:
            raise TypeError("audio_input 必须是文件路径(str)或音频数据(bytes)")

    def _get_audio_duration(self) -> float:
        """获取音频时长（秒），可选依赖 pydub"""
        if not self.file_binary:
            return 0.0
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(BytesIO(self.file_binary))
            return audio.duration_seconds
        except ImportError:
            return 60.0  # fallback: 假设 60 秒

    def run(
        self,
        callback: Optional[Callable[[int, str], None]] = None,
        **kwargs,
    ) -> ASRData:
        """执行 ASR，返回 ASRData"""
        resp_data = self._run(callback, **kwargs)
        segments = self._make_segments(resp_data)
        return ASRData(segments)

    def _run(self, callback=None, **kwargs) -> dict:
        raise NotImplementedError

    def _make_segments(self, resp_data: dict) -> List[ASRDataSeg]:
        raise NotImplementedError


# ============================================================================
# BcutASR - 必剪 ASR (Bilibili)
# ============================================================================

class BcutASR(BaseASR):
    """Bilibili 必剪 ASR — 完全独立，开箱即用，不需要任何外部服务。

    使用 Bilibili 云端语音识别，支持分片上传。
    免费，无需 API Key，无需签名服务。

    Parameters:
        audio_input:           音频文件路径 (str) 或音频字节数据 (bytes)
        need_word_time_stamp:  是否返回词级时间戳 (默认 False，返回句子级)

    Example:
        # 最简单用法
        result = BcutASR("audio.mp3").run()
        result.save("output.srt")

        # 带进度回调
        def progress(pct, msg):
            print(f"进度 {pct}%: {msg}")
        result = BcutASR("audio.mp3").run(callback=progress)

        # 词级时间戳
        result = BcutASR("audio.mp3", need_word_time_stamp=True).run()

        # 从 bytes 读取
        result = BcutASR(audio_bytes).run()
    """

    API_BASE = "https://member.bilibili.com/x/bcut/rubick-interface"
    HEADERS = {
        "User-Agent": "Bilibili/1.0.0 (https://www.bilibili.com)",
        "Content-Type": "application/json",
    }

    def __init__(self, audio_input: Union[str, bytes], need_word_time_stamp: bool = False):
        super().__init__(audio_input)
        self.need_word_time_stamp = need_word_time_stamp
        self.session = requests.Session()
        self.task_id: Optional[str] = None
        self._etags: List[str] = []
        self._download_url: Optional[str] = None

    # ---- upload ----

    def _upload(self) -> None:
        """上传音频文件到 Bilibili"""
        if not self.file_binary:
            raise ValueError("没有音频数据")

        # 1) 请求上传授权
        payload = json.dumps({
            "type": 2,
            "name": "audio.mp3",
            "size": len(self.file_binary),
            "ResourceFileType": "mp3",
            "model_id": "8",
        })
        resp = requests.post(
            f"{self.API_BASE}/resource/create",
            data=payload, headers=self.HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()["data"]

        upload_urls = data["upload_urls"]
        per_size = data["per_size"]
        in_boss_key = data["in_boss_key"]
        resource_id = data["resource_id"]
        upload_id = data["upload_id"]

        # 2) 分片上传
        for i, url in enumerate(upload_urls):
            start = i * per_size
            end = min((i + 1) * per_size, len(self.file_binary))
            r = requests.put(url, data=self.file_binary[start:end], headers=self.HEADERS)
            r.raise_for_status()
            etag = r.headers.get("Etag")
            if etag:
                self._etags.append(etag)

        # 3) 提交上传
        resp = requests.post(
            f"{self.API_BASE}/resource/create/complete",
            data=json.dumps({
                "InBossKey": in_boss_key,
                "ResourceId": resource_id,
                "Etags": ",".join(self._etags),
                "UploadId": upload_id,
                "model_id": "8",
            }),
            headers=self.HEADERS,
        )
        resp.raise_for_status()
        self._download_url = resp.json()["data"]["download_url"]

    # ---- task ----

    def _create_task(self) -> str:
        """创建识别任务"""
        resp = requests.post(
            f"{self.API_BASE}/task",
            json={"resource": self._download_url, "model_id": "8"},
            headers=self.HEADERS,
        )
        resp.raise_for_status()
        self.task_id = resp.json()["data"]["task_id"]
        return self.task_id

    def _query_result(self) -> dict:
        """查询识别结果"""
        resp = requests.get(
            f"{self.API_BASE}/task/result",
            params={"model_id": 7, "task_id": self.task_id},
            headers=self.HEADERS,
        )
        resp.raise_for_status()
        return resp.json()["data"]

    # ---- run ----

    def _run(self, callback=None, **kwargs) -> dict:
        if callback:
            callback(*ASRStatus.UPLOADING.callback_tuple())
        self._upload()

        if callback:
            callback(*ASRStatus.CREATING_TASK.callback_tuple())
        self._create_task()

        if callback:
            callback(*ASRStatus.TRANSCRIBING.callback_tuple())

        # 轮询结果（最多 500 秒）
        for _ in range(500):
            result = self._query_result()
            if result["state"] == 4:
                break
            time.sleep(1)
        else:
            raise TimeoutError("ASR 任务超时")

        if callback:
            callback(*ASRStatus.COMPLETED.callback_tuple())

        return json.loads(result["result"])

    def _make_segments(self, resp_data: dict) -> List[ASRDataSeg]:
        if self.need_word_time_stamp:
            return [
                ASRDataSeg(w["label"].strip(), w["start_time"], w["end_time"])
                for u in resp_data["utterances"]
                for w in u["words"]
            ]
        return [
            ASRDataSeg(u["transcript"], u["start_time"], u["end_time"])
            for u in resp_data["utterances"]
        ]


# ============================================================================
# JianYingASR - 剪映 ASR (字节跳动)
# ============================================================================

# ---- AWS 签名工具函数 ----

def _aws_sign_key(secret_key: str, date_stamp: str, region: str, service: str) -> bytes:
    k = hmac.new(("AWS4" + secret_key).encode(), date_stamp.encode(), hashlib.sha256).digest()
    k = hmac.new(k, region.encode(), hashlib.sha256).digest()
    k = hmac.new(k, service.encode(), hashlib.sha256).digest()
    k = hmac.new(k, "aws4_request".encode(), hashlib.sha256).digest()
    return k


def _aws_signature(
    secret_key: str,
    request_parameters: str,
    headers: Dict[str, str],
    region: str = "cn-north-1",
    service: str = "vod",
) -> str:
    """生成 AWS V4 签名"""
    canonical_uri = "/"
    canonical_querystring = request_parameters
    signed_headers = ";".join(headers.keys())
    canonical_headers = "\n".join(f"{k}:{v}" for k, v in headers.items()) + "\n"
    payload_hash = hashlib.sha256(b"").hexdigest()
    canonical_request = (
        f"GET\n{canonical_uri}\n{canonical_querystring}\n"
        f"{canonical_headers}\n{signed_headers}\n{payload_hash}"
    )
    amzdate = headers["x-amz-date"]
    datestamp = amzdate.split("T")[0]
    credential_scope = f"{datestamp}/{region}/{service}/aws4_request"
    string_to_sign = (
        f"AWS4-HMAC-SHA256\n{amzdate}\n{credential_scope}\n"
        f"{hashlib.sha256(canonical_request.encode()).hexdigest()}"
    )
    signing_key = _aws_sign_key(secret_key, datestamp, region, service)
    return hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()


class JianYingASR(BaseASR):
    """字节跳动剪映 ASR — 识别准确率高，但需外部签名服务。

    使用剪映云端语音识别，通过 S3 风格上传音频。
    免费，无需 API Key。

    签名服务:
        默认使用第三方签名服务 asrtools-update.bkfeng.top
        如果不可用，请自行替换类属性:
            JianYingASR.SIGN_SERVICE_URL = "https://你的服务/sign"

    Parameters:
        audio_input:           音频文件路径 (str) 或音频字节数据 (bytes)
        need_word_time_stamp:  是否返回词级时间戳 (默认 False)
        start_time:            音频开始时间，秒 (默认 0)
        end_time:              音频结束时间，秒 (默认 6000，基本覆盖全部)

    Example:
        # 最简单用法
        result = JianYingASR("audio.mp3").run()
        result.save("output.srt")

        # 如果签名服务不可用，替换为自己的:
        JianYingASR.SIGN_SERVICE_URL = "https://my-sign.com/sign"
        result = JianYingASR("audio.mp3").run()

        # 设置音频时间范围
        result = JianYingASR("audio.mp3", start_time=10, end_time=120).run()
    """

    SIGN_SERVICE_URL = "https://asrtools-update.bkfeng.top/sign"
    LEGACY_STATIC_SIGN = "9ea624edbaf4993b326ed127069b8c3f"
    LEGACY_STATIC_DEVICE_TIME = "1626958657"
    LEGACY_STATIC_TDID = "3958721115876654"
    ALLOW_LEGACY_STATIC_SIGN = True

    def __init__(
        self,
        audio_input: Union[str, bytes],
        need_word_time_stamp: bool = False,
        start_time: float = 0,
        end_time: float = 6000,
    ):
        super().__init__(audio_input)
        self.need_word_time_stamp = need_word_time_stamp
        self.start_time = start_time
        self.end_time = end_time

        # 上传相关
        self._session_token: Optional[str] = None
        self._secret_key: Optional[str] = None
        self._access_key: Optional[str] = None
        self._store_uri: Optional[str] = None
        self._auth: Optional[str] = None
        self._upload_id: Optional[str] = None
        self._session_key: Optional[str] = None
        self._upload_hosts: Optional[str] = None

        self._tdid = self._gen_tdid()
        self._last_appvr = "6.6.0"
        self._using_legacy_static_sign = False

    # ---- 设备 ID ----

    @staticmethod
    def _gen_tdid() -> str:
        year = str(datetime.datetime.now().year)
        i = year[3]
        fr = 390 + int(i)
        if int(i) % 2 != 0:
            ed = "3278516897751"
        else:
            ed = f"{uuid.getnode():013d}"
        return f"{fr}{ed}"

    # ---- 签名 ----

    def _get_sign(self, url: str, pf: str = "4", appvr: str = "6.6.0") -> Tuple[str, str]:
        """通过远程服务获取请求签名"""
        self._last_appvr = appvr
        current_time = str(int(time.time()))
        force_legacy = os.environ.get("JIANYING_FORCE_LEGACY_STATIC_SIGN", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        allow_legacy = self.ALLOW_LEGACY_STATIC_SIGN and os.environ.get(
            "JIANYING_DISABLE_LEGACY_STATIC_SIGN", ""
        ).strip().lower() not in {"1", "true", "yes"}

        if not force_legacy:
            try:
                resp = requests.post(
                    self.SIGN_SERVICE_URL,
                    json={
                        "url": url,
                        "current_time": current_time,
                        "pf": pf,
                        "appvr": appvr,
                        "tdid": self._tdid,
                    },
                    headers={
                        "User-Agent": "VideoCaptioner",
                        "tdid": self._tdid,
                        "t": current_time,
                    },
                    timeout=20,
                )
                resp.raise_for_status()
                data = resp.json()
                sign = data.get("sign")
                if sign:
                    return sign.lower(), current_time
            except Exception:
                if not allow_legacy:
                    raise
        if not allow_legacy:
            raise ValueError("签名服务不可用，且内置兼容签名已禁用")

        self._using_legacy_static_sign = True
        self._tdid = self.LEGACY_STATIC_TDID
        return self.LEGACY_STATIC_SIGN, self.LEGACY_STATIC_DEVICE_TIME

    def _build_headers(self, device_time: str, sign: str) -> Dict[str, str]:
        return {
            "User-Agent": "Cronet/TTNetVersion:d4572e53 2024-06-12 QuicVersion:4bf243e0 2023-04-17",
            "appvr": self._last_appvr,
            "device-time": device_time,
            "pf": "4",
            "sign": sign,
            "sign-ver": "1",
            "tdid": self._tdid,
        }

    # ---- 上传 ----

    def _upload_sign(self):
        """获取上传凭证"""
        url = "https://lv-pc-api-sinfonlinec.ulikecam.com/lv/v1/upload_sign"
        sign, device_time = self._get_sign("/lv/v1/upload_sign", appvr="1.4.4")
        headers = self._build_headers(device_time, sign)
        resp = requests.post(url, data=json.dumps({"biz": "pc-recognition"}), headers=headers)
        resp.raise_for_status()
        body = resp.json()
        if body.get("ret") != "0" or not isinstance(body.get("data"), dict):
            raise RuntimeError(f"剪映 upload_sign 失败：{body.get('errmsg') or body}")
        data = body["data"]
        self._access_key = data["access_key_id"]
        self._secret_key = data["secret_access_key"]
        self._session_token = data["session_token"]

    def _upload_auth(self):
        """获取上传授权"""
        file_size = len(self.file_binary) if self.file_binary else 0
        request_params = (
            f"Action=ApplyUploadInner&FileSize={file_size}&FileType=object"
            f"&IsInner=1&SpaceName=lv-mac-recognition&Version=2020-11-19&s=5y0udbjapi"
        )

        t = datetime.datetime.now(datetime.timezone.utc)
        amz_date = t.strftime("%Y%m%dT%H%M%SZ")
        datestamp = t.strftime("%Y%m%d")
        headers = {"x-amz-date": amz_date, "x-amz-security-token": self._session_token}

        region = "cn-north-1"
        signature = _aws_signature(self._secret_key, request_params, headers, region=region)
        authorization = (
            f"AWS4-HMAC-SHA256 Credential={self._access_key}/{datestamp}/{region}/vod/aws4_request, "
            f"SignedHeaders=x-amz-date;x-amz-security-token, Signature={signature}"
        )
        headers["authorization"] = authorization

        resp = requests.get(
            f"https://vod.bytedanceapi.com/?{request_params}", headers=headers
        )
        resp.raise_for_status()
        store_info = resp.json()["Result"]["UploadAddress"]["StoreInfos"][0]

        self._store_uri = store_info["StoreUri"]
        self._auth = store_info["Auth"]
        self._upload_id = store_info["UploadID"]
        self._session_key = resp.json()["Result"]["UploadAddress"]["SessionKey"]
        self._upload_hosts = resp.json()["Result"]["UploadAddress"]["UploadHosts"][0]

    def _upload_file(self):
        """上传音频文件"""
        url = f"https://{self._upload_hosts}/{self._store_uri}?partNumber=1&uploadID={self._upload_id}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36 Thea/1.0.1"
            ),
            "Authorization": self._auth,
            "Content-CRC32": self.crc32_hex,
        }
        resp = requests.put(url, data=self.file_binary, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if data["success"] != 0:
            raise RuntimeError(f"文件上传失败: {resp.text}")

    def _upload_check(self):
        """确认上传"""
        url = f"https://{self._upload_hosts}/{self._store_uri}?uploadID={self._upload_id}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36 Thea/1.0.1"
            ),
            "Authorization": self._auth,
            "Content-CRC32": self.crc32_hex,
        }
        resp = requests.post(url, data=f"1:{self.crc32_hex}", headers=headers)
        resp.raise_for_status()

    def _upload_commit(self):
        """提交上传"""
        url = (
            f"https://{self._upload_hosts}/{self._store_uri}?"
            f"uploadID={self._upload_id}&partNumber=1&x-amz-security-token={self._session_token}"
        )
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36 Thea/1.0.1"
            ),
            "Authorization": self._auth,
            "Content-CRC32": self.crc32_hex,
        }
        resp = requests.put(url, data=self.file_binary, headers=headers)
        try:
            resp.raise_for_status()
        except requests.RequestException:
            # The legacy JianYing upload flow often returns MismatchChecksum here,
            # while the preceding upload/check already made the StoreUri usable for
            # audio_subtitle/submit. Keep the URI and let submit/query be decisive.
            return self._store_uri
        return self._store_uri

    def _upload(self) -> str:
        """完整上传流程"""
        self._upload_sign()
        self._upload_auth()
        self._upload_file()
        self._upload_check()
        return self._upload_commit()

    # ---- 提交与查询 ----

    def _submit(self) -> str:
        """提交识别任务"""
        url = "https://lv-pc-api-sinfonlinec.ulikecam.com/lv/v1/audio_subtitle/submit"
        sign, device_time = self._get_sign("/lv/v1/audio_subtitle/submit")
        headers = self._build_headers(device_time, sign)

        payload = {
            "adjust_endtime": 200,
            "audio": self._store_uri,
            "caption_type": 2,
            "client_request_id": str(uuid.uuid4()),
            "max_lines": 1,
            "songs_info": [{"end_time": self.end_time, "id": "", "start_time": self.start_time}],
            "words_per_line": 16,
        }
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if data.get("ret") != "0":
            raise RuntimeError(f"API 错误: {data.get('errmsg', '未知')} (ret={data.get('ret')})")
        return data["data"]["id"]

    def _query(self, query_id: str) -> dict:
        """查询识别结果"""
        url = "https://lv-pc-api-sinfonlinec.ulikecam.com/lv/v1/audio_subtitle/query"
        sign, device_time = self._get_sign("/lv/v1/audio_subtitle/query")
        headers = self._build_headers(device_time, sign)

        resp = requests.post(
            url,
            json={"id": query_id, "pack_options": {"need_attribute": True}},
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("ret") != "0":
            raise RuntimeError(f"API 错误: {data.get('errmsg', '未知')} (ret={data.get('ret')})")
        return data

    # ---- run ----

    def _run(self, callback=None, **kwargs) -> dict:
        if callback:
            callback(*ASRStatus.UPLOADING.callback_tuple())
        self._upload()

        if callback:
            callback(*ASRStatus.SUBMITTING.callback_tuple())
        query_id = self._submit()

        if callback:
            callback(*ASRStatus.QUERYING_RESULT.callback_tuple())
        result = self._query(query_id)

        if callback:
            callback(*ASRStatus.COMPLETED.callback_tuple())
        return result

    def _make_segments(self, resp_data: dict) -> List[ASRDataSeg]:
        if self.need_word_time_stamp:
            return [
                ASRDataSeg(w["text"].strip(), w["start_time"], w["end_time"])
                for u in resp_data["data"]["utterances"]
                for w in u["words"]
            ]
        return [
            ASRDataSeg(u["text"], u["start_time"], u["end_time"])
            for u in resp_data["data"]["utterances"]
        ]


# ============================================================================
# KuaiShouASR - 快手 ASR
# ============================================================================

class KuaiShouASR(BaseASR):
    """快手在线字幕接口。

    该接口来自 AsrTools 的 K 接口实现，调用方式简单，不需要签名服务。
    2026-06-25 实测接口返回 `code=501, msg=效果subtitle_generate禁用`，
    因此当前 WebUI 会将其标记为不可用；保留适配器用于后续接口恢复验证。
    """

    API_URL = "https://ai.kuaishou.com/api/effects/subtitle_generate"

    def _run(self, callback=None, **kwargs) -> dict:
        if callback:
            callback(*ASRStatus.UPLOADING.callback_tuple())
        if not self.file_binary:
            raise ValueError("没有音频数据")
        files = [("file", ("audio.mp3", self.file_binary, "audio/mpeg"))]
        response = requests.post(
            self.API_URL,
            data={"typeId": "1"},
            files=files,
            timeout=int(kwargs.get("timeout", 60)),
        )
        response.raise_for_status()
        data = response.json()
        if data.get("code") != 0:
            raise RuntimeError(f"快手 ASR 不可用：{data.get('msg') or data}")
        if not isinstance(data.get("data"), dict) or not isinstance(data["data"].get("text"), list):
            raise RuntimeError(f"快手 ASR 返回格式异常：{data}")
        if callback:
            callback(*ASRStatus.COMPLETED.callback_tuple())
        return data

    def _make_segments(self, resp_data: dict) -> List[ASRDataSeg]:
        return [
            ASRDataSeg(item["text"], item["start_time"], item["end_time"])
            for item in resp_data["data"]["text"]
            if item.get("text")
        ]


# ============================================================================
# 便捷函数
# ============================================================================

def bcut_transcribe(audio: Union[str, bytes], word_level: bool = False) -> ASRData:
    """使用必剪 ASR 转写音频"""
    return BcutASR(audio, need_word_time_stamp=word_level).run()


def jianying_transcribe(audio: Union[str, bytes], word_level: bool = False) -> ASRData:
    """使用剪映 ASR 转写音频"""
    return JianYingASR(audio, need_word_time_stamp=word_level).run()


def kuaishou_transcribe(audio: Union[str, bytes]) -> ASRData:
    """使用快手 ASR 转写音频"""
    return KuaiShouASR(audio).run()


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python online_asr.py <音频文件> [bcut|jianying|kuaishou]")
        print("  bcut       - 使用必剪 ASR (默认)")
        print("  jianying   - 使用剪映 ASR")
        print("  kuaishou   - 使用快手 ASR")
        sys.exit(1)

    audio_file = sys.argv[1]
    engine = sys.argv[2] if len(sys.argv) > 2 else "bcut"

    if engine == "bcut":
        result = bcut_transcribe(audio_file)
    elif engine == "jianying":
        result = jianying_transcribe(audio_file)
    elif engine == "kuaishou":
        result = kuaishou_transcribe(audio_file)
    else:
        print(f"未知引擎: {engine}")
        sys.exit(1)

    print(f"\n识别到 {len(result)} 个片段:\n")
    for seg in result:
        ts = seg.to_srt_ts()
        print(f"  [{ts}] {seg.text}")

    # 同时保存 SRT 文件
    srt_path = os.path.splitext(audio_file)[0] + f".{engine}.srt"
    result.save(srt_path, fmt="srt")
    print(f"\n已保存: {srt_path}")
