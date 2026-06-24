import json
import sys
from pathlib import Path


MVP_DIR = Path(__file__).resolve().parents[1] / "mvp_pipeline"
_INSERTED_MVP_DIR = str(MVP_DIR) not in sys.path
if _INSERTED_MVP_DIR:
    sys.path.insert(0, str(MVP_DIR))

import term_mapper  # noqa: E402
from term_mapper import filter_context_only_replacements  # noqa: E402

if _INSERTED_MVP_DIR:
    sys.path.remove(str(MVP_DIR))


def test_filter_context_only_replacements_keeps_pdf_backed_name_alias() -> None:
    payload = {
        "replacements": [
            {"wrong": "陈欣", "correct": "陈歆"},
            {"wrong": "先兆子痫", "correct": "先兆子鞍"},
            {"wrong": "高血压危象", "correct": "高血压急症"},
        ]
    }
    context = "上海交通大学医学院附属瑞金医院 陈歆 高血压急症 高血压危象 先兆子痫"

    filtered = filter_context_only_replacements(payload, context)

    assert filtered["replacements"] == [
        {"wrong": "陈欣", "correct": "陈歆"},
        {
            "wrong": "高血压危象",
            "correct": "高血压急症",
            "review_warning": "wrong_also_appears_in_pdf_context",
        },
    ]
    assert [item["drop_reason"] for item in filtered["dropped_replacements"]] == [
        "correct_not_in_pdf_context",
    ]


def test_read_context_includes_source_filename_for_course_names(tmp_path: Path) -> None:
    context = tmp_path / "黄绮芳-高血压课程.txt"
    context.write_text("上海市高血压研究所", encoding="utf-8")

    text = term_mapper.read_context(context)
    filtered = filter_context_only_replacements(
        {"replacements": [{"wrong": "黄启芳", "correct": "黄绮芳"}]},
        text,
    )

    assert "资料文件名：黄绮芳-高血压课程.txt" in text
    assert filtered["replacements"] == [{"wrong": "黄启芳", "correct": "黄绮芳"}]


def test_deepseek_v4_requests_disable_thinking(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "{\"replacements\": []}"}}]},
                ensure_ascii=False,
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(term_mapper.urllib.request, "urlopen", fake_urlopen)

    result = term_mapper.chat_completion(
        [{"role": "user", "content": "输出 JSON"}],
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com/v1",
        api_key="test",
    )

    assert result == "{\"replacements\": []}"
    assert captured["body"]["thinking"] == {"type": "disabled"}
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["max_tokens"] >= 4096
