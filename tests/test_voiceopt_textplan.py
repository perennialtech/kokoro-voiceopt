import pytest
from kokoro_voiceopt.config import TextConfig
from kokoro_voiceopt.textplan import (build_text_plan, load_validation_texts,
                                      split_transcript_sentences)


def test_requires_non_empty_transcript():
    with pytest.raises(ValueError):
        build_text_plan(TextConfig(target_transcript="   "))


def test_sentence_splitting():
    sentences = split_transcript_sentences("Hello world. This is a test! Are we done?")
    assert sentences == ["Hello world.", "This is a test!", "Are we done?"]


def test_text_plan_merges_and_limits():
    plan = build_text_plan(
        TextConfig(
            target_transcript="Short. Another short sentence. This one is longer and useful.",
            min_text_chars=20,
            max_text_chars=80,
            max_optimization_texts=2,
            max_validation_texts=2,
        )
    )
    assert len(plan.optimization_texts) <= 2
    assert len(plan.validation_texts) <= 2
    assert all(t.strip() for t in plan.optimization_texts)


def test_validation_text_loading(tmp_path):
    path = tmp_path / "validation.txt"
    path.write_text(
        "# comment\nOne validation sentence.\n\nAnother validation sentence.\n",
        encoding="utf-8",
    )

    texts = load_validation_texts(path)
    assert texts == ["One validation sentence.", "Another validation sentence."]

    plan = build_text_plan(
        TextConfig(
            target_transcript="This is the target transcript.",
            validation_texts_path=path,
            max_validation_texts=2,
        )
    )
    assert plan.validation_texts == [
        "One validation sentence.",
        "Another validation sentence.",
    ]
