from avtv.config import Settings


def test_settings_loads_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("PEXELS_API_KEY", "test-pexels")
    monkeypatch.setenv("PIXABAY_API_KEY", "test-pixabay")
    monkeypatch.setenv("UNSPLASH_API_KEY", "test-unsplash")

    s = Settings()
    assert s.anthropic_api_key == "test-anthropic"
    assert s.default_llm_provider == "claude"
    assert s.target_block_duration == 8.0
    assert s.target_resolution == (1920, 1080)
    assert s.target_fps == 30
    assert s.clip_audio_db_offset == -20.0


def test_settings_default_provider_override(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("PEXELS_API_KEY", "x")
    monkeypatch.setenv("PIXABAY_API_KEY", "x")
    monkeypatch.setenv("UNSPLASH_API_KEY", "x")
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "gpt")

    s = Settings()
    assert s.default_llm_provider == "gpt"
