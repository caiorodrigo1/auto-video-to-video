"""Streamlit UI for auto-video-to-video.

Launch with `avtv ui` (or `streamlit run src/avtv/ui/streamlit_app.py`).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import streamlit as st

from avtv.assembler import assemble_run
from avtv.briefing import get_provider
from avtv.config import Settings
from avtv.downloader import Downloader
from avtv.parser import parse_script
from avtv.runs import RunDir
from avtv.search.archive_org import ArchiveOrgAdapter
from avtv.search.base import SearchAdapter
from avtv.search.orchestrator import SearchOrchestrator
from avtv.search.pexels import PexelsAdapter
from avtv.search.pixabay import PixabayAdapter
from avtv.search.unsplash import UnsplashAdapter
from avtv.search.wikimedia import WikimediaAdapter
from avtv.selector import select_per_block

st.set_page_config(page_title="auto-video-to-video", layout="wide")

STAGES = [
    ("Parse", RunDir.BLOCKS),
    ("Topic", RunDir.TOPIC),
    ("Brief", RunDir.BRIEFS),
    ("Search", RunDir.SEARCH),
    ("Select", RunDir.SELECTIONS),
]


@st.cache_resource
def _settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


def _build_orchestrator(settings: Settings) -> SearchOrchestrator:
    adapters: list[SearchAdapter] = [
        PexelsAdapter(api_key=settings.pexels_api_key),  # type: ignore[list-item]
        PixabayAdapter(api_key=settings.pixabay_api_key),  # type: ignore[list-item]
        ArchiveOrgAdapter(),  # type: ignore[list-item]
        WikimediaAdapter(),  # type: ignore[list-item]
        UnsplashAdapter(api_key=settings.unsplash_api_key),  # type: ignore[list-item]
    ]
    return SearchOrchestrator(adapters=adapters, top_k=settings.search_top_k)


def _list_runs(base: Path) -> list[dict[str, object]]:
    if not base.exists():
        return []
    out = []
    for child in sorted(base.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        rd = RunDir(base_dir=base, run_id=child.name)
        out.append(
            {
                "id": child.name,
                "blocks": rd.has(RunDir.BLOCKS),
                "briefs": rd.has(RunDir.BRIEFS),
                "search": rd.has(RunDir.SEARCH),
                "selections": rd.has(RunDir.SELECTIONS),
                "output": (child / "output.mp4").exists(),
            }
        )
    return out


def _stage_flags(r: dict[str, object]) -> str:
    keys = ["blocks", "briefs", "search", "selections", "output"]
    return "".join("✓" if r[k] else "·" for k in keys)


def _run_stage_parse(rd: RunDir, script_path: Path) -> int:
    blocks = parse_script(script_path.read_text(encoding="utf-8"))
    rd.save_blocks(blocks)
    return len(blocks)


def _run_stage_topic_extract(rd: RunDir, provider_name: str, settings: Settings) -> str:
    """Run the LLM topic detector. Returns the suggested topic string."""
    from avtv.briefing.topic import extract_topic_via_llm

    blocks = rd.load_blocks()
    return extract_topic_via_llm(blocks, provider_name=provider_name, settings=settings)


def _run_stage_topic_save(rd: RunDir, topic: str, suggested: str) -> None:
    from avtv.models import Topic

    source = "user" if topic.strip() != suggested.strip() else "llm"
    rd.save_topic(Topic(topic=topic.strip(), llm_suggestion=suggested, source=source))


def _run_stage_brief(rd: RunDir, provider_name: str) -> int:
    blocks = rd.load_blocks()
    llm = get_provider(provider_name)
    topic_str = rd.load_topic().topic if rd.has_topic() else None
    briefs = llm.generate_briefs(blocks, topic=topic_str)
    rd.save_briefs(briefs)
    return len(briefs)


def _run_stage_search(rd: RunDir, settings: Settings) -> int:
    briefs = rd.load_briefs()
    orch = _build_orchestrator(settings)
    results = asyncio.run(
        orch.search_for_briefs(briefs, concurrency=settings.search_concurrency)
    )
    rd.save_search_results(results)
    return sum(len(v) for v in results.values())


def _run_stage_select(rd: RunDir, settings: Settings) -> int:
    briefs = rd.load_briefs()
    candidates = rd.load_search_results()
    orch = _build_orchestrator(settings)
    sels = asyncio.run(
        select_per_block(
            briefs,
            candidates,
            settings=settings,
            image_search=orch.search_images_for_brief,
        )
    )
    rd.save_selections(sels)
    return len(sels)


def _run_stage_assemble(
    rd: RunDir,
    audio_path: Path,
    settings: Settings,
    on_segment: object | None = None,
    on_mux: object | None = None,
) -> Path:
    sels = rd.load_selections()
    out_path = rd.path / "output.mp4"
    downloader = Downloader(cache_dir=Path(settings.cache_dir))
    assemble_run(
        selections=sels,
        narration_path=audio_path,
        output_path=out_path,
        work_dir=rd.path / "segments",
        downloader=downloader,
        target_w=settings.target_resolution[0],
        target_h=settings.target_resolution[1],
        target_fps=settings.target_fps,
        clip_audio_db=settings.clip_audio_db_offset,
        on_segment=on_segment,  # type: ignore[arg-type]
        on_mux=on_mux,  # type: ignore[arg-type]
    )
    return out_path


def _save_uploads(
    rd: RunDir, audio_file: object, script_file: object
) -> tuple[Path, Path]:
    uploads = rd.path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    audio_path = uploads / audio_file.name  # type: ignore[attr-defined]
    script_path = uploads / script_file.name  # type: ignore[attr-defined]
    audio_path.write_bytes(audio_file.getvalue())  # type: ignore[attr-defined]
    script_path.write_bytes(script_file.getvalue())  # type: ignore[attr-defined]
    return audio_path, script_path


def _audio_path_for(rd: RunDir) -> Path | None:
    uploads = rd.path / "uploads"
    if not uploads.exists():
        return None
    files = sorted(uploads.glob("*.mp3"))
    return files[0] if files else None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

settings = _settings()
runs_base = Path(settings.runs_dir)

with st.sidebar:
    st.title("avtv")
    if st.button("➕ New build", use_container_width=True, type="primary"):
        st.session_state["mode"] = "new"
        st.session_state.pop("run_id", None)
        st.rerun()

    st.divider()
    st.caption("Existing runs")
    runs = _list_runs(runs_base)
    if not runs:
        st.caption("(none yet)")
    for r in runs:
        flags = _stage_flags(r)
        is_selected = st.session_state.get("run_id") == r["id"]
        label = f"{flags}  {r['id']}"
        if st.button(
            label,
            key=f"run-{r['id']}",
            use_container_width=True,
            type="secondary" if not is_selected else "primary",
        ):
            st.session_state["run_id"] = r["id"]
            st.session_state["mode"] = "run"
            st.rerun()

    st.divider()
    st.caption("Legend: parse · brief · search · select · output")
    st.caption(f"Provider default: **{settings.default_llm_provider}**")


# ---------------------------------------------------------------------------
# Main: New build
# ---------------------------------------------------------------------------

mode = st.session_state.get("mode", "new")

if mode == "new":
    st.title("New build")

    col1, col2 = st.columns([2, 1])

    with col1:
        audio_file = st.file_uploader("Audio (MP3)", type=["mp3"])
        script_file = st.file_uploader("Script (DOTTI SYNC .txt)", type=["txt"])

    with col2:
        provider = st.radio(
            "LLM provider",
            ["claude", "gpt"],
            index=0 if settings.default_llm_provider == "claude" else 1,
        )
        custom_id = st.text_input("Run ID (optional)", placeholder="auto-generated")

    files_ready = audio_file is not None and script_file is not None

    # Phase 1: files uploaded but topic not yet detected.
    if files_ready and "pending_topic_run" not in st.session_state:
        if st.button("📑 Detect topic", type="primary"):
            rd = (
                RunDir(base_dir=runs_base, run_id=custom_id)
                if custom_id
                else RunDir.new(runs_base)
            )
            rd.ensure()
            audio_path, script_path = _save_uploads(rd, audio_file, script_file)
            with st.spinner("Parsing & detecting topic…"):
                _run_stage_parse(rd, script_path)
                suggested = _run_stage_topic_extract(rd, provider, settings)
            st.session_state["pending_topic_run"] = {
                "run_id": rd.run_id,
                "audio_path": str(audio_path),
                "suggested_topic": suggested,
                "provider": provider,
            }
            st.rerun()

    # Phase 2: topic suggested, awaiting confirmation.
    if "pending_topic_run" in st.session_state:
        ctx = st.session_state["pending_topic_run"]
        st.info(f"Run ID: `{ctx['run_id']}`")
        st.markdown(f"**Suggested topic:** {ctx['suggested_topic']}")
        confirmed_topic = st.text_input(
            "Confirm or edit the topic",
            value=ctx["suggested_topic"],
            key="confirmed_topic_field",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            cancel = st.button("✖ Cancel")
        with col_b:
            go = st.button("▶ Start build", type="primary")

        if cancel:
            st.session_state.pop("pending_topic_run", None)
            st.rerun()

        if go:
            rd = RunDir(base_dir=runs_base, run_id=ctx["run_id"])
            _run_stage_topic_save(rd, confirmed_topic, ctx["suggested_topic"])
            audio_path = Path(ctx["audio_path"])
            chosen_provider = ctx["provider"]

            with st.status("Running pipeline…", expanded=True) as status:
                st.write(f"**Run ID:** `{rd.run_id}`")
                st.write(f"📑 Topic: `{confirmed_topic}`")

                st.write(f"🧠 **Brief** (via {chosen_provider})")
                n_briefs = _run_stage_brief(rd, chosen_provider)
                st.write(f"  → {n_briefs} briefs")

                st.write("🔍 **Search**")
                n_cands = _run_stage_search(rd, settings)
                st.write(f"  → {n_cands} candidates")

                st.write("🎯 **Select**")
                n_sels = _run_stage_select(rd, settings)
                st.write(f"  → {n_sels} clips")

                st.write("🎬 **Assemble** (download + ffmpeg)")
                seg_bar = st.progress(0.0, text="Encoding segments…")

                def on_segment(i: int, total: int) -> None:
                    seg_bar.progress(i / total, text=f"Segment {i}/{total}")

                def on_mux() -> None:
                    seg_bar.progress(1.0, text="Muxing final MP4…")

                out_path = _run_stage_assemble(
                    rd, audio_path, settings, on_segment=on_segment, on_mux=on_mux
                )
                seg_bar.empty()
                st.write(f"  → {out_path}")

                status.update(label="✅ Build complete", state="complete")

            st.session_state["run_id"] = rd.run_id
            st.session_state["mode"] = "run"
            st.session_state.pop("pending_topic_run", None)
            st.rerun()


# ---------------------------------------------------------------------------
# Main: Run inspector
# ---------------------------------------------------------------------------

elif mode == "run":
    run_id = st.session_state["run_id"]
    rd = RunDir(base_dir=runs_base, run_id=run_id)

    if not rd.path.exists():
        st.error(f"Run `{run_id}` not found")
        st.stop()

    st.title(f"Run: `{run_id}`")

    # Stage indicators
    cols = st.columns(len(STAGES) + 1)
    for col, (label, fname) in zip(cols, STAGES, strict=False):
        col.metric(label, "✓" if rd.has(fname) else "·")
    cols[-1].metric("Output", "✓" if (rd.path / "output.mp4").exists() else "·")

    st.divider()

    tab_briefs, tab_selections, tab_output, tab_actions = st.tabs(
        ["🧠 Briefs", "🎯 Selections", "🎬 Output", "⚙️ Actions"]
    )

    # --- Briefs tab
    with tab_briefs:
        if rd.has(RunDir.BRIEFS):
            briefs = rd.load_briefs()
            st.caption(f"{len(briefs)} briefs")
            st.json([b.model_dump() for b in briefs[:50]], expanded=False)
            if len(briefs) > 50:
                st.caption(f"… (showing first 50 of {len(briefs)})")
        else:
            st.info("No briefs yet — run the brief stage.")

    # --- Selections tab (editable)
    with tab_selections:
        if rd.has(RunDir.SELECTIONS):
            sels_path = rd.path / RunDir.SELECTIONS
            current = sels_path.read_text(encoding="utf-8")

            try:
                sels_data = json.loads(current)
            except json.JSONDecodeError:
                st.warning(
                    "Preview unavailable: JSON is invalid (still editable below)"
                )
            else:
                with st.expander(
                    f"Preview ({len(sels_data)} selections)", expanded=False
                ):
                    for sel in sels_data:
                        idx = sel.get("idx")
                        kind = sel.get("kind")
                        if kind == "image_montage":
                            urls = sel.get("montage_urls", [])
                            sources = sel.get("montage_sources", [])
                            st.caption(f"{idx:03d} · image_montage")
                            if urls:
                                cols = st.columns(len(urls))
                                for col, u, s in zip(
                                    cols, urls, sources, strict=False
                                ):
                                    with col:
                                        st.image(u, width=140)
                                        st.caption(s)
                        elif kind == "continuation":
                            st.caption(
                                f"{idx:03d} · continuation (extends previous)"
                            )
                        elif kind in ("video", "image"):
                            url = sel.get("url", "")
                            source = sel.get("source", "")
                            st.caption(f"{idx:03d} · {kind} · {source}")
                            if url:
                                st.image(url, width=180)
                        st.markdown("---")

            st.caption("Edit `04_selections.json` directly. Re-run assemble after saving.")
            edited = st.text_area(
                "selections", current, height=500, label_visibility="collapsed"
            )
            if st.button("💾 Save selections", key="save-sels"):
                try:
                    json.loads(edited)
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")
                else:
                    sels_path.write_text(edited, encoding="utf-8")
                    st.success("Saved.")
        else:
            st.info("No selections yet.")

    # --- Output tab
    with tab_output:
        out_path = rd.path / "output.mp4"
        if out_path.exists():
            size_mb = out_path.stat().st_size / 1_048_576
            st.caption(f"{out_path.name} — {size_mb:.1f} MB")
            st.video(str(out_path))
            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇ Download MP4",
                    f,
                    file_name=f"{run_id}.mp4",
                    mime="video/mp4",
                )
        else:
            st.info("No output yet — run assemble.")

    # --- Actions tab (per-stage re-run)
    with tab_actions:
        st.caption("Re-run individual stages. Useful after editing selections.")

        a_path: Path | None = _audio_path_for(rd)
        if not a_path:
            st.warning(
                "No audio in `uploads/`. To re-assemble, copy your MP3 into "
                f"`{rd.path / 'uploads'}/`."
            )

        s_files = (
            sorted((rd.path / "uploads").glob("*.txt"))
            if (rd.path / "uploads").exists()
            else []
        )
        s_path: Path | None = s_files[0] if s_files else None

        c1, c2, c3, c4, c5 = st.columns(5)

        if c1.button("📄 Parse", use_container_width=True, disabled=s_path is None):
            assert s_path is not None
            with st.spinner("Parsing…"):
                n = _run_stage_parse(rd, s_path)
            st.success(f"{n} blocks")
            st.rerun()

        provider_choice = st.session_state.get(
            "actions_provider", settings.default_llm_provider
        )
        if c2.button(
            "🧠 Brief",
            use_container_width=True,
            disabled=not rd.has(RunDir.BLOCKS),
        ):
            with st.spinner(f"Briefing via {provider_choice}…"):
                n = _run_stage_brief(rd, provider_choice)
            st.success(f"{n} briefs")
            st.rerun()

        if c3.button(
            "🔍 Search",
            use_container_width=True,
            disabled=not rd.has(RunDir.BRIEFS),
        ):
            with st.spinner("Searching…"):
                n = _run_stage_search(rd, settings)
            st.success(f"{n} candidates")
            st.rerun()

        if c4.button(
            "🎯 Select",
            use_container_width=True,
            disabled=not rd.has(RunDir.SEARCH),
        ):
            with st.spinner("Selecting…"):
                n = _run_stage_select(rd, settings)
            st.success(f"{n} clips")
            st.rerun()

        if c5.button(
            "🎬 Assemble",
            use_container_width=True,
            disabled=not (rd.has(RunDir.SELECTIONS) and a_path),
            type="primary",
        ):
            assert a_path is not None
            with st.status("Assembling…", expanded=True) as status:
                bar = st.progress(0.0, text="Encoding segments…")

                def _on_seg(i: int, total: int) -> None:
                    bar.progress(i / total, text=f"Segment {i}/{total}")

                def _on_mux() -> None:
                    bar.progress(1.0, text="Muxing final MP4…")

                out_path = _run_stage_assemble(
                    rd, a_path, settings, on_segment=_on_seg, on_mux=_on_mux
                )
                bar.empty()
                status.update(label=f"✅ Done → {out_path}", state="complete")
            st.rerun()

        st.divider()
        st.caption("Provider for **Brief** stage:")
        st.radio(
            "provider",
            ["claude", "gpt"],
            index=0 if provider_choice == "claude" else 1,
            key="actions_provider",
            label_visibility="collapsed",
            horizontal=True,
        )
