from __future__ import annotations

from pathlib import Path

import torch
import streamlit as st

from recommender.model import PlaylistRecommender

st.set_page_config(
    page_title="Spotify Playlist Recommender",
    page_icon="🎵",
    layout="wide",
)

SHOW_TOP_K = 30

GREEN = "#1DB954"
GREEN_DARK = "#158a3e"

st.markdown(
    f"""
    <style>
        /* Spotify-green primary buttons */
        div.stButton > button[kind="primary"] {{
            background-color: {GREEN};
            color: #fff;
            border: none;
        }}
        div.stButton > button[kind="primary"]:hover {{
            background-color: {GREEN_DARK};
            color: #fff;
            border: none;
        }}

        /* secondary buttons (+, -, Clear playlist) — green outline + hover */
        div.stButton button[kind="secondary"],
        div.stButton button[data-testid="baseButton-secondary"] {{
            background-color: transparent !important;
            border: 1px solid {GREEN} !important;
            color: {GREEN} !important;
        }}
        div.stButton button[kind="secondary"]:hover,
        div.stButton button[data-testid="baseButton-secondary"]:hover {{
            background-color: {GREEN} !important;
            border: 1px solid {GREEN} !important;
            color: #fff !important;
        }}

        /* Recommendation cards */
        .rec-card {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 12px;
            margin-bottom: 4px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background: #fafafa;
        }}

        .rec-pos {{
            font-size: 0.8rem;
            color: #888;
            min-width: 24px;
            margin-right: 10px;
        }}
        .rec-text {{
            flex: 1;
            font-size: 0.9rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .rec-artist {{
            font-weight: 600;
        }}

        /* Tighten spacing in the playlist panel */
        .playlist-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 5px 10px;
            border-radius: 6px;
        }}
        .playlist-row:hover {{
            background: #f0faf4;
        }}

        /* Green accent on the page title */
        h1 span.green {{ color: {GREEN}; }}

        /* Reduce default top padding */
        .block-container {{ padding-top: 2rem; }}

    </style>
    """,
    unsafe_allow_html=True,
)


def _load_model(checkpoint_path: str):
    """Load model from checkpoint and store in session state."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PlaylistRecommender.from_dict(checkpoint["model"])
    inf_model = model.to_inference_model(freeze_track_embedder=True)
    st.session_state.inf_model = inf_model
    st.session_state.device = device


if "inf_model" not in st.session_state:
    st.markdown(
        "<h1>🎵 Spotify Playlist <span style='color:#1DB954'>Recommender</span></h1>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.subheader("Load Model")
        checkpoint_path = st.text_input(
            "Checkpoint path",
            value="checkpoints/model.pt",
            placeholder="checkpoints/model.pt",
        )
        if st.button("Load Model", type="primary", use_container_width=True):
            if not checkpoint_path.strip():
                st.error("Please enter a checkpoint path.")
            elif not Path(checkpoint_path).exists():
                st.error(f"File not found: `{checkpoint_path}`")
            else:
                with st.spinner("Loading model…"):
                    try:
                        _load_model(checkpoint_path.strip())
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Failed to load model: {exc}")
    st.stop()

inf_model = st.session_state.inf_model
tensoriser = inf_model.tensoriser
vocab_size = tensoriser.vocab_size

if "playlist" not in st.session_state:
    st.session_state.playlist = []  # list of track_id ints
if "search_key" not in st.session_state:
    st.session_state.search_key = 0


def _track_label(track_id: int) -> str:
    return (
        f"{tensoriser.track_id_to_artist[track_id]} — "
        f"{tensoriser.track_id_to_name[track_id]}"
    )


def _add_track(track_id: int) -> None:
    st.session_state.playlist.append(track_id)


def _remove_track(index: int) -> None:
    st.session_state.playlist.pop(index)


def _search_add() -> None:
    """on_change callback for the search selectbox."""
    key = f"search_select_{st.session_state.search_key}"
    selected_label = st.session_state.get(key)
    if selected_label and selected_label != _SEARCH_PLACEHOLDER:
        track_id = st.session_state._search_label_to_id.get(selected_label)
        if track_id is not None:
            _add_track(track_id)
    st.session_state.search_key += 1


_SEARCH_PLACEHOLDER = "Search tracks…"


st.markdown(
    "<h1 style='margin-bottom:0.2rem'>🎵 Spotify Playlist "
    "<span style='color:#1DB954'>Recommender</span></h1>",
    unsafe_allow_html=True,
)
st.markdown("<hr style='margin-top:0.4rem;margin-bottom:1rem'>", unsafe_allow_html=True)

left, right = st.columns([2, 1], gap="large")


with left:
    playlist_name = st.text_input("Playlist name", placeholder="Enter a playlist name…")

    cols_controls = st.columns([1, 1])
    with cols_controls[0]:
        top_k = st.number_input(
            "Top K (recommendations to fetch)",
            min_value=1,
            max_value=vocab_size,
            value=min(1000, vocab_size),
            step=1,
        )
    with cols_controls[1]:
        exclude_added = st.checkbox(
            "Exclude added tracks from recommendations", value=True
        )

    if not playlist_name.strip():
        st.info("Give the playlist a name to see recommendations")
    else:
        
        allowed_mask: torch.Tensor | None = None
        if exclude_added and st.session_state.playlist:
            allowed_mask = ~inf_model.tensoriser.make_track_mask(
                st.session_state.playlist
            )

        # fetch recommendations — cached so that the throwaway run triggered by
        # st.rerun() (before the playlist has actually changed) is a no-op.
        _rec_key = (
            playlist_name.strip(),
            tuple(st.session_state.playlist),
            int(top_k),
            exclude_added,
        )
        if st.session_state.get("_rec_key") != _rec_key:
            recs = inf_model.get_recommendations(
                playlist_name=playlist_name.strip(),
                playlist=st.session_state.playlist,
                top_k=int(top_k),
                allowed_mask=allowed_mask,
            )
            st.session_state["_rec_key"] = _rec_key
            st.session_state["_recs"] = recs
        else:
            recs = st.session_state["_recs"]

        # search selectbox over all top_k results
        label_to_id: dict[str, int] = {
            f"{r.artist} — {r.track}": r.track_id for r in recs
        }
        st.session_state._search_label_to_id = label_to_id
        search_options = [_SEARCH_PLACEHOLDER] + list(label_to_id.keys())
        search_key = f"search_select_{st.session_state.search_key}"
        st.selectbox(
            "Search and add a track",
            options=search_options,
            index=0,
            key=search_key,
            on_change=_search_add,
        )

        st.subheader("You might like next")
        for rec in recs[:SHOW_TOP_K]:
            card_col, btn_col = st.columns([10, 1])
            with card_col:
                st.markdown(
                    f"""<div class="rec-card">
                        <span class="rec-pos">{rec.position}</span>
                        <span class="rec-text">
                            <span class="rec-artist">{rec.artist}</span>
                            &nbsp;—&nbsp;{rec.track}
                        </span>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with btn_col:
                if st.button(
                    "＋",
                    key=f"add_{rec.track_id}_{rec.position}",
                    help="Add to playlist"
                ):
                    _add_track(rec.track_id)
                    st.rerun()

with right:
    playlist_display_name = (
        playlist_name.strip()
        if "playlist_name" in dir() and playlist_name.strip()
        else "Playlist"
    )
    st.header(playlist_display_name)

    if not st.session_state.playlist:
        st.caption("No tracks yet. Add some from the recommendations!")
    else:
        for idx, track_id in enumerate(st.session_state.playlist):
            row_left, row_right = st.columns([8, 1])
            with row_left:
                st.write(f"{idx + 1}. {_track_label(track_id)}")
            with row_right:
                if st.button("－", key=f"remove_{idx}_{track_id}", help="Remove from playlist"):
                    _remove_track(idx)
                    st.rerun()

    st.markdown("---")
    if st.button("Clear playlist", use_container_width=True):
        st.session_state.playlist = []
        st.rerun()
