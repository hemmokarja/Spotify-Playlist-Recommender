"""Streamlit UI for the Spotify Playlist Recommender."""

from __future__ import annotations

# import sys
from pathlib import Path

import torch
import streamlit as st

# # Allow imports from src/ when running via `streamlit run src/ui/app.py`
# _src = Path(__file__).resolve().parents[2]
# if str(_src) not in sys.path:
#     sys.path.insert(0, str(_src))

from recommender.model import PlaylistRecommender

st.set_page_config(
    page_title="Spotify Playlist Recommender",
    page_icon="🎵",
    layout="wide",
)

def _load_model(checkpoint_path: str):
    """Load model from checkpoint and store in session state."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PlaylistRecommender.from_dict(checkpoint["model"])
    inf_model = model.to_inference_model()
    st.session_state.inf_model = inf_model
    st.session_state.device = device


if "inf_model" not in st.session_state:
    st.title("🎵 Spotify Playlist Recommender")
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


left, right = st.columns([2, 1], gap="large")


with left:
    st.header("Recommendations")

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
        # build allowed_mask
        allowed_mask: torch.Tensor | None = None
        if exclude_added and st.session_state.playlist:
            allowed_mask = torch.ones(vocab_size, dtype=torch.bool)
            for tid in st.session_state.playlist:
                if 0 <= tid < vocab_size:
                    allowed_mask[tid] = False

        # fetch recommendations
        recs = inf_model.get_recommendations(
            playlist_name=playlist_name.strip(),
            playlist=st.session_state.playlist,
            top_k=int(top_k),
            allowed_mask=allowed_mask,
        )

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

        # Top-30 list
        st.subheader("Top 30 recommendations")
        top_30 = recs[:30]
        for rec in top_30:
            row_left, row_right = st.columns([8, 1])
            with row_left:
                st.write(f"**{rec.artist}** — {rec.track}")
            with row_right:
                if st.button("＋", key=f"add_{rec.track_id}_{rec.position}", help="Add to playlist"):
                    _add_track(rec.track_id)
                    st.rerun()

# ===== RIGHT: Current playlist =====

with right:
    playlist_display_name = playlist_name.strip() if "playlist_name" in dir() and playlist_name.strip() else "Playlist"
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
