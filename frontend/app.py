
import sys
from pathlib import Path
import streamlit as st
from backend import pipeline
from backend.youtube_utils import extract_video_id, watch_url, InvalidYouTubeURL

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


st.set_page_config(page_title="YouTube Chatbot",
                   page_icon="🎥", layout="centered")


def init_state():
    defaults = {
        "video_id": None,
        "chain": None,
        "stats": None,
        "messages": [],
        "error": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_state():
    """Drop the current video, its chain and the chat history."""
    for key in ("video_id", "chain", "stats", "error"):
        st.session_state[key] = None
    st.session_state["messages"] = []


init_state()
is_ready = st.session_state["chain"] is not None


st.title("🎥 YouTube Chatbot")
st.caption("Chat with any YouTube video")
st.divider()


if not is_ready:
    url = st.text_input(
        "YouTube Video URL",
        placeholder="Paste YouTube video URL here...",
        key="url_input",
    )

    if st.button("Process Video", type="primary", use_container_width=True):
        st.session_state["error"] = None
        try:
            video_id = extract_video_id(url)
        except InvalidYouTubeURL as e:
            st.session_state["error"] = f"❌ {e}"
        else:
            # st.status gives a live, expandable log of each stage.
            with st.status("Processing video...", expanded=True) as status:
                st.write(f"✓ Video ID extracted: `{video_id}`")
                try:
                    result = pipeline.process_video(
                        video_id, progress=st.write)
                except pipeline.PipelineError as e:
                    status.update(label="Processing failed", state="error")
                    st.session_state["error"] = f"❌ {e}"
                except Exception as e:
                    # Last resort: never show the user a traceback.
                    status.update(label="Processing failed", state="error")
                    st.session_state["error"] = (
                        f"❌ Unexpected problem while processing this video "
                        f"({type(e).__name__})."
                    )
                else:
                    st.session_state["video_id"] = video_id
                    st.session_state["chain"] = result["chain"]
                    st.session_state["stats"] = {
                        "chars": result["transcript_chars"],
                        "chunks": result["num_chunks"],
                    }
                    status.update(label="Video ready!", state="complete")
                    st.rerun()

    if st.session_state["error"]:
        st.error(st.session_state["error"])

    with st.expander("Supported URL formats"):
        st.markdown(
            "- `youtube.com/watch?v=ID` (extra `&list=` / `&index=` parameters are fine)\n"
            "- `youtu.be/ID`\n"
            "- `youtube.com/shorts/ID`\n"
            "- `youtube.com/embed/ID` and `youtube.com/live/ID`\n\n"
            "The video must have captions available — the chatbot reads its transcript."
        )


else:
    video_id = st.session_state["video_id"]
    stats = st.session_state["stats"]

    left, right = st.columns([3, 2])
    with left:
        st.video(watch_url(video_id))
    with right:
        st.subheader("Video Information")
        st.markdown(
            f"**Video ID**  \n`{video_id}`\n\n"
            f"**Status**  \n✅ Ready\n\n"
            f"**Transcript**  \n{stats['chars']:,} characters\n\n"
            f"**Indexed**  \n{stats['chunks']} chunks"
        )
        if st.button("🔄 Load Another Video", use_container_width=True):
            reset_state()
            st.rerun()

    st.divider()
    st.subheader("Chat with Video")

    # Replay the conversation so far.
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about this video...")

    if question:
        if not question.strip():
            st.warning("Please type a question.")
        else:
            st.session_state["messages"].append(
                {"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Reuses the chain built at process time: no re-embedding.
                        answer = pipeline.answer_question(
                            st.session_state["chain"], question
                        )
                    except pipeline.PipelineError as e:
                        answer = f"❌ {e}"
                    except Exception as e:
                        answer = (
                            f"❌ Unexpected problem answering that "
                            f"({type(e).__name__})."
                        )
                st.markdown(answer)

            st.session_state["messages"].append(
                {"role": "assistant", "content": answer}
            )
