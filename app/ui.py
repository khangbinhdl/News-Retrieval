import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


st.set_page_config(
    page_title="Vietnamese News Retrieval",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 4.5rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 2.1rem;
    font-weight: 750;
    line-height: 1.25;
    margin-top: 0.5rem;
    margin-bottom: 0.35rem;
    padding-top: 0.25rem;
}

.subtitle {
    color: #666;
    margin-bottom: 1.2rem;
}

.entity-card {
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.55rem;
    background: rgba(250, 250, 250, 0.55);
}

.entity-name {
    font-weight: 700;
    font-size: 1rem;
}

.entity-meta {
    color: #666;
    font-size: 0.85rem;
}

.result-card {
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.9rem;
}

.metric-chip {
    display: inline-block;
    padding: 0.18rem 0.55rem;
    border-radius: 999px;
    background: rgba(128, 128, 128, 0.12);
    margin-right: 0.35rem;
    font-size: 0.83rem;
}

.small-muted {
    color: #777;
    font-size: 0.88rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_state() -> None:
    defaults = {
        "ner_result": None,
        "last_ner_text": "",
        "selected_entity_index": None,
        "search_result": None,
        "last_search_payload": None,
        "last_search_mode": None,
        "api_health": None,
        "segmenter": "underthesea",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def post_json(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{API_URL}{endpoint}"

    try:
        response = requests.post(url, json=payload, timeout=120)
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Không kết nối được API hoặc API đã crash giữa request. "
            "Hãy kiểm tra terminal đang chạy make run-api."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("API xử lý quá lâu và bị timeout.") from exc

    if response.status_code >= 400:
        raise RuntimeError(response.text)

    return response.json()


def get_json(endpoint: str) -> Dict[str, Any]:
    url = f"{API_URL}{endpoint}"

    try:
        response = requests.get(url, timeout=30)
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Không kết nối được API. Hãy chạy make run-api trước."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("API health check bị timeout.") from exc

    if response.status_code >= 400:
        raise RuntimeError(response.text)

    return response.json()


def reset_ner_and_search() -> None:
    st.session_state.ner_result = None
    st.session_state.last_ner_text = ""
    st.session_state.selected_entity_index = None
    st.session_state.search_result = None
    st.session_state.last_search_payload = None
    st.session_state.last_search_mode = None


def reset_search_only() -> None:
    st.session_state.search_result = None
    st.session_state.last_search_payload = None
    st.session_state.last_search_mode = None


def entity_display_label(entity: Dict[str, Any]) -> str:
    text = entity.get("text", "")
    ent_type = entity.get("type", "")
    return f"{text} · {ent_type}" if ent_type else text


def render_entity_cards(entities: List[Dict[str, Any]]) -> None:
    if not entities:
        st.info("Không tìm thấy thực thể.")
        return

    for ent in entities:
        st.markdown(
            f"""
            <div class="entity-card">
                <div class="entity-name">{ent.get("text", "")}</div>
                <div class="entity-meta">
                    type: <b>{ent.get("type", "")}</b>
                    &nbsp; | &nbsp;
                    segmented: <code>{ent.get("segmented_text", "")}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_word_labels(word_labels: List[Dict[str, Any]]) -> None:
    if not word_labels:
        return

    with st.expander("Xem nhãn NER theo từng token"):
        rows = [
            {
                "word": item.get("word", ""),
                "label": item.get("label", ""),
            }
            for item in word_labels
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)


def render_results(results: List[Dict[str, Any]]) -> None:
    if not results:
        st.warning("Không có kết quả.")
        return

    for rank, item in enumerate(results, start=1):
        title = f"#{rank} · {item.get('doc_id', '')} · score={item.get('score', 0):.4f}"

        with st.expander(title, expanded=(rank <= 3)):
            chips = [
                f"method: {item.get('method', '')}",
                f"score: {item.get('score', 0):.4f}",
            ]

            if item.get("method") == "hybrid":
                chips.append(f"vector: {item.get('vector_score', 0):.4f}")
                chips.append(f"bm25: {item.get('bm25_score', 0):.4f}")

            st.markdown(
                " ".join(
                    f'<span class="metric-chip">{chip}</span>'
                    for chip in chips
                ),
                unsafe_allow_html=True,
            )

            st.markdown("#### Nội dung")
            st.write(item.get("display_text", ""))

            with st.expander("Segmented text"):
                st.code(item.get("segmented_text", ""), language="text")

            entities = item.get("entities", [])
            if entities:
                entity_preview = ", ".join(
                    f"{ent.get('text', '')} ({ent.get('type', '')})"
                    for ent in entities[:15]
                )
                st.markdown("#### Thực thể trong bài")
                st.write(entity_preview)


def build_search_payload(
    query: str,
    method: str,
    top_k: int,
    alpha: float,
) -> Dict[str, Any]:
    return {
        "query": query,
        "method": method,
        "top_k": top_k,
        "alpha": alpha,
        "segmenter": st.session_state.segmenter,
    }


def build_entity_search_payload(
    entity_text: str,
    entity_type: str,
    original_text: str,
    method: str,
    top_k: int,
    alpha: float,
) -> Dict[str, Any]:
    return {
        "entity_text": entity_text,
        "entity_type": entity_type,
        "original_text": original_text,
        "method": method,
        "top_k": top_k,
        "alpha": alpha,
        "segmenter": st.session_state.segmenter,
    }


def run_ner(text: str) -> None:
    payload = {
        "text": text,
        "max_length": 256,
        "segmenter": st.session_state.segmenter,
    }

    result = post_json("/ner", payload)

    st.session_state.ner_result = result
    st.session_state.last_ner_text = text
    st.session_state.selected_entity_index = None
    st.session_state.search_result = None
    st.session_state.last_search_payload = None
    st.session_state.last_search_mode = None


def run_direct_search(
    text: str,
    method: str,
    top_k: int,
    alpha: float,
) -> None:
    payload = build_search_payload(text, method, top_k, alpha)
    result = post_json("/search", payload)

    st.session_state.search_result = result
    st.session_state.last_search_payload = payload
    st.session_state.last_search_mode = "direct"


def run_entity_search(
    entity_text: str,
    entity_type: str,
    original_text: str,
    method: str,
    top_k: int,
    alpha: float,
) -> None:
    payload = build_entity_search_payload(
        entity_text,
        entity_type,
        original_text,
        method,
        top_k,
        alpha,
    )
    result = post_json("/entity-search", payload)

    st.session_state.search_result = result
    st.session_state.last_search_payload = payload
    st.session_state.last_search_mode = "entity"


init_state()


st.markdown('<div class="main-title">📰 Vietnamese News Entity Retrieval</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">PhoBERT NER · Vietnamese Embedding · FAISS · BM25 · underthesea / PyVi / VnCoreNLP</div>',
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("API")
    st.caption(API_URL)

    if st.button("Kiểm tra API", use_container_width=True):
        try:
            st.session_state.api_health = get_json("/health")
        except Exception as exc:
            st.session_state.api_health = {"status": "error", "message": str(exc)}

    if st.session_state.api_health:
        health = st.session_state.api_health

        if health.get("status") == "ok":
            st.success("API sẵn sàng")

            device = health.get("device", "unknown")
            device_env = health.get("device_env", "unknown")
            cuda_available = health.get("cuda_available", False)
            mps_available = health.get("mps_available", False)

            st.markdown("**Thiết bị đang dùng**")
            st.code(device, language="text")

            st.caption(
                f"DEVICE env: {device_env} | "
                f"CUDA: {cuda_available} | "
                f"MPS: {mps_available}"
            )
        else:
            st.error("API lỗi")
            st.json(health)

    st.divider()
    st.header("Word segmentation")

    segmenter = st.selectbox(
        "Bộ tách từ",
        ["underthesea", "pyvi", "vncorenlp"],
        index=["underthesea", "pyvi", "vncorenlp"].index(st.session_state.segmenter),
        help="VnCoreNLP cần Java và model VnCoreNLP local. Nếu API báo unavailable, hãy dùng underthesea hoặc pyvi.",
    )

    if segmenter != st.session_state.segmenter:
        st.session_state.segmenter = segmenter
        st.session_state.ner_result = None
        st.session_state.search_result = None
        st.session_state.selected_entity_index = None

    st.divider()

    st.header("Cấu hình truy xuất")

    method = st.radio(
        "Phương pháp",
        ["hybrid", "bm25", "vector"],
        index=0,
        horizontal=True,
        help="Chỉ áp dụng khi bạn bấm nút search. Đổi tham số không tự gọi API.",
    )

    top_k = st.slider(
        "Top K",
        min_value=1,
        max_value=20,
        value=5,
        help="Chỉ áp dụng ở lần search kế tiếp.",
    )

    if method == "hybrid":
        alpha = st.slider(
            "Alpha",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="0.0 = ưu tiên BM25, 1.0 = ưu tiên vector.",
        )
    else:
        alpha = 0.6


default_query = "Bệnh nhân 129 ở Hà Nội từng nhập cảnh qua sân bay Nội Bài."

user_text = st.text_area(
    "Câu truy vấn",
    value=default_query,
    height=115,
    placeholder="Nhập câu tiếng Việt cần nhận diện thực thể và truy xuất tin liên quan...",
)

if user_text != st.session_state.last_ner_text and st.session_state.last_ner_text:
    st.warning("Câu truy vấn đã thay đổi. Bấm 'Nhận diện thực thể' để cập nhật kết quả NER.")


main_actions = st.columns([1, 1, 1])

with main_actions[0]:
    if st.button("Nhận diện thực thể", type="primary", use_container_width=True):
        try:
            run_ner(user_text)
        except Exception as exc:
            st.error(str(exc))

with main_actions[1]:
    if st.button("Search toàn bộ câu", use_container_width=True):
        try:
            run_direct_search(user_text, method, top_k, alpha)
        except Exception as exc:
            st.error(str(exc))

with main_actions[2]:
    if st.button("Xóa kết quả", use_container_width=True):
        reset_ner_and_search()
        st.rerun()


left_col, right_col = st.columns([0.42, 0.58], gap="large")


with left_col:
    st.subheader("Nhận diện thực thể")

    ner_result = st.session_state.ner_result

    if not ner_result:
        st.info("Bấm 'Nhận diện thực thể' để xem kết quả.")
    else:
        st.markdown("**Word segmentation**")
        st.code(ner_result.get("segmented_text", ""), language="text")

        entities = ner_result.get("entities", [])
        word_labels = ner_result.get("word_labels", [])

        st.markdown("**Thực thể tìm được**")
        render_entity_cards(entities)
        render_word_labels(word_labels)

        if entities:
            labels = [entity_display_label(ent) for ent in entities]

            selected_label = st.selectbox(
                "Chọn thực thể để truy xuất",
                labels,
                index=0 if st.session_state.selected_entity_index is None else st.session_state.selected_entity_index,
                help="Không thêm số thứ tự vào entity. Đây chỉ là danh sách thực thể model nhận diện được.",
            )

            selected_index = labels.index(selected_label)
            st.session_state.selected_entity_index = selected_index
            selected_entity = entities[selected_index]

            st.markdown(
                f"""
                <div class="small-muted">
                    Đang chọn: <b>{selected_entity.get("text", "")}</b>
                    · type: <b>{selected_entity.get("type", "")}</b>
                    · segmented: <code>{selected_entity.get("segmented_text", "")}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("Search theo thực thể đã chọn", use_container_width=True):
                try:
                    run_entity_search(
                        entity_text=selected_entity["text"],
                        entity_type=selected_entity.get("type", ""),
                        original_text=user_text,
                        method=method,
                        top_k=top_k,
                        alpha=alpha,
                    )
                except Exception as exc:
                    st.error(str(exc))


with right_col:
    st.subheader("Kết quả truy xuất")

    search_result = st.session_state.search_result

    if not search_result:
        st.info("Chưa có kết quả search.")
    else:
        if st.session_state.last_search_mode == "entity":
            st.markdown("**Final query**")
            st.write(search_result.get("final_query", ""))
        else:
            st.markdown("**Query**")
            st.write(search_result.get("query", ""))

        st.markdown("**Segmenter**")
        st.code(search_result.get("segmenter", st.session_state.segmenter), language="text")

        st.markdown("**Segmented query**")
        st.code(search_result.get("segmented_query", ""), language="text")

        payload = st.session_state.last_search_payload or {}
        st.markdown(
            f"""
            <div class="small-muted">
                method: <b>{payload.get("method", "")}</b>
                · top_k: <b>{payload.get("top_k", "")}</b>
                · alpha: <b>{payload.get("alpha", "")}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        render_results(search_result.get("results", []))
