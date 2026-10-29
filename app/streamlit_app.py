import json

import streamlit as st

from finassist.cli import _record
from finassist.config import ROOT

st.set_page_config(page_title="Finance AI Assistant", layout="wide")
st.title("Finance AI Assistant · AAPL / MSFT / NVDA")
st.caption("Trợ lý tài chính dựa trên báo cáo 10-K, có trích dẫn nguồn.")

replay = st.sidebar.checkbox("Replay recorded run (offline fallback)", value=False)


@st.cache_resource
def get_app():
    from finassist.runtime import build_app

    return build_app()


def _show_prices(out: dict) -> None:
    prices = out.get("stock_prices") or ([out["stock_price"]] if out.get("stock_price") else [])
    if not prices:
        return
    st.subheader("Stock price")
    for p in prices:
        cols = st.columns(4)
        cols[0].metric(p["ticker"], f"{p['price']:,.2f}")
        cols[1].metric("Market cap", f"{p['market_cap']:,.0f}" if p.get("market_cap") is not None else "n/a")
        cols[2].metric("P/E", f"{p['pe_ratio']:.1f}" if p.get("pe_ratio") is not None else "n/a")
        cols[3].write(f"as of {p['as_of']}")
        if p.get("stale"):
            cols[3].markdown(":orange-badge[stale]")


question = st.text_input("Question", "What is NVIDIA's current stock price and how do export controls threaten it?")
if st.button("Ask") and question:
    if replay:
        try:
            rec = json.loads((ROOT / "runs" / "demo.json").read_text())
            rec["output"]["intent"]
        except Exception as e:
            st.error(
                f"Could not load recorded run runs/demo.json ({type(e).__name__}: {e}). "
                'Generate it with `finassist ask "..." --record runs/demo.json`.'
            )
            st.stop()
        st.info(
            f"Replaying a recorded run, not a live one. Recorded question: **{rec.get('question', '?')}**. "
            "The question typed above is ignored."
        )
    else:
        from finassist.runtime import ask

        try:
            with st.spinner("Thinking..."):
                rec = _record(ask(get_app(), question, tags=["demo"]))
        except Exception as e:
            st.error(f"Live run failed: {type(e).__name__}: {e}")
            st.caption("Tick 'Replay recorded run' in the sidebar to use the offline recording.")
            st.stop()
        st.caption(f"Q: {question}")
    out = rec["output"]
    st.caption(f"intent: **{out['intent']}**")
    st.markdown(out["answer"].replace("$", r"\$"))
    if rec.get("error"):
        st.warning(rec["error"])
    _show_prices(out)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Citations")
        cites = out.get("citations") or []
        if cites:
            st.table([{"source": c["source"], "item": c["item"], "chunk_id": c["chunk_id"], "quote": c["quote"]} for c in cites])
        else:
            st.caption("No citations.")
        st.subheader("Retrieved chunks")
        for r in rec.get("retrieved") or []:
            label = r["chunk_id"] if r.get("score") is None else f"{r['chunk_id']}  (distance {r['score']:.3f})"
            with st.expander(label):
                st.text(r["text"])
    with c2:
        st.subheader("Structured output (JSON)")
        st.json(out)
