"""Live demo: classify an in-game chat message as neutral / negative / derogatory.

Run from the project root:
    streamlit run app/streamlit_app.py

Uses the models trained on the deduplicated data (models/dedup/). Each model is
loaded only when selected and then cached, so the app starts fast on a laptop.
"""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `import src...` works when run via streamlit

from src.config import LABELS, RESULTS_DIR  # noqa: E402
from src.labeling import weak_label  # noqa: E402
from src.model_io import list_bundles, load_bundle  # noqa: E402
from src.plot_style import CLASS_COLORS  # noqa: E402
from src.text_utils import normalize_text  # noqa: E402

DEFAULT_MODEL = "logreg_wordchar"
EXAMPLES = ["ggwp thanks team", "my bad, i missed the stun", "you're bad",
            "gg ez noob team", "report mid useless feeder", "shut up bitch"]
CLASS_EMOJI = {0: "🟦", 1: "🟧", 2: "🟩"}  # same colours as the charts in results/figures

st.set_page_config(page_title="PvP Chat Toxicity", page_icon="🎮", layout="centered")


@st.cache_data
def model_table() -> pd.DataFrame:
    """Every saved model with its test macro F1, best first."""
    paths = {p.stem: p for p in list_bundles("dedup")}
    summary_path = RESULTS_DIR / "summary.csv"
    if summary_path.exists():
        s = pd.read_csv(summary_path)
        s = s[(s["dataset"] == "dedup") & s["key"].isin(paths)]
        s = s[["key", "model", "task", "test_macro_f1"]]
    else:  # evaluate.py hasn't been run: still list the models, without scores
        s = pd.DataFrame({"key": list(paths), "model": list(paths), "task": "?",
                          "test_macro_f1": float("nan")})
    s["path"] = s["key"].map(lambda k: str(paths[k]))
    return s.sort_values("test_macro_f1", ascending=False).reset_index(drop=True)


@st.cache_resource(show_spinner="Loading model ...")
def get_model(path: str):
    return load_bundle(Path(path))


def predict(bundle, text: str) -> tuple[int, list[float] | None]:
    norm = normalize_text(text)  # exactly the preprocessing used in training
    pred = int(bundle.predict([norm])[0])
    proba = bundle.predict_proba([norm])
    return pred, (None if proba is None else proba[0].tolist())


def class_names(bundle) -> dict[int, str]:
    return dict(enumerate(bundle.class_names))


def proba_chart(names: dict[int, str], proba: list[float]) -> alt.Chart:
    df = pd.DataFrame({"class": [names[i] for i in names], "probability": proba,
                       "id": list(names)})
    colors = [CLASS_COLORS[i] for i in names]
    base = alt.Chart(df).encode(
        y=alt.Y("class:N", sort=list(df["class"]), title=None),
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"),
                title="probability"),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=26).encode(
        color=alt.Color("class:N", scale=alt.Scale(domain=list(df["class"]), range=colors),
                        legend=None),
        tooltip=["class", alt.Tooltip("probability:Q", format=".1%")],
    )
    labels = base.mark_text(align="left", dx=5).encode(
        text=alt.Text("probability:Q", format=".1%"))
    return (bars + labels).properties(height=40 * len(df))


# ------------------------------------------------------------------ UI
st.title("🎮 PvP chat toxicity classifier")
st.caption("Classifies a League of Legends / Dota 2 chat message as **neutral/positive**, "
           "**negative attitude** or **derogatory**.")

models = model_table()
if models.empty:
    st.error("No trained models found in models/dedup/. Run `python -m src.train_ml` "
             "(and optionally `python -m src.train_nn`) first.")
    st.stop()

with st.sidebar:
    st.header("Model")
    labels = [f"{r.model}  ·  F1 {r.test_macro_f1:.3f}" if pd.notna(r.test_macro_f1)
              else r.model for r in models.itertuples()]
    default = int(models.index[models["key"] == DEFAULT_MODEL][0]) \
        if DEFAULT_MODEL in set(models["key"]) else 0
    choice = st.selectbox("Choose a trained model", range(len(models)), index=default,
                          format_func=lambda i: labels[i])
    row = models.iloc[choice]
    st.caption("Sorted by macro F1 on the deduplicated test set. Binary SVMs only predict "
               "neutral vs toxic and give no probabilities.")
    st.divider()
    st.markdown("**Classes**\n\n"
                f"{CLASS_EMOJI[0]} 0 neutral/positive: *ggwp, thanks, nice*\n\n"
                f"{CLASS_EMOJI[1]} 1 negative: *noob team, such a feeder*\n\n"
                f"{CLASS_EMOJI[2]} 2 derogatory: slurs, identity-based insults")

if "message" not in st.session_state:
    st.session_state.message = EXAMPLES[3]
st.write("Try an example:")
cols = st.columns(3)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 3].button(ex, width="stretch"):
        st.session_state.message = ex

text = st.text_input("Chat message", key="message", placeholder="type a chat message ...")

if text.strip():
    bundle = get_model(row["path"])
    pred, proba = predict(bundle, text)
    names = class_names(bundle)
    color_id = pred if bundle.task != "binary" else (0 if pred == 0 else 1)

    st.subheader(f"{CLASS_EMOJI[color_id]} {names[pred]}")
    if proba is not None:
        st.altair_chart(proba_chart(names, proba), width="stretch")
    else:
        st.info("This model (SVM) outputs a decision only, no probabilities.")

    rule = weak_label(normalize_text(text))
    with st.expander("What happened under the hood"):
        st.markdown(f"- **Normalised text** (what the model sees): `{normalize_text(text)}`\n"
                    f"- **Rule-based weak label** (what we trained on): "
                    f"{rule.label} = {LABELS[rule.label]}  (rule: `{rule.rule}`)\n"
                    f"- **Model**: {bundle.name}, features: `{bundle.features}`")
        if bundle.task == "multiclass" and rule.label != pred:
            st.warning("The model and the lexicon rules disagree on this message.")

    with st.expander("Compare all models on this message"):
        if st.button("Run every model (first run loads all models, a few seconds)"):
            rows = []
            for r in models.itertuples():
                b = get_model(r.path)
                p, pr = predict(b, text)
                rows.append({"model": r.model, "prediction": class_names(b)[p],
                             "confidence": None if pr is None else max(pr),
                             "test macro F1": r.test_macro_f1})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={
                             "confidence": st.column_config.NumberColumn(format="%.2f"),
                             "test macro F1": st.column_config.NumberColumn(format="%.3f")})

st.divider()
st.caption("Labels come from lexicon rules, not human annotators, so predictions reflect "
           "those rules. Models: trained on deduplicated LoL Tribunal + GOSU.AI Dota 2 chat.")
