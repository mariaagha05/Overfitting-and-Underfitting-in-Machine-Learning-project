"""Bias-Variance Lab: underfitting, overfitting, loss curves and fixes (Streamlit + Groq)."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split, validation_curve
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# ---------- Page + theme (forced high-contrast so text is readable on any device) ----------
st.set_page_config(page_title="Bias-Variance Lab", page_icon="📈", layout="wide")

INK, PAPER, PANEL, EDGE = "#F8FAFC", "#0B1220", "#162033", "#334155"
C_TRAIN, C_VAL, C_FIX, C_TRUTH = "#38BDF8", "#FB923C", "#4ADE80", "#CBD5E1"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Inter:wght@400;500;600&display=swap');
html, body, .stApp, [data-testid="stAppViewContainer"] {{ background:{PAPER} !important; }}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {{ background:{PANEL} !important; }}
[data-testid="stHeader"] {{ background:transparent !important; }}
.stApp, .stApp p, .stApp span, .stApp label, .stApp li, .stApp div, .stApp small,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stMarkdown, [data-testid="stMetricValue"],
[data-testid="stMetricLabel"], [data-testid="stCaptionContainer"], button[data-baseweb="tab"] p
{{ color:{INK} !important; font-family:'Inter', system-ui, sans-serif; }}
h1, h2, h3 {{ font-family:'Fraunces', Georgia, serif !important; letter-spacing:-0.01em; }}
input, textarea {{ color:#0F172A !important; background:#F1F5F9 !important; }}
div[data-baseweb="select"] * {{ color:#0F172A !important; }}
div[data-baseweb="select"] > div {{ background:#F1F5F9 !important; }}
.stButton > button {{ background:{C_FIX} !important; color:#052E16 !important; font-weight:600;
  border:0; border-radius:10px; padding:.6rem 1.2rem; }}
.stButton > button p {{ color:#052E16 !important; }}
[data-testid="stMetric"] {{ background:{PANEL}; border:1px solid {EDGE}; border-radius:12px; padding:12px 16px; }}
button[data-baseweb="tab"] {{ font-weight:600; }}
.note {{ background:{PANEL}; border-left:4px solid {C_VAL}; padding:12px 16px; border-radius:6px; }}
.note, .note * {{ color:{INK} !important; }}
.block-container {{ padding-top:2rem; max-width:1200px; }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
def style(fig, title, h=420):
    fig.update_layout(
        title=dict(text=title, font=dict(color=INK, size=18)),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, height=h,
        font=dict(color=INK, size=14),
        legend=dict(font=dict(color=INK), orientation="h", y=-0.22),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(gridcolor=EDGE, zeroline=False, tickfont=dict(color=INK), title_font=dict(color=INK))
    fig.update_yaxes(gridcolor=EDGE, zeroline=False, tickfont=dict(color=INK), title_font=dict(color=INK))
    return fig


def make_pipe(degree=1, alpha=1e-8):
    return Pipeline([
        ("poly", PolynomialFeatures(degree, include_bias=False)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


@st.cache_data(show_spinner=False)
def get_data(n, noise, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + rng.normal(0, noise, n)
    return train_test_split(x.reshape(-1, 1), y, test_size=0.3, random_state=seed)


@st.cache_data(show_spinner=False)
def tune(n, noise, seed):
    Xtr, _, ytr, _ = get_data(n, noise, seed)
    grid = GridSearchCV(
        make_pipe(),
        {"poly__degree": list(range(1, 16)), "model__alpha": [1e-4, 1e-3, 1e-2, 0.1, 1, 10]},
        cv=5, scoring="neg_mean_squared_error",
    ).fit(Xtr, ytr)
    return int(grid.best_params_["poly__degree"]), float(grid.best_params_["model__alpha"]), float(-grid.best_score_)


def evaluate(model, Xtr, Xte, ytr, yte):
    model.fit(Xtr, ytr)
    return (mean_squared_error(ytr, model.predict(Xtr)), mean_squared_error(yte, model.predict(Xte)))


def fit_plot(model, Xtr, Xte, ytr, yte, title, color):
    xs = np.linspace(0, 1, 300).reshape(-1, 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs[:, 0], y=np.sin(2 * np.pi * xs[:, 0]), name="True pattern",
                             line=dict(color=C_TRUTH, dash="dot", width=2)))
    fig.add_trace(go.Scatter(x=Xtr[:, 0], y=ytr, mode="markers", name="Train data",
                             marker=dict(color=C_TRAIN, size=9)))
    fig.add_trace(go.Scatter(x=Xte[:, 0], y=yte, mode="markers", name="Test data",
                             marker=dict(color=C_VAL, size=9, symbol="diamond")))
    fig.add_trace(go.Scatter(x=xs[:, 0], y=np.clip(model.predict(xs), -3, 3), name="Model",
                             line=dict(color=color, width=4)))
    fig.update_yaxes(range=[-2.5, 2.5], title="y")
    fig.update_xaxes(title="x")
    return style(fig, title)


def metric_row(tr, te):
    a, b, c = st.columns(3)
    a.metric("Train error (MSE)", f"{tr:.3f}")
    b.metric("Test error (MSE)", f"{te:.3f}")
    c.metric("Gap (test ÷ train)", f"{te / max(tr, 1e-9):.1f}×")


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Controls")
    n = st.slider("Number of samples", 30, 200, 60, 5)
    noise = st.slider("Noise level", 0.05, 0.60, 0.25, 0.05)
    seed = st.number_input("Random seed", 0, 9999, 42, 1)
    st.divider()
    st.subheader("Groq AI tutor")
    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        secret_key = ""
    api_key = st.text_input("Groq API key", value=secret_key, type="password",
                            help="Free key at console.groq.com. Or set GROQ_API_KEY in .streamlit/secrets.toml")
    llm = st.selectbox("Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])

# ---------- Compute everything ----------
Xtr, Xte, ytr, yte = get_data(n, noise, int(seed))
under = make_pipe(1)
over = make_pipe(15)
u_tr, u_te = evaluate(under, Xtr, Xte, ytr, yte)
o_tr, o_te = evaluate(over, Xtr, Xte, ytr, yte)
best_deg, best_alpha, best_cv = tune(n, noise, int(seed))
best = make_pipe(best_deg, best_alpha)
b_tr, b_te = evaluate(best, Xtr, Xte, ytr, yte)

# ---------- Header ----------
st.title("Bias-Variance Lab")
st.write("Fit a curve to noisy data and watch a model go from too simple, to too clever, to just right.")

tabs = st.tabs(["1 Underfitting", "2 Overfitting", "3 Loss curves", "4 The fix", "5 AI tutor"])

with tabs[0]:
    st.subheader("Underfitting: a straight line can't bend")
    st.write("A degree-1 model has high bias. It misses the wave, so error is high on training and test data alike.")
    metric_row(u_tr, u_te)
    st.plotly_chart(fit_plot(under, Xtr, Xte, ytr, yte, "Degree 1 model", C_FIX), use_container_width=True)

with tabs[1]:
    st.subheader("Overfitting: a degree-15 model memorises the noise")
    st.write("It threads through every training point, so train error is tiny while test error jumps. That gap is high variance.")
    metric_row(o_tr, o_te)
    st.plotly_chart(fit_plot(over, Xtr, Xte, ytr, yte, "Degree 15 model, no real regularisation", "#F472B6"),
                    use_container_width=True)

with tabs[2]:
    st.subheader("Training vs validation loss")
    st.write("Each point is a model of different complexity, scored with 5-fold cross-validation on the training set only.")
    degrees = list(range(1, 16))
    tr_s, va_s = validation_curve(make_pipe(), Xtr, ytr, param_name="poly__degree", param_range=degrees,
                                  cv=5, scoring="neg_mean_squared_error")
    tr_loss, va_loss = -tr_s.mean(axis=1), -va_s.mean(axis=1)
    sweet = degrees[int(np.argmin(va_loss))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=degrees, y=tr_loss, name="Training loss", mode="lines+markers",
                             line=dict(color=C_TRAIN, width=3)))
    fig.add_trace(go.Scatter(x=degrees, y=va_loss, name="Validation loss", mode="lines+markers",
                             line=dict(color=C_VAL, width=3)))
    fig.add_vline(x=sweet, line_dash="dash", line_color=C_FIX,
                  annotation_text=f"best degree = {sweet}", annotation_font_color=INK)
    fig.update_xaxes(title="Model complexity (polynomial degree)", dtick=1)
    fig.update_yaxes(title="Loss (MSE, log scale)", type="log")
    st.plotly_chart(style(fig, "Loss vs model complexity"), use_container_width=True)
    st.markdown(
        f"""<div class="note"><b>How to read it.</b> Both lines high on the left means underfitting.
        Training loss falling while validation loss climbs on the right means overfitting.
        The lowest validation loss (degree {sweet}) is the sweet spot.</div>""",
        unsafe_allow_html=True,
    )

with tabs[3]:
    st.subheader("Fixing the model with cross-validation, tuning and regularisation")
    st.write("Grid search tries degrees 1 to 15 and Ridge penalties from 0.0001 to 10, scoring each with 5-fold CV. "
             "The test set stays untouched until the final check.")
    a, b = st.columns(2)
    a.metric("Chosen degree", best_deg)
    b.metric("Chosen Ridge alpha", f"{best_alpha:g}")
    metric_row(b_tr, b_te)
    st.plotly_chart(fit_plot(best, Xtr, Xte, ytr, yte, f"Tuned model (degree {best_deg}, alpha {best_alpha:g})", C_FIX),
                    use_container_width=True)
    names = ["Underfit (deg 1)", "Overfit (deg 15)", "Tuned"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[u_tr, o_tr, b_tr], name="Train", marker_color=C_TRAIN))
    fig.add_trace(go.Bar(x=names, y=[u_te, o_te, b_te], name="Test", marker_color=C_VAL))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="MSE (log scale)", type="log")
    st.plotly_chart(style(fig, "Train vs test error, all three models", 380), use_container_width=True)

with tabs[4]:
    st.subheader("Ask the AI tutor")
    st.write("Groq's LLM reads your actual results and explains them in plain language.")
    question = st.text_area("Your question", "Explain what these results say about bias and variance, and what I should try next.")
    if st.button("Ask the tutor"):
        if not api_key.strip():
            st.warning("Add your Groq API key in the sidebar first.")
        else:
            facts = (
                f"Data: {n} samples of y=sin(2*pi*x)+noise (noise sd {noise}). 70/30 train/test split.\n"
                f"Underfit degree 1: train MSE {u_tr:.4f}, test MSE {u_te:.4f}.\n"
                f"Overfit degree 15: train MSE {o_tr:.4f}, test MSE {o_te:.4f}.\n"
                f"Tuned (GridSearchCV, 5-fold, Ridge): degree {best_deg}, alpha {best_alpha:g}, "
                f"CV MSE {best_cv:.4f}, train MSE {b_tr:.4f}, test MSE {b_te:.4f}."
            )
            try:
                from groq import Groq

                with st.spinner("Thinking..."):
                    reply = Groq(api_key=api_key.strip()).chat.completions.create(
                        model=llm,
                        temperature=0.4,
                        max_tokens=700,
                        messages=[
                            {"role": "system", "content": "You are a friendly machine learning tutor. Use only the "
                             "numbers given. Be concise, use short paragraphs, and end with 2 concrete next steps."},
                            {"role": "user", "content": f"{facts}\n\nQuestion: {question}"},
                        ],
                    )
                st.markdown(f'<div class="note">{reply.choices[0].message.content}</div>'.replace("\n", "<br>"),
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Groq request failed: {e}. Check your API key, model name and internet connection.")
