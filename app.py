import io
import os
import sys
import warnings

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, QED
from streamlit_ketcher import st_ketcher

# Runtime setup
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "model_v2.ckpt"
)

# ─── Task definitions ─────────────────────────────────────────────────────────

TASKS = {
    "De Novo": {
        "description": (
            "Generate drug-like molecules from scratch, with no input constraints. "
            "The model freely explores chemical space."
        ),
        "num_fragments": 0,
        "defaults": {
            "softmax_temp": 1.0,
            "randomness": 0.3,
            "gamma": 0.0,
            "min_add_len": 60,
        },
    },
    "Motif Extension": {
        "description": (
            "Grow a molecule around a given fragment. "
            "Add `[*]` to mark each open attachment point."
        ),
        "num_fragments": 1,
        "defaults": {
            "softmax_temp": 1.2,
            "randomness": 1.0,
            "gamma": 0.3,
            "min_add_len": 24,
        },
    },
    "Scaffold Decoration": {
        "description": (
            "Add side chains to a scaffold. "
            "No attachment point notation is needed — the model discovers them."
        ),
        "num_fragments": 1,
        "defaults": {
            "softmax_temp": 1.5,
            "randomness": 2.0,
            "gamma": 0.3,
            "min_add_len": 24,
        },
    },
    "Superstructure Generation": {
        "description": (
            "Generate larger molecules that contain your fragment as a strict substructure. "
            "Draw the fragment without attachment points."
        ),
        "num_fragments": 1,
        "defaults": {
            "softmax_temp": 1.5,
            "randomness": 2.0,
            "gamma": 0.4,
            "min_add_len": 24,
        },
    },
    "Linker Design (1-step)": {
        "description": (
            "Connect two fragments with a linker in a single generation pass. "
            "Recommended with the V2 model. "
            "Each fragment should have a `[*]` attachment point."
        ),
        "num_fragments": 2,
        "defaults": {
            "softmax_temp": 1.2,
            "randomness": 3.0,
            "gamma": 0.0,
            "min_add_len": 30,
        },
    },
    "Linker Design (2-step)": {
        "description": (
            "Connect two fragments with a linker via two separate generation passes, "
            "then combine. Each fragment should have a `[*]` attachment point."
        ),
        "num_fragments": 2,
        "defaults": {
            "softmax_temp": 1.2,
            "randomness": 3.0,
            "gamma": 0.0,
            "min_add_len": 30,
        },
    },
}

# ─── Cached model loader ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading GenMol model…")
def load_sampler():
    from genmol.sampler import Sampler
    return Sampler(CHECKPOINT_PATH)


# ─── Generation dispatcher ────────────────────────────────────────────────────

def run_generation(task, fragment1, fragment2, params):
    sampler = load_sampler()

    if task == "De Novo":
        return sampler.de_novo_generation(
            num_samples=params["num_samples"],
            softmax_temp=params["softmax_temp"],
            randomness=params["randomness"],
            min_add_len=params["min_add_len"],
        )

    if task == "Linker Design (1-step)":
        fragment = f"{fragment1}.{fragment2}"
        return sampler.fragment_linking_onestep(
            fragment=fragment,
            num_samples=params["num_samples"],
            softmax_temp=params["softmax_temp"],
            randomness=params["randomness"],
            gamma=params["gamma"],
            min_add_len=params["min_add_len"],
        )

    if task == "Linker Design (2-step)":
        fragment = f"{fragment1}.{fragment2}"
        return sampler.fragment_linking(
            fragment=fragment,
            num_samples=params["num_samples"],
            softmax_temp=params["softmax_temp"],
            randomness=params["randomness"],
            gamma=params["gamma"],
            min_add_len=params["min_add_len"],
        )

    # All fragment_completion tasks
    return sampler.fragment_completion(
        fragment=fragment1,
        num_samples=params["num_samples"],
        apply_filter=True,
        softmax_temp=params["softmax_temp"],
        randomness=params["randomness"],
        gamma=params["gamma"],
    )


# ─── Chemistry helpers ────────────────────────────────────────────────────────

def compute_properties(smiles_list):
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        rows.append(
            {
                "smiles": smi,
                "qed": round(QED.qed(mol), 3),
                "mw": round(Descriptors.MolWt(mol), 1),
                "logp": round(Descriptors.MolLogP(mol), 2),
                "hbd": int(Descriptors.NumHDonors(mol)),
                "hba": int(Descriptors.NumHAcceptors(mol)),
            }
        )
    return pd.DataFrame(rows)


def smiles_to_png(smiles, size=(220, 160)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def fragment_input(label, key, hint=""):
    """Render a Ketcher drawer + editable SMILES box; return the final SMILES string."""
    method = st.radio(
        f"{label} — input method",
        ["Draw", "Enter SMILES"],
        horizontal=True,
        key=f"method_{key}",
    )
    if method == "Draw":
        drawn = st_ketcher(key=key)
        # Let the user refine the Ketcher output (e.g. to add [*])
        smiles = st.text_input(
            "Edit SMILES (add `[*]` for attachment points)",
            value=drawn or "",
            placeholder=hint,
            key=f"edit_{key}",
            help=(
                "Insert [*] at any position in the SMILES to mark where the linker "
                "or extension should attach. Example: change `CCCCC` to `[*]CCCCC`."
            ),
        )
    else:
        smiles = st.text_input(
            f"{label} SMILES",
            placeholder=hint,
            key=f"text_{key}",
        )

    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            png = smiles_to_png(smiles, size=(300, 200))
            st.image(png, caption=smiles)
        else:
            st.error("Invalid SMILES — please fix before generating.")
            smiles = ""
    return smiles or ""


# ─── Page layout ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="GenMol", layout="wide")
st.title("GenMol — Molecule Generator")
st.caption("Discrete diffusion model for drug-like molecule generation")

# Task picker
task = st.selectbox("Generation task", list(TASKS.keys()))
task_cfg = TASKS[task]
st.info(task_cfg["description"])
defaults = task_cfg["defaults"]

# ─── Sidebar: parameters ──────────────────────────────────────────────────────

with st.sidebar:
    st.header("Generation parameters")

    num_samples = st.number_input(
        "Number of molecules", min_value=1, max_value=500, value=10
    )
    softmax_temp = st.slider(
        "Temperature", 0.5, 2.0, float(defaults["softmax_temp"]), 0.05,
        help="Higher → more diverse; lower → more focused."
    )
    randomness = st.slider(
        "Randomness", 0.0, 5.0, float(defaults["randomness"]), 0.1,
        help="Variance in token sampling."
    )
    gamma = st.slider(
        "Context masking γ", 0.0, 1.0, float(defaults["gamma"]), 0.05,
        help="Fraction of context tokens masked for classifier-free guidance. 0 = off."
    )

    # min_add_len only makes sense for tasks that use it
    show_min_add = task != "Scaffold Decoration" and "Superstructure" not in task
    if show_min_add:
        min_add_len = st.number_input(
            "Min tokens to add", min_value=10, max_value=120,
            value=int(defaults["min_add_len"]),
            help="Minimum number of masked tokens inserted before generation."
        )
    else:
        min_add_len = int(defaults["min_add_len"])

# ─── Fragment input area ──────────────────────────────────────────────────────

fragment1, fragment2 = "", ""

if task_cfg["num_fragments"] == 0:
    st.info("No input needed — the model generates molecules from scratch.")

elif task_cfg["num_fragments"] == 1:
    st.subheader("Input fragment")
    if task == "Motif Extension":
        st.caption(
            "Draw or type a SMILES with `[*]` marking open attachment points, "
            "e.g. `[*]c1ccccc1` or `[*]c1cccnc1[*]`."
        )
        fragment1 = fragment_input("Fragment", "frag1", hint="[*]c1ccccc1")
        if fragment1 and "[*]" not in fragment1:
            st.error("Fragment is missing `[*]`. Add at least one attachment point, e.g. `[*]c1ccccc1`.")
    elif task == "Scaffold Decoration":
        st.caption(
            "Draw or type a scaffold SMILES (no `[*]` needed). "
            "The model will add side chains at chemically sensible positions."
        )
        fragment1 = fragment_input("Scaffold", "frag1", hint="c1ccc2ncccc2c1")
    else:  # Superstructure
        st.caption(
            "Draw or type a fragment SMILES. "
            "The model will generate larger molecules that contain it as a substructure."
        )
        fragment1 = fragment_input("Fragment", "frag1", hint="c1ccccc1")

else:  # 2 fragments
    st.subheader("Fragments to link")
    st.warning(
        "Each fragment **must** contain `[*]` to mark the attachment point for the linker. "
        "Without it the model cannot extract linker candidates and will return nothing. "
        "Example: `[*]c1cccnc1` (pyridine side chain).",
        icon="⚠️",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Fragment 1**")
        fragment1 = fragment_input(
            "Fragment 1", "frag1", hint="[*]c1ccc2c(c1)cccc2"
        )
        if fragment1 and "[*]" not in fragment1:
            st.error("Fragment 1 is missing `[*]`. Add an attachment point, e.g. `[*]CCCCC(CCC)C`.")
    with col_b:
        st.markdown("**Fragment 2**")
        fragment2 = fragment_input(
            "Fragment 2", "frag2", hint="[*]CCO"
        )
        if fragment2 and "[*]" not in fragment2:
            st.error("Fragment 2 is missing `[*]`. Add an attachment point, e.g. `[*]c1cnc[nH]1`.")
    if fragment1 and fragment2:
        st.caption(f"Combined SMILES passed to the model: `{fragment1}.{fragment2}`")

# ─── Generate ─────────────────────────────────────────────────────────────────

st.divider()
generate_btn = st.button("Generate molecules", type="primary", use_container_width=True)

if generate_btn:
    # Validate inputs
    error = None
    if task_cfg["num_fragments"] >= 1:
        if not fragment1:
            error = "Please provide Fragment 1 before generating."
        elif Chem.MolFromSmiles(fragment1) is None:
            error = f"Fragment 1 is not a valid SMILES: `{fragment1}`"
        elif task == "Motif Extension" and "[*]" not in fragment1:
            error = (
                "Motif Extension requires at least one `[*]` attachment point in the fragment. "
                "Example: `[*]c1ccccc1`."
            )
    if task_cfg["num_fragments"] == 2 and error is None:
        if not fragment2:
            error = "Please provide Fragment 2 before generating."
        elif Chem.MolFromSmiles(fragment2) is None:
            error = f"Fragment 2 is not a valid SMILES: `{fragment2}`"
        elif "[*]" not in fragment1:
            error = (
                "Fragment 1 is missing `[*]`. "
                "The linker design pipeline needs an explicit attachment point on each fragment."
            )
        elif "[*]" not in fragment2:
            error = (
                "Fragment 2 is missing `[*]`. "
                "The linker design pipeline needs an explicit attachment point on each fragment."
            )

    if error:
        st.error(error)
    else:
        params = dict(
            num_samples=num_samples,
            softmax_temp=softmax_temp,
            randomness=randomness,
            gamma=gamma,
            min_add_len=min_add_len,
        )

        with st.spinner(f"Generating {num_samples} molecules…"):
            try:
                results = run_generation(task, fragment1, fragment2, params)
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                st.exception(exc)
                results = []

        if results:
            df = compute_properties(results)

            # Summary metrics
            st.subheader("Results")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Requested", num_samples)
            c2.metric("Valid", len(df))
            c3.metric("Unique", int(df["smiles"].nunique()))
            avg_qed = f"{df['qed'].mean():.3f}" if len(df) else "—"
            c4.metric("Avg QED", avg_qed)

            # Molecule gallery (up to 20)
            st.subheader("Gallery")
            display = df.head(20)
            n_cols = 4
            for row_start in range(0, len(display), n_cols):
                row_slice = display.iloc[row_start : row_start + n_cols]
                cols = st.columns(n_cols)
                for j, (_, row) in enumerate(row_slice.iterrows()):
                    with cols[j]:
                        png = smiles_to_png(row["smiles"])
                        if png:
                            st.image(png)
                        st.caption(
                            f"QED={row['qed']}  MW={row['mw']}  logP={row['logp']}"
                        )
                        with st.expander("SMILES"):
                            st.code(row["smiles"], language="text")

            # Full data table
            st.subheader("Data table")
            st.dataframe(df, use_container_width=True)

            # Download
            st.download_button(
                "Download CSV",
                df.to_csv(index=False),
                file_name="genmol_results.csv",
                mime="text/csv",
            )
        else:
            st.warning(
                "No valid molecules were generated. "
                "Try increasing the number of samples or adjusting the parameters."
            )
