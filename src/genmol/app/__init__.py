"""genmol.app — helper functions for the GenMol Colab notebook.

Keeps the notebook cells minimal: all chemistry, display logic, and
chemical-space visualisation live here.
"""

import base64
import io
import re
import warnings

import numpy as np
import pandas as pd
from IPython.display import HTML, display
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, QED

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Molecule properties
# ---------------------------------------------------------------------------

def compute_properties(smiles_list):
    """Return a DataFrame of physicochemical properties for valid SMILES."""
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        rows.append({
            "SMILES": smi,
            "QED":    round(QED.qed(mol), 3),
            "MW":     round(Descriptors.MolWt(mol), 1),
            "LogP":   round(Descriptors.MolLogP(mol), 2),
            "HBD":    int(Descriptors.NumHDonors(mol)),
            "HBA":    int(Descriptors.NumHAcceptors(mol)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Molecule rendering
# ---------------------------------------------------------------------------

def _mol_to_png_b64(smiles, size=(200, 150)):
    """Render a SMILES string to a base64-encoded PNG string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _mol_to_svg(smiles, size=(200, 150)):
    """Render a SMILES string to an SVG string (for Plotly tooltips)."""
    from rdkit.Chem.Draw import rdMolDraw2D
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def display_results(smiles_list):
    """Display generated molecules as an HTML table (one row per molecule).

    Each row shows the 2D structure, SMILES, QED, MW, LogP, HBD, HBA.
    Also stores results in the module-level `generated_smiles` list so the
    chemical-space cell can pick them up without re-running generation.
    """
    global generated_smiles

    df = compute_properties(smiles_list)
    if df.empty:
        print("No valid molecules were generated.")
        return

    generated_smiles = df["SMILES"].tolist()

    rows_html = []
    for _, row in df.iterrows():
        b64 = _mol_to_png_b64(row["SMILES"])
        img_tag = (
            f'<img src="data:image/png;base64,{b64}" style="width:160px"/>'
            if b64 else "—"
        )
        rows_html.append(f"""
        <tr>
          <td style="text-align:center;vertical-align:middle">{img_tag}</td>
          <td style="font-family:monospace;font-size:11px;vertical-align:middle;max-width:200px;word-break:break-all">{row['SMILES']}</td>
          <td style="text-align:center;vertical-align:middle">{row['QED']}</td>
          <td style="text-align:center;vertical-align:middle">{row['MW']}</td>
          <td style="text-align:center;vertical-align:middle">{row['LogP']}</td>
          <td style="text-align:center;vertical-align:middle">{row['HBD']}</td>
          <td style="text-align:center;vertical-align:middle">{row['HBA']}</td>
        </tr>""")

    header = """
    <tr style="background:#f0f0f0">
      <th>Structure</th><th>SMILES</th>
      <th>QED</th><th>MW</th><th>LogP</th><th>HBD</th><th>HBA</th>
    </tr>"""

    table = f"""
    <div style="max-height:600px;overflow-y:auto">
    <table style="border-collapse:collapse;width:100%;font-size:13px">
      <thead>{header}</thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
    </div>"""

    print(f"{len(df)} valid molecules generated.")
    display(HTML(table))


# Global store so the chemical-space cell can access the last batch.
generated_smiles = []


# ---------------------------------------------------------------------------
# SMILES normalisation for fragment inputs
# ---------------------------------------------------------------------------

def clean_fragment(smiles):
    """Normalise R-group notation from JSME to standard [*] attachment points."""
    return re.sub(r'\[R\d*\]', '[*]', smiles).replace('*', '[*]').replace('[[*]]', '[*]')


# ---------------------------------------------------------------------------
# Chemical space — PCA on ECFP4 fingerprints
# ---------------------------------------------------------------------------

def _ecfp4(smiles_list, radius=2, n_bits=2048):
    """Return a (N, n_bits) numpy array of ECFP4 fingerprints."""
    from rdkit.Chem import AllChem
    fps, valid_smi = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        arr = np.zeros(n_bits, dtype=np.uint8)
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(bv, arr)
        fps.append(arr)
        valid_smi.append(smi)
    return np.array(fps), valid_smi


def _nearest_ref_smiles(gen_fp, ref_fps, ref_smi_list, k=3):
    """Return the k reference SMILES nearest to gen_fp by Tanimoto similarity."""
    gen_fp = gen_fp.astype(float)
    ref_fps_f = ref_fps.astype(float)
    dot = ref_fps_f @ gen_fp
    sims = dot / (ref_fps_f.sum(axis=1) + gen_fp.sum() - dot + 1e-9)
    top_k = np.argsort(sims)[::-1][:k]
    return [ref_smi_list[i] for i in top_k]


def _mol_grid_html(smiles_list, size=(160, 120), label=None):
    """Return an HTML string showing molecule images side by side."""
    imgs = []
    for smi in smiles_list:
        b64 = _mol_to_png_b64(smi, size=size)
        if b64:
            imgs.append(
                f'<div style="display:inline-block;text-align:center;margin:4px">'
                f'<img src="data:image/png;base64,{b64}" style="width:{size[0]}px"/>'
                f'<div style="font-size:10px;font-family:monospace;max-width:{size[0]}px;'
                f'word-break:break-all">{smi}</div></div>'
            )
    header = f'<div style="font-weight:bold;margin-bottom:4px">{label}</div>' if label else ""
    return header + '<div style="display:flex;flex-wrap:wrap">' + "".join(imgs) + "</div>"


def plot_chemical_space(reference_smiles, generated_smiles, n_ref_display=3000):
    """Plot a 2-D PCA of ECFP4 fingerprints.

    Click any point to see its 2D structure and (for generated molecules) the
    3 nearest PubChem neighbours rendered below the plot.

    Args:
        reference_smiles:  List of reference SMILES (e.g. a PubChem sample).
        generated_smiles:  List of newly generated SMILES.
        n_ref_display:     Max reference molecules shown (for performance).
    """
    import plotly.graph_objects as go
    from ipywidgets import Output
    from IPython.display import clear_output
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if not generated_smiles:
        print("No generated molecules to plot. Run a generation task first.")
        return

    rng = np.random.default_rng(42)
    if len(reference_smiles) > n_ref_display:
        idx = rng.choice(len(reference_smiles), n_ref_display, replace=False)
        ref_display = [reference_smiles[i] for i in idx]
    else:
        ref_display = list(reference_smiles)

    ref_fps, ref_valid = _ecfp4(ref_display)
    gen_fps, gen_smi  = _ecfp4(list(generated_smiles))

    if len(ref_fps) + len(gen_fps) < 3 or len(gen_fps) == 0:
        print("Not enough valid molecules to compute PCA.")
        return

    all_fps = np.vstack([ref_fps, gen_fps])
    n_ref   = len(ref_fps)

    coords = PCA(n_components=2).fit_transform(
        StandardScaler(with_std=False).fit_transform(all_fps)
    )
    ref_coords = coords[:n_ref]
    gen_coords = coords[n_ref:]

    gen_props = compute_properties(gen_smi)

    # Build figure
    fig = go.FigureWidget()

    fig.add_trace(go.Scatter(
        x=ref_coords[:, 0], y=ref_coords[:, 1],
        mode="markers",
        marker=dict(color="lightgrey", size=4, opacity=0.5),
        name="PubChem reference",
        text=ref_valid,
        hovertemplate="%{text}<extra>Reference</extra>",
    ))

    fig.add_trace(go.Scatter(
        x=gen_coords[:, 0], y=gen_coords[:, 1],
        mode="markers",
        marker=dict(
            color=gen_props["QED"].tolist() if len(gen_props) == len(gen_smi) else "steelblue",
            colorscale="Viridis",
            size=14,
            colorbar=dict(title="QED"),
            line=dict(width=1, color="white"),
        ),
        name="Generated",
        text=gen_smi,
        hovertemplate="%{text}<extra>Generated</extra>",
    ))

    fig.update_layout(
        title="Chemical space — PCA of ECFP4 fingerprints<br>"
              "<sup>Click any point to see its structure and nearest neighbours</sup>",
        xaxis_title="PC 1",
        yaxis_title="PC 2",
        legend=dict(orientation="h", y=-0.12),
        height=560,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False)

    # Output widget that displays clicked molecule(s) below the plot
    detail_out = Output()

    def on_click(trace, points, state):
        if not points.point_inds:
            return
        idx = points.point_inds[0]
        with detail_out:
            clear_output(wait=True)
            is_generated = (trace.name == "Generated")
            clicked_smi = gen_smi[idx] if is_generated else ref_valid[idx]

            if is_generated:
                neighbours = _nearest_ref_smiles(gen_fps[idx], ref_fps, ref_valid, k=3)
                html = (
                    _mol_grid_html([clicked_smi], size=(200, 150), label="Generated molecule") +
                    _mol_grid_html(neighbours, size=(160, 120), label="3 nearest PubChem neighbours")
                )
            else:
                html = _mol_grid_html([clicked_smi], size=(200, 150), label="PubChem reference")

            display(HTML(html))

    fig.data[0].on_click(on_click)  # reference trace
    fig.data[1].on_click(on_click)  # generated trace

    display(fig)
    display(detail_out)
