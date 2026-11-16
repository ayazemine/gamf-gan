"""Shared helpers for LaTeX table output and model/dataset names."""
from pathlib import Path

MODEL_NAMES = {
    "gamf": "GAMF-GAN (ours)",
    "attngan": "AttnGAN$^{\\dagger}$",
    "dmgan": "DM-GAN$^{\\dagger}$",
    "stylegan_t": "TC-StyleGAN$^{\\dagger}$",
    "dfgan": "DF-GAN$^{\\dagger}$",
    "gamf_no_clip": "w/o CLIP branch",
    "gamf_no_attn": "w/o word attention",
    "gamf_no_mem": "w/o dynamic memory",
    "abl_no_attn": "w/o word attention",
    "abl_no_mem": "w/o dynamic memory",
    "abl_no_style": "w/o style stage",
    "abl_no_perc": "w/o $L_{\\mathrm{perc}}$",
    "abl_no_styleloss": "w/o $L_{\\mathrm{style}}$",
    "abl_no_memloss": "w/o $L_{\\mathrm{mem}}$",
}
PLAIN_NAMES = {"gamf": "GAMF-GAN", "attngan": "AttnGAN", "dmgan": "DM-GAN",
               "stylegan_t": "TC-StyleGAN", "dfgan": "DF-GAN"}
DATASET_NAMES = {"cub": "CUB-200-2011", "flowers": "Oxford-102 Flowers", "coco": "MS COCO"}
BASELINE_NOTE = ("$^{\\dagger}$Re-implemented and trained by the authors under the identical protocol "
                 "(same subsets, splits, epochs, batch size and evaluation).")


def model_name(key):
    return MODEL_NAMES.get(key, key.replace("_", "\\_"))


def dataset_name(key):
    return DATASET_NAMES.get(key, key)


def table_env(tabular, caption, label, wide=False, note=None):
    """Wrap tabular lines in a complete IEEE-style table environment."""
    env = "table*" if wide else "table"
    lines = [f"\\begin{{{env}}}[!t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\setlength{\\tabcolsep}{4pt}", *tabular]
    if note:
        lines.append(f"\\par\\smallskip\\parbox{{\\linewidth}}{{\\footnotesize {note}}}")
    lines.append(f"\\end{{{env}}}")
    return lines


def write_lines(lines, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
