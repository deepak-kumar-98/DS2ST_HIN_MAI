# #!/usr/bin/env python3
# """
# ASR Evaluation Results — WER / CER Comparison Plot
# Hindi TTS Benchmark (2001 utterances)
# """

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import numpy as np

# # ─────────────────────────────────────────────────────────────────────────────
# # DATA
# # ─────────────────────────────────────────────────────────────────────────────
# models = [
#     "indicwav2vec-hindi\n(ai4bharat)",
#     "indic-conformer-600m\n(ai4bharat)",
#     "whisper-large-v3\n(openai)",
#     "wav2vec2-xlsr-hindi\n(theainerd)",
# ]

# wer = [20.54, 23.27, 32.17, 79.72]
# cer = [11.52, 11.94, 13.97, 35.49]

# # ─────────────────────────────────────────────────────────────────────────────
# # PLOT
# # ─────────────────────────────────────────────────────────────────────────────
# x       = np.arange(len(models))
# width   = 0.35

# fig, ax = plt.subplots(figsize=(11, 6))
# fig.patch.set_facecolor("#FAFAFA")
# ax.set_facecolor("#FAFAFA")

# bars_wer = ax.bar(x - width / 2, wer, width, label="WER %",
#                   color="#185FA5", zorder=3, clip_on=False)
# bars_cer = ax.bar(x + width / 2, cer, width, label="CER %",
#                   color="#1D9E75", zorder=3, clip_on=False)

# # ── value labels on top of each bar ──────────────────────────────────────────
# for bar in bars_wer:
#     ax.text(bar.get_x() + bar.get_width() / 2,
#             bar.get_height() + 0.8,
#             f"{bar.get_height():.2f}%",
#             ha="center", va="bottom", fontsize=9.5,
#             color="#185FA5", fontweight="bold")

# for bar in bars_cer:
#     ax.text(bar.get_x() + bar.get_width() / 2,
#             bar.get_height() + 0.8,
#             f"{bar.get_height():.2f}%",
#             ha="center", va="bottom", fontsize=9.5,
#             color="#1D9E75", fontweight="bold")

# # ── axes & grid ───────────────────────────────────────────────────────────────
# ax.set_xticks(x)
# ax.set_xticklabels(models, fontsize=10.5)
# ax.set_ylabel("Error Rate (%)", fontsize=11, labelpad=10)
# ax.set_ylim(0, 92)
# ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
# ax.grid(axis="y", color="#DDDDDD", linewidth=0.8, zorder=0)
# ax.set_axisbelow(True)

# for spine in ["top", "right"]:
#     ax.spines[spine].set_visible(False)
# for spine in ["left", "bottom"]:
#     ax.spines[spine].set_color("#CCCCCC")

# # ── legend ────────────────────────────────────────────────────────────────────
# legend_patches = [
#     mpatches.Patch(color="#185FA5", label="WER %  (Word Error Rate)"),
#     mpatches.Patch(color="#1D9E75", label="CER %  (Character Error Rate)"),
# ]
# ax.legend(handles=legend_patches, fontsize=10, frameon=False,
#           loc="upper left", bbox_to_anchor=(0.01, 0.99))

# # ── title & subtitle ──────────────────────────────────────────────────────────
# fig.text(0.5, 0.97,
#          "Hindi ASR Benchmark — WER & CER Comparison",
#          ha="center", va="top", fontsize=14, fontweight="bold", color="#222222")
# fig.text(0.5, 0.92,
#          "Evaluated on 2001 Hindi TTS utterances  •  Lower is better",
#          ha="center", va="top", fontsize=10, color="#666666")

# plt.tight_layout(rect=[0, 0, 1, 0.90])

# # ─────────────────────────────────────────────────────────────────────────────
# # SAVE
# # ─────────────────────────────────────────────────────────────────────────────
# OUT = "asr_wer_cer_comparison.png"
# plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
# print(f"Saved → {OUT}")
# plt.close()



import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Image

W, H = A4

models = [
    "ai4bharat/indicwav2vec-hindi",
    "ai4bharat/indic-conformer-600m-multilingual",
    "openai/whisper-large-v3",
    "theainerd/Wav2Vec2-large-xlsr-hindi",
]
wer = [20.54, 23.27, 32.17, 79.72]
cer = [11.52, 11.94, 13.97, 35.49]

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F8F8F8")
x = np.arange(len(models))
w = 0.32
b1 = ax.bar(x - w/2, wer, w, color="#185FA5", zorder=3)
b2 = ax.bar(x + w/2, cer, w, color="#1D9E75", zorder=3)
for bar, val, col in zip(list(b1)+list(b2), wer+cer, ["#185FA5"]*4+["#1D9E75"]*4):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
            f"{val:.2f}%", ha="center", va="bottom", fontsize=9, color=col, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=8.5, wrap=True)
ax.set_ylabel("Error Rate (%)", fontsize=10)
ax.set_ylim(0, 92)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color("#CCCCCC")
ax.legend(handles=[mpatches.Patch(color="#185FA5", label="WER %"),
                   mpatches.Patch(color="#1D9E75", label="CER %")],
          fontsize=9, frameon=False, loc="upper left")
plt.tight_layout(pad=0.8)
buf = io.BytesIO()
plt.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
buf.seek(0)
plt.close()

OUT = "/home/aditya/extraxtor_LLM/asr_testing/asr_scores.pdf"
doc = SimpleDocTemplate(OUT, pagesize=A4,
                        leftMargin=18*mm, rightMargin=18*mm,
                        topMargin=18*mm, bottomMargin=18*mm)
img_w = 169*mm
img_h = img_w * (5.5/10)
doc.build([Image(buf, width=img_w, height=img_h)])
print(f"Saved → {OUT}")