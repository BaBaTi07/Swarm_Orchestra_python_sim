import numpy as np
import matplotlib.pyplot as plt
from TOOLS.evaluation import build_filename
from TOOLS.logger import logger

class QualityScoresHistory:
    # This class is used to store the history of the different scores for each metric,
    # to be able to plot their mean values and box plots.
    def __init__(self):
        self.phase_sync_history_score = []
        self.scale_harmony_history_score = []
        self.chord_harmony_history_score = []
        self.beat_evenness_history_score = []
        self.note_diversity_by_beat_history_score = []

    def add_scores(self, scores: dict):
        self.phase_sync_history_score.append(scores.get("phase_sync", np.nan))
        self.scale_harmony_history_score.append(scores.get("scale_harmony", np.nan))
        self.chord_harmony_history_score.append(scores.get("chord_harmony", np.nan))
        self.beat_evenness_history_score.append(scores.get("beat_evenness", np.nan))
        self.note_diversity_by_beat_history_score.append(scores.get("note_diversity_by_beat", np.nan))
    
    def get_all_scores(self):
        return {
            "phase_sync": self.phase_sync_history_score,
            "scale_harmony": self.scale_harmony_history_score,
            "chord_harmony": self.chord_harmony_history_score,
            "beat_evenness": self.beat_evenness_history_score,
            "note_diversity_by_beat": self.note_diversity_by_beat_history_score,
        }
    
        
def plot_all_score_history(self, base_name: str, folder: str):
    """
    Plot one boxplot per musical metric.

    Each boxplot shows the distribution of the scores
    across all trials.
    """

    all_scores = self.get_all_scores()

    labels = []
    data = []

    for metric_name, scores in all_scores.items():

        scores = np.array(scores, dtype=float)
        scores = scores[~np.isnan(scores)]

        if len(scores) == 0:
            continue

        labels.append(metric_name)
        data.append(scores * 100.0)

    if len(data) == 0:
        logger.log("WARNING", "No musical quality scores to plot.")
        return

    plt.figure(figsize=(12, 6))

    box = plt.boxplot(
        data,
        patch_artist=True,
        labels=labels,
        showmeans=True,
        )

    # Optional: slightly nicer visuals
    for patch in box['boxes']:
        patch.set_alpha(0.7)

    plt.ylabel("Score (%)")
    plt.xlabel("Metric")
    plt.title("Distribution of Musical Quality Metrics")
    plt.ylim(0, 100)

    plt.grid(True, axis="y", alpha=0.3)

    plt.xticks(rotation=15)

    plt.tight_layout()

    png_path = build_filename(f"{base_name}_musical_quality_history", folder, file_extension="png")
    plt.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.savefig("metrics/last/last_musical_quality_history.png", dpi=180, bbox_inches="tight")
    plt.close()
    logger.log("INFO",f"Saved musical quality history graph: {png_path}")