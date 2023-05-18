# Shared variables and helper functions across all analysis notebooks.
import itertools
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

EVAL_TYPES = ["Direct", "MetaQuestionSimple", "MetaInstruct", "MetaQuestionComplex"]
META_EVAL_TYPES = [e for e in EVAL_TYPES if e.startswith("Meta")]
eval_type_pairs = list(itertools.combinations(EVAL_TYPES, 2))
direct_pairs = [pair for pair in eval_type_pairs if "Direct" in pair]
OPTION_ORDERS = ["goodFirst", "badFirst"]

# Models are sorted by size (smallest to largest)
HF_MODELS = [
    "Flan-T5 SM",
    "Flan-T5 LG",
    "Flan-T5 XL",
]
OPENAI_MODELS = [
    "text-curie-001", 
    "text-davinci-002", 
    "text-davinci-003"
]
GLOBAL_MODEL_ORDER = HF_MODELS + OPENAI_MODELS

def pretty_model(model):
    model_map = {
        "google/flan-t5-small": "Flan-T5 SM",
        "google/flan-t5-large": "Flan-T5 LG",
        "google/flan-t5-xl": "Flan-T5 XL",
    }
    if model in model_map:
        return model_map[model]
    else:
        return model
    
def pretty_evaltype(eval_type):
    return eval_type[0].upper() + eval_type[1:]

# =============================================================================
# STYLING / AESTHETICS
# =============================================================================

# Define consistent color palette for all models.
# Hue corresponds to model family; lightness corresponds to size.
blues = sns.color_palette("Blues")
reds = sns.color_palette("Reds")

MODEL_PAL = {
    "Flan-T5 SM": blues[0],
    "Flan-T5 LG": blues[2],
    "Flan-T5 XL": blues[4],
    "text-curie-001": reds[0],
    "text-davinci-002": reds[2],
    "text-davinci-003": reds[4]
}

pal = sns.color_palette("Set2")
EVAL_TYPE_PAL = {
    "Direct": pal[0],
    "MetaQuestionSimple": pal[1],
    "MetaInstruct": pal[2],
    "MetaQuestionComplex": pal[3]
}

# Consistent styling for error bars.
BAR_STYLE = dict(capsize=0.05, errwidth=1) # lw=0.5)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def report_corr(xs, ys, alpha=0.05):
    pr, pp = stats.pearsonr(xs, ys)
    sr, sp = stats.spearmanr(xs, ys)
    d = pd.DataFrame([
        dict(method="pearson", r=pr, p=pp, sig=(pp<alpha)),
        dict(method="spearman", r=sr, p=sp, sig=(sp<alpha))
    ])
    return d

def render(out_file):
    plt.savefig(f"figures/{out_file}", bbox_inches="tight", dpi=300)
    print(f"Rendered figure to {out_file}")
    
def plot_metric(df, metric, ax=None, x="model", hue=None, 
                xlabel=None, ylabel=None, chance=None, ylim=None, 
                legend_pos=None, verbose=False,
                **bar_kwargs):
    
    if verbose:
        # Print out group-level means.
        levels = [x] if hue is None else [x, hue]
        sorted_means = df.groupby(levels)[metric].mean().sort_values()
        print(sorted_means)
    
    # Define parameters to pass to sns.barplot.
    if ax is None:
        ax = plt.gca()
    kwargs = dict(data=df, x=x, y=metric, ax=ax, **BAR_STYLE, **bar_kwargs)
    
    # Define order of x-axis groups.
    if x == "model":
        kwargs["order"] = [m for m in GLOBAL_MODEL_ORDER if m in df.model.unique()]
    elif x == "eval_type":
        kwargs["order"] = EVAL_TYPES
    
    # Define color palette.
    if hue is None:
        # Fall back to x-variable for color palette.
        if x == "model":
            kwargs["palette"] = MODEL_PAL
        elif x == "eval_type":
            kwargs["palette"] = EVAL_TYPE_PAL
    else:
        kwargs["hue"] = hue
        if hue == "model":
            kwargs["palette"] = MODEL_PAL
        if hue == "eval_type":
            kwargs["palette"] = EVAL_TYPE_PAL
            
    # Generate plot.
    ax = sns.barplot(**kwargs)
    
    # Optionally set chance reference line and y-axis limits.
    if chance is not None:
        ax.axhline(chance, linestyle="--", color="k", alpha=0.7)
    if ylim is not None:
        ax.set_ylim(*ylim)
        
    # Deal with labels and axis formatting.
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    # ax.xaxis.tick_top()
    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels, rotation=25, ha="right")
    
    # Deal with position of legend.
    if legend_pos is not None:
        ncol = df[hue].nunique()
        if legend_pos == "outside":
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        elif legend_pos == "top":
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=ncol)
        elif legend_pos == "bottom":
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=ncol)
    
    return ax
