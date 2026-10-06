# visualizations.py
# Generate paper-ready plots from merged_comments_cleaned_dates.csv
# (matplotlib only; no seaborn)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------
# Config (edit these)
# ----------------------------
INPUT_CSV = "manual_outputs/merged/merged_comments_cleaned_dates.csv"
OUTDIR = Path("manual_outputs/plots/final_plots")
OUTDIR.mkdir(parents=True, exist_ok=True)

# YEAR1–2025 window (your prof's framing). Change YEAR1 if needed.
YEAR1 = 2020
YEAR_END = 2025

# "X largest projects" = top projects by #comments in the time window
TOP_X_PROJECTS = 13

# Plot readability controls
MAX_AGENCIES = 6           # top agencies shown; rest -> "Other" for some plots
THRESHOLDS_DAYS = [30, 60, 90]
FACET_ECDF_BY_ROUND = True  # also save ECDF per round (Round 1 / 2 / 3+)

# Response-time sanity bounds (optional)
MIN_RT = 0
MAX_RT = 5000  # days


# ----------------------------
# Helpers
# ----------------------------
def ensure_cols(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable: {list(df.columns)}")

def round_group(x):
    """Map numeric rounds to '1', '2', '3+'."""
    if pd.isna(x):
        return np.nan
    try:
        x = float(x)
        if x >= 3:
            return "3+"
        if x == 1:
            return "1"
        if x == 2:
            return "2"
        # fallback
        return str(int(x)) if x.is_integer() else str(x)
    except Exception:
        return np.nan

def ecdf(values: np.ndarray):
    """Return sorted x and ECDF y."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.array([]), np.array([])
    v = np.sort(v)
    y = np.arange(1, len(v) + 1) / len(v)
    return v, y

def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


# ----------------------------
# Load + normalize
# ----------------------------
df = pd.read_csv(INPUT_CSV, low_memory=False)

# Required columns (adjust here if your schema differs)
ensure_cols(df, ["project", "agency", "round", "date_received"])

# Parse dates (safe even if already datetime)
for c in ["date_received", "date_responded"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

# Compute response_time_days if missing / incomplete
if "response_time_days" not in df.columns:
    df["response_time_days"] = np.nan

if "date_responded" in df.columns:
    need = df["response_time_days"].isna()
    df.loc[need, "response_time_days"] = (df.loc[need, "date_responded"] - df.loc[need, "date_received"]).dt.days

df["response_time_days"] = pd.to_numeric(df["response_time_days"], errors="coerce")

# Round group
df["round_group"] = df["round"].apply(round_group)
round_order = ["1", "2", "3+"]
df["round_group"] = pd.Categorical(df["round_group"], categories=round_order, ordered=True)

# Flags
df["has_response"] = df["date_responded"].notna() if "date_responded" in df.columns else df["response_time_days"].notna()

# Time filter: YEAR1–YEAR_END based on date_received
start = pd.to_datetime(f"{YEAR1}-01-01")
end = pd.to_datetime(f"{YEAR_END}-12-31")
dfw = df[(df["date_received"] >= start) & (df["date_received"] <= end)].copy()

# Keep the "X largest projects" by comment volume in the window
top_projects = (
    dfw.groupby("project")
       .size()
       .sort_values(ascending=False)
       .head(TOP_X_PROJECTS)
       .index
)
dfw = dfw[dfw["project"].isin(top_projects)].copy()

# Clean response time for response-time plots
resp = dfw[dfw["has_response"] & dfw["response_time_days"].notna()].copy()
resp = resp[(resp["response_time_days"] >= MIN_RT) & (resp["response_time_days"] <= MAX_RT)].copy()

# Top agencies (for ECDF / %within plots)
top_agencies = (
    dfw["agency"].value_counts()
       .head(MAX_AGENCIES)
       .index
       .tolist()
)


print(f"Loaded rows: {len(df):,}")
print(f"Window rows (YEAR1–{YEAR_END}): {len(dfw):,}")
print(f"Top X projects = {TOP_X_PROJECTS}: {list(top_projects)[:5]}{'...' if len(top_projects) > 5 else ''}")
print(f"Top agencies: {top_agencies}")
print(f"Writing plots to: {OUTDIR.resolve()}")


# ============================================================
# 1) Average response time by round
# ============================================================
avg_by_round = (
    resp.groupby("round_group")["response_time_days"]
        .agg(mean="mean", median="median", n="count")
        .reindex(round_order)
)

plt.figure(figsize=(7.2, 4.6))
plt.bar(avg_by_round.index.astype(str), avg_by_round["mean"].values)
plt.title("Average response time by round (days)")
plt.xlabel("Round")
plt.ylabel("Average response time (days)")

# annotate sample sizes
for i, (rg, row) in enumerate(avg_by_round.iterrows()):
    if pd.notna(row["mean"]):
        plt.text(i, row["mean"], f"n={int(row['n'])}", ha="center", va="bottom", fontsize=9)

savefig(OUTDIR / "avg_response_time_by_round.png")



# ============================================================
# 2) ECDF of response time by agency, faceted by round (ONE image)
#    -> 3 subplots: Round 1, Round 2, Round 3+
# ============================================================
resp_ag = resp[resp["agency"].isin(top_agencies)].copy()

# Fixed color mapping so each agency keeps the same color across all panels
base_colors = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])
# Fallback to tab20 if the default cycle is too short
if len(base_colors) < len(top_agencies):
    cmap = plt.get_cmap("tab20")
    base_colors = [cmap(i) for i in range(len(top_agencies))]

agency_color = {ag: base_colors[i % len(base_colors)] for i, ag in enumerate(top_agencies)}

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), sharey=True)

for ax, rg in zip(axes, round_order):
    sub = resp_ag[resp_ag["round_group"].astype(str) == rg].copy()

    for ag in top_agencies:
        vals = sub.loc[sub["agency"] == ag, "response_time_days"].values
        x, y = ecdf(vals)
        if x.size:
            ax.step(x, y, where="post", label=str(ag), color=agency_color[ag])

    ax.set_title(f"ECDF by agency — Round {rg}")
    ax.set_xlabel("Response time (days)")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Cumulative share of responded comments")

# One shared legend for the whole figure (fixed ordering + fixed colors)
legend_handles = [Line2D([0], [0], color=agency_color[ag], lw=2) for ag in top_agencies]
legend_labels = [str(ag) for ag in top_agencies]

fig.legend(
    legend_handles,
    legend_labels,
    title="Agency",
    loc="lower center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=min(len(legend_labels), 6),
    fontsize=9,
    frameon=False,
)

fig.suptitle("Response-time distribution (ECDF) by agency, faceted by round", y=0.98)
# Leave room for the title and the legend (so nothing overlaps)
fig.tight_layout(rect=[0, 0.10, 1, 0.92])
fig.savefig(OUTDIR / "ecdf_response_time_by_agency_by_round.png", dpi=250)
plt.close(fig)


# ============================================================
# 3) Prof’s suggestion: cumulative # comments by agency across rounds (ONE image)
#    -> 3 pie charts (cumulative up to Round 1 / 2 / 3+)
# ============================================================
# For readability: top agencies + "Other"
agency_counts = dfw["agency"].value_counts()
main_agencies = agency_counts.head(MAX_AGENCIES).index.tolist()
dfw["agency_plot"] = dfw["agency"].where(dfw["agency"].isin(main_agencies), other="Other")

counts = (
    dfw.groupby(["agency_plot", "round_group"])
       .size()
       .unstack("round_group")
       .reindex(columns=round_order)
       .fillna(0)
       .astype(int)
)

cum_counts = counts.cumsum(axis=1)  # cumulative across rounds, per agency

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))

for ax, rg in zip(axes, round_order):
    vals = cum_counts[rg].sort_values(ascending=False)
    total = vals.sum()
    if total == 0:
        ax.set_title(f"Up to Round {rg} (no data)")
        ax.axis("off")
        continue

    ax.pie(
        vals.values,
        labels=[str(i) for i in vals.index],
        autopct=lambda p: f"{p:.0f}%" if p >= 6 else "",  # show only bigger slices
        startangle=90,
    )
    ax.set_title("")
    ax.text(0.5, -0.10, f"Up to Round {rg}", ha="center", va="top", transform=ax.transAxes)

fig.suptitle(
    f"Cumulative comments by agency across rounds (Top {TOP_X_PROJECTS} projects, {YEAR1}–{YEAR_END})",
    y=0.98,
)
# Leave room for the main title and the per-pie captions below
fig.tight_layout(rect=[0, 0.08, 1, 0.92])
fig.savefig(OUTDIR / "cumulative_comments_by_agency_pies.png", dpi=250)
plt.close(fig)


print("Done. Generated files:")
print(" -", OUTDIR / "avg_response_time_by_round.png")
print(" -", OUTDIR / "cumulative_comments_by_agency_pies.png")
print(" -", OUTDIR / "ecdf_response_time_by_agency_by_round.png")














# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# import os

# # Create plots directory if it doesn't exist
# os.makedirs('manual_outputs/plots', exist_ok=True)
# os.makedirs('manual_outputs/reports', exist_ok=True)

# # Read the cleaned and merged comments data
# df = pd.read_csv('manual_outputs/merged/merged_comments_cleaned_dates.csv')

# # Generate report
# report_file = 'manual_outputs/reports/project_statistics.txt'
# with open(report_file, 'w') as f:
#     # Header
#     f.write("EMLI Comments Analysis Report\n")
#     f.write("=" * 50 + "\n")
#     f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
#     # Overall Statistics
#     total_projects = df['project'].nunique()
#     total_comments = len(df)
    
#     f.write("Overall Statistics:\n")
#     f.write(f"Total Number of Projects: {total_projects}\n")
#     f.write(f"Total Number of Comments from Ministry: {total_comments}\n")
#     f.write("-" * 50 + "\n\n")

#     # List all projects
#     f.write("List of All Projects:\n")
#     for idx, project in enumerate(sorted(df['project'].unique()), 1):
#         project_comments = len(df[df['project'] == project])
#         percentage = (project_comments / total_comments) * 100
#         f.write(f"{idx}. {project} ({project_comments} comments, {percentage:.1f}%)\n")
#     f.write("-" * 50 + "\n\n")
    
#     # Maximum number of rounds
#     max_round = df['round'].max()
#     f.write(f"Maximum Number of Rounds: {max_round}\n")

#     # Comments by Round
#     f.write("Comments Distribution by Round:\n")
#     round_stats = df['round'].value_counts().sort_index()
#     for round_num, count in round_stats.items():
#         percentage = (count / total_comments) * 100
#         f.write(f"Round {round_num}: {count} comments ({percentage:.1f}%)\n")
#     f.write("-" * 50 + "\n\n")

#     # Analyze comment progression across rounds
#     f.write("Comment Progression Analysis:\n")
    
#     # Group by project and comment_id to track progression
#     comment_groups = df.groupby(['project', 'comment_id'])
    
#     total_round1 = 0
#     progressed_to_round2 = 0
#     progressed_to_round3 = 0
    
#     for (project, comment_id), group in comment_groups:
#         rounds = set(group['round'])
#         if 1 in rounds:  # If this comment has a round 1
#             total_round1 += 1
#             if 2 in rounds:  # If it also has a round 2
#                 progressed_to_round2 += 1
#             if 3 in rounds:  # If it also has a round 3
#                 progressed_to_round3 += 1
    
#     # Calculate percentages
#     pct_to_round2 = (progressed_to_round2 / total_round1 * 100) if total_round1 > 0 else 0
#     pct_to_round3 = (progressed_to_round3 / total_round1 * 100) if total_round1 > 0 else 0
    
#     f.write(f"Total comments with Round 1: {total_round1}\n")
#     f.write(f"Comments progressing to Round 2: {progressed_to_round2} ({pct_to_round2:.1f}%)\n")
#     f.write(f"Comments progressing to Round 3: {progressed_to_round3} ({pct_to_round3:.1f}%)\n")
#     f.write("-" * 50 + "\n\n")

#     # Calculate project approval times
#     df['date_received'] = pd.to_datetime(df['date_received'])
#     df['date_responded'] = pd.to_datetime(df['date_responded'])
    
#     # Filter dates after year 2000
#     df_filtered = df[(df['date_received'].dt.year >= 2000) & (df['date_responded'].dt.year >= 2000)]
    
#     f.write("Project Approval Times:\n")
#     f.write("Time from Earliest Received to Latest Responded in each Project:\n")
#     approval_times = []
    
#     for project in df_filtered['project'].unique():
#         project_data = df_filtered[df_filtered['project'] == project]
#         earliest_received = project_data['date_received'].min()
#         latest_responded = project_data['date_responded'].max()
        
#         if pd.notna(earliest_received) and pd.notna(latest_responded):
#             time_diff = latest_responded - earliest_received
#             days = time_diff.days
#             years = days // 365
#             remaining_days = days % 365
#             months = remaining_days // 30
#             days = remaining_days % 30
            
#             approval_times.append({
#                 'project': project,
#                 'time_diff': time_diff,
#                 'formatted': f"{years}y {months}m {days}d"
#             })
            
#             f.write(f"{project}: {years}y {months}m {days}d\n")
    
#     if approval_times:
#         max_time = max(approval_times, key=lambda x: x['time_diff'])
#         f.write("\nMaximum Approval Time:\n")
#         f.write(f"Project: {max_time['project']}\n")
#         f.write(f"Time: {max_time['formatted']}\n")

#         # Calculate average approval time across all projects
#         total_days = sum([x['time_diff'].days for x in approval_times])
#         avg_days = total_days / len(approval_times) if approval_times else 0
#         avg_years = int(avg_days // 365)
#         remaining_days = int(avg_days % 365)
#         avg_months = remaining_days // 30
#         avg_days_final = remaining_days % 30

#         f.write(f"\nAverage Approval Time Across Projects: {avg_years}y {avg_months}m {avg_days_final}d\n")
    























# df2 = pd.read_csv('outputs/merged/merged_comments_cleaned_dates.csv')

# sns.set(style="whitegrid")

# # Plot 1: Number of Comments by Round
# plt.figure(figsize=(8, 5))
# sns.countplot(data=df2, x="round", palette="crest")
# plt.title("Number of Comments by Round", fontsize=14)
# plt.xlabel("Round")
# plt.ylabel("Number of Comments")
# plt.grid(True, axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig("outputs/plots/comments_by_round.png")
# plt.close()

# # Plot 2: Top Comment Types
# plt.figure(figsize=(10, 6))
# top_types = df2["comment_type"].value_counts().nlargest(10)
# sns.barplot(y=top_types.index, x=top_types.values, palette="pastel")
# plt.title("Top Comment Types", fontsize=14)
# plt.xlabel("Number of Comments")
# plt.ylabel("Comment Type")
# plt.grid(True, axis='x', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig("outputs/plots/top_comment_types.png")
# plt.close()

# # Plot 3: Average Response Time by Round
# plt.figure(figsize=(6, 4))
# avg_response = df2[(df2["response_time_days"] > 0) & (df2["response_time_days"] < 1000)].groupby("round")["response_time_days"].mean().dropna()
# sns.barplot(x=avg_response.index.astype(int), y=avg_response.values, palette="magma")
# plt.title("Average Response Time by Round", fontsize=14)
# plt.xlabel("Round")
# plt.ylabel("Avg Response Time (days)")
# plt.grid(True, axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig("outputs/plots/response_time_by_round.png")
# plt.close()

# # Inspect highest average response times in Round 3
# round3 = df2[(df2["round"] == 3) & (df2["response_time_days"] > 0)]
# top_round3 = round3.groupby(["project", "comment_id", "round"])["response_time_days"].mean().sort_values(ascending=False).head(5)
# print("\nTop 5 highest average response times in Round 3:")
# print(top_round3)

# # Plot 4: Top Projects by Number of Comments
# plt.figure(figsize=(10, 6))
# top_projects = df2["project"].value_counts().nlargest(10)
# sns.barplot(y=top_projects.index, x=top_projects.values, palette="flare")
# plt.title("Top 10 Projects by Number of Comments", fontsize=14)
# plt.xlabel("Number of Comments")
# plt.ylabel("Project")
# plt.grid(True, axis='x', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig("outputs/plots/comments_by_project.png")
# plt.close()

