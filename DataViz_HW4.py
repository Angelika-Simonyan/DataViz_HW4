import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

df = pd.read_csv("bundesliga.csv")
# print(df.head())

# Task 1.1
season_goals = df.groupby("SEASON")["FTTG"].sum().reset_index(name="TotalGoals")
season_matches = df.groupby("SEASON")["FTTG"].count().reset_index(name="Matches")
season_summary = pd.merge(season_goals, season_matches, on="SEASON")
season_summary["AvgGoalsPerMatch"] = season_summary["TotalGoals"] / season_summary["Matches"]
season_summary = season_summary.sort_values("SEASON")
# print(season_summary)

plt.figure(figsize=(10,6))
plt.plot(season_summary["SEASON"], season_summary["AvgGoalsPerMatch"], marker='o')
plt.title("Average Goals Per Match by Season")
plt.xlabel("Season")
plt.ylabel("Avg Goals Per Match")
plt.grid(True)
# plt.show()

# Task 1.2
df["SEASON"] = df["SEASON"].astype(str)
avg_goals = df.groupby("SEASON")["FTTG"].mean()
unique_seasons = sorted(df["SEASON"].unique())
season_color = {
    season: ('pink' if avg_goals.loc[season] > 2.5 else 'blue')
    for season in unique_seasons
}

plt.figure(figsize=(10, 6))
sns.violinplot(x="SEASON", y="FTTG", data=df, hue="SEASON",
               palette=season_color, order=unique_seasons, legend=False)
plt.title("Goal Distribution Per Season (FTTG) - Violin Plot")
plt.xlabel("Season")
plt.ylabel("Total Goals per Match")
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()

# Task 1.3
# df["DATE"] = pd.to_datetime(df["DATE"])
#
# home = df[['SEASON', 'DATE', 'HOMETEAM', 'FTHG']].copy()
# home.rename(columns={'HOMETEAM': 'TEAM', 'FTHG': 'Goals'}, inplace=True)
#
# away = df[['SEASON', 'DATE', 'AWAYTEAM', 'FTAG']].copy()
# away.rename(columns={'AWAYTEAM': 'TEAM', 'FTAG': 'Goals'}, inplace=True)
#
# data = pd.concat([home, away], ignore_index=True).sort_values(["SEASON", "DATE"])
# data["Match"] = data.groupby(["SEASON", "TEAM"]).cumcount() + 1
# data["CumGoals"] = data.groupby(["SEASON", "TEAM"])["Goals"].cumsum()
#
# # Saving into PDF
# with PdfPages("season_goal_trends.pdf") as pdf:
#     for season, season_df in data.groupby("SEASON"):
#         fig, ax = plt.subplots(figsize=(10, 6))
#
#         for team, team_df in season_df.groupby("TEAM"):
#             if "bayern" in team.lower() and "munchen" in team.lower():
#                 color, lw = "red", 2.5
#             else:
#                 color, lw = "gray", 1.5
#             # ax.plot(team_df["Match"], team_df["CumGoals"], color=color, linewidth=lw)
#
#         total_goals = df.loc[df["SEASON"] == season, "FTTG"].sum()
#         bayern_goals = season_df.loc[
#             season_df["TEAM"].str.lower().str.contains("bayern") &
#             season_df["TEAM"].str.lower().str.contains("munchen"),
#             "Goals"
#         ].sum()
#
#         ax.set_title(f"Season: {season} | Total Goals: {total_goals}", fontsize=14)
#         ax.set_xlabel("Match Number")
#         ax.set_ylabel("Cumulative Goals")
#         ax.grid(True)
#
#         # Footnote
#         plt.figtext(0.5, 0.01, f"Bayern Munchen Total Goals: {bayern_goals}", ha="center", fontsize=10)
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # pdf.savefig(fig)
        # plt.close(fig)

# Task 2.1
df['HomeWin'] = (df['FTHG'] > df['FTAG']).astype(int)
df['AwayWin'] = (df['FTAG'] > df['FTHG']).astype(int)

home_wins = df.groupby(['SEASON', 'HOMETEAM'])['HomeWin'].sum().reset_index()
away_wins = df.groupby(['SEASON', 'AWAYTEAM'])['AwayWin'].sum().reset_index()
# merged = pd.merge(home_wins, away_wins,
#                   left_on=['SEASON', 'HOMETEAM'],
#                   right_on=['SEASON', 'AWAYTEAM'],
#                   how='outer').fillna(0)
#
# merged['TEAM'] = merged['HOMETEAM'].combine_first(merged['AWAYTEAM'])
# merged = merged[['SEASON', 'TEAM', 'HomeWin', 'AwayWin']]
#
# seasons = sorted(merged['SEASON'].unique())
# for season in seasons:
#     season_data = merged[merged['SEASON'] == season].set_index('TEAM')[['HomeWin', 'AwayWin']]
#
#     plt.figure(figsize=(8, max(4, 0.5 * len(season_data))))
#     sns.heatmap(season_data, annot=True, fmt=".0f", cmap="coolwarm", cbar=True)
#     plt.title(f"Home vs. Away Wins per Team - Season {season}")
#     plt.ylabel("Team")
#     plt.xlabel("Win Type")
#     plt.tight_layout()
    # plt.show()

# Part 2.2
wins = pd.merge(home_wins, away_wins, left_on="HOMETEAM", right_on="AWAYTEAM", how="outer").fillna(0)
wins["TEAM"] = wins["HOMETEAM"].combine_first(wins["AWAYTEAM"])
wins["WinDiff"] = wins["HomeWin"] - wins["AwayWin"]

plt.figure(figsize=(10, 6))
sns.kdeplot(data=wins, x="WinDiff", fill=True, color="pink")
plt.title("Density of Win Differential (Home Wins - Away Wins) per Team")
plt.xlabel("Win Differential")
plt.ylabel("Density")
# plt.show()

# Part 3.1
# For home matches:
home = df[['SEASON', 'HOMETEAM', 'FTHG', 'FTAG']].copy()
home['Points'] = home.apply(lambda r: 3 if r['FTHG'] > r['FTAG']
                                        else (1 if r['FTHG'] == r['FTAG'] else 0), axis=1)
home['GoalsFor'] = home['FTHG']
home['GoalsAgainst'] = home['FTAG']
home.rename(columns={'HOMETEAM': 'TEAM'}, inplace=True)

# For away matches:
away = df[['SEASON', 'AWAYTEAM', 'FTAG', 'FTHG']].copy()
away['Points'] = away.apply(lambda r: 3 if r['FTAG'] > r['FTHG']
                                        else (1 if r['FTAG'] == r['FTHG'] else 0), axis=1)
away['GoalsFor'] = away['FTAG']
away['GoalsAgainst'] = away['FTHG']
away.rename(columns={'AWAYTEAM': 'TEAM'}, inplace=True)

# Combine home and away results.
stats = pd.concat([
    home[['SEASON', 'TEAM', 'Points', 'GoalsFor', 'GoalsAgainst']],
    away[['SEASON', 'TEAM', 'Points', 'GoalsFor', 'GoalsAgainst']]
], ignore_index=True)

# Aggregate season stats per team.
standings = stats.groupby(['SEASON', 'TEAM']).agg({
    'Points': 'sum',
    'GoalsFor': 'sum',
    'GoalsAgainst': 'sum'
}).reset_index()

standings['GoalDiff'] = standings['GoalsFor'] - standings['GoalsAgainst']

# Assigning ranking per season
def assign_ranks(group):
    group = group.sort_values(by=['Points', 'GoalDiff', 'GoalsFor'], ascending=False)
    group['Rank'] = range(1, len(group)+1)
    return group

standings = standings.groupby('SEASON', group_keys=False).apply(assign_ranks)

# Filter only the top 6 teams in each season.
top6 = standings[standings['Rank'] <= 6]

rank_pivot = top6.pivot(index='SEASON', columns='TEAM', values='Rank')
rank_pivot = rank_pivot.sort_index()

plt.figure(figsize=(12, 8))
for team in rank_pivot.columns:
    plt.plot(rank_pivot.index, rank_pivot[team], marker='o', label=team)
    for season, rank in rank_pivot[team].dropna().items():
        if rank == 1:
            plt.annotate("Champion", xy=(season, rank), xytext=(0, -10),
                         textcoords="offset points", color='red', fontsize=8, ha='center')

plt.gca().invert_yaxis()
plt.xlabel("Season")
plt.ylabel("Final League Position (Rank)")
plt.title("Seasonal Position Trajectories for Top 6 Teams\n(Title-winning seasons are annotated)")
plt.legend(title="Team", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
# plt.show()

# Task 3.2
volatility = standings.groupby('TEAM')['Rank'].std().reset_index().rename(columns={'Rank': 'Volatility'})
volatility = volatility.sort_values(by='Volatility', ascending=False)

median_vol = volatility['Volatility'].median()
volatility['Color'] = volatility['Volatility'].apply(lambda x: 'red' if x > median_vol else 'green')

plt.figure(figsize=(12, 6))
bars = plt.bar(volatility['TEAM'], volatility['Volatility'], color=volatility['Color'])

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}',
             ha='center', va='bottom', fontsize=9)

plt.xlabel("Team")
plt.ylabel("Volatility (Std Dev of Rank)")
plt.title("Team Volatility Index: Standard Deviation of Final Rank Over Seasons")
plt.xticks(rotation=90)
plt.tight_layout()
# plt.show()

# Task 5.2
df_home = df[['SEASON', 'HOMETEAM', 'FTHG', 'FTAG']].copy()
df_home['goal_diff'] = df_home['FTHG'] - df_home['FTAG']
df_home = df_home.rename(columns={'HOMETEAM': 'TEAM'})

# Compute goal difference for away matches (FTAG - FTHG); rename AWAYTEAM to TEAM.
df_away = df[['SEASON', 'AWAYTEAM', 'FTHG', 'FTAG']].copy()
df_away['goal_diff'] = df_away['FTAG'] - df_away['FTHG']
df_away = df_away.rename(columns={'AWAYTEAM': 'TEAM'})

# Combine home and away data and aggregate goal difference per team per season.
df_combined = pd.concat([df_home[['SEASON', 'TEAM', 'goal_diff']],
                         df_away[['SEASON', 'TEAM', 'goal_diff']]])

season_goal_diff = df_combined.groupby(['SEASON', 'TEAM'], as_index=False)['goal_diff'].sum()

# Get sorted unique seasons
seasons = sorted(season_goal_diff['SEASON'].unique())

# Save all season plots into a single PDF.
with PdfPages("season_goal_diff.pdf") as pdf:
    for season in seasons:
        # Filter season data and sort descending by goal difference.
        season_data = season_goal_diff[season_goal_diff['SEASON'] == season].copy()
        season_data = season_data.sort_values(by='goal_diff', ascending=False)
        n_teams = len(season_data)

        # Assign a unique color to each team using a rainbow palette.
        colors = plt.cm.rainbow(np.linspace(0, 1, n_teams))
        team_colors = {team: color for team, color in zip(season_data['TEAM'], colors)}

        # Identify the winner (team with the highest goal difference).
        winner = season_data.iloc[0]['TEAM']

        # For each team: winner gets its unique color; others get grey.
        season_data['bar_color'] = season_data['TEAM'].apply(
            lambda x: team_colors[x] if x == winner else "grey"
        )

        # Create horizontal bar plot.
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(season_data['TEAM'], season_data['goal_diff'], color=season_data['bar_color'])
        ax.invert_yaxis()  # Highest goal difference at the top

        # Annotate each bar with the exact goal difference value.
        for bar in ax.patches:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2,
                    f"{width:.0f}", va='center',
                    ha='left' if width >= 0 else 'right',
                    color="black", fontsize=9)

        ax.set_title(f"Season {season} - Goal Difference (Winner Highlighted)")
        ax.set_xlabel("Goal Difference")
        ax.set_ylabel("Team")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)