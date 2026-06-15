import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# 1. DATA LOADING & CLEANING
# ==========================================

# Load the dataset (using correct file name)
df = pd.read_csv('netflix_titles.csv.csv')

# Clean 'date_added' and extract the year added
df['date_added_clean'] = df['date_added'].str.strip()
df['date_added_parsed'] = pd.to_datetime(df['date_added_clean'], format='%B %d, %Y', errors='coerce')
df['year_added'] = df['date_added_parsed'].dt.year

# Data Correction: Fix anomalies where duration values leaked into the rating column
mask = df['duration'].isna() & df['rating'].str.contains('min', na=False)
df.loc[mask, 'duration'] = df.loc[mask, 'rating']
df.loc[mask, 'rating'] = np.nan


# ==========================================
# 2. EXPLORATORY DATA ANALYSIS (VISUALIZATIONS)
# ==========================================

# Task A: Analyze Netflix Content Growth Over Time (Post-2008)
growth_df = df[df['year_added'] >= 2008].groupby(['year_added', 'type']).size().unstack().fillna(0)

fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.lineplot(data=growth_df, markers=True, linewidth=2.5, ax=ax1)
ax1.set_title('Netflix Content Growth Over Time (Year Added)', fontsize=14)
ax1.set_xlabel('Year Added', fontsize=12)
ax1.set_ylabel('Number of Titles', fontsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
fig1.savefig('netflix_growth_over_time.png')
plt.close(fig1)

# Task B: Visualize Distributions of Genres, Ratings, and Content Type
fig2, axes = plt.subplots(1, 3, figsize=(22, 7))

# Content Type Distribution
sns.countplot(data=df, x='type', ax=axes[0], palette='pastel')
axes[0].set_title('Distribution of Content Type', fontsize=12)
axes[0].set_xlabel('Content Type')
axes[0].set_ylabel('Count')

# Top 10 Ratings Distribution (Sorted automatically via value_counts)
top_ratings = df['rating'].value_counts().iloc[:10]
sns.barplot(x=top_ratings.values, y=top_ratings.index, ax=axes[1], palette='vlag', orient='h')
axes[1].set_title('Top 10 Content Ratings', fontsize=12)
axes[1].set_xlabel('Count')

# Top 10 Genres Distribution (Splitting multi-genre labels and sorting)
all_genres = df['listed_in'].str.split(', ').explode()
top_genres = all_genres.value_counts().iloc[:10]
sns.barplot(x=top_genres.values, y=top_genres.index, ax=axes[2], palette='rocket', orient='h')
axes[2].set_title('Top 10 Genres on Netflix', fontsize=12)
axes[2].set_xlabel('Count')

plt.tight_layout()
fig2.savefig('netflix_distributions.png')
plt.close(fig2)

# Task C: Identify Country-Level Content Contributions (Top 10)
all_countries = df['country'].dropna().str.split(', ').explode()
top_countries = all_countries.value_counts().iloc[:10]

fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.barplot(x=top_countries.values, y=top_countries.index, ax=ax3, palette='coolwarm', orient='h')
ax3.set_title('Top 10 Country-Level Content Contributors', fontsize=14)
ax3.set_xlabel('Number of Titles')
plt.tight_layout()
fig3.savefig('netflix_country_contributions.png')
plt.close(fig3)


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

# Feature 1: Content Length Category
def categorize_length(row):
    duration = row['duration']
    if pd.isna(duration):
        return 'Unknown'
    if 'Season' in duration:
        try:
            seasons = int(duration.split()[0])
            return 'Short TV Show (1 Season)' if seasons == 1 else 'Long TV Show (2+ Seasons)'
        except:
            return 'Unknown'
    elif 'min' in duration:
        try:
            minutes = int(duration.split()[0])
            if minutes < 60:
                return 'Short Movie (<60 min)'
            elif minutes <= 120:
                return 'Standard Movie (60-120 min)'
            else:
                return 'Long Movie (>120 min)'
        except:
            return 'Unknown'
    return 'Unknown'

df['duration_category'] = df.apply(categorize_length, axis=1)

# Feature 2: Original vs. Licensed (Proxy heuristic based on release vs added gap)
def estimate_original_status(row):
    if pd.isna(row['year_added']):
        return 'Unknown'
    # Proxy: Content arriving within 1 year of production is classified as Original/New Release
    if (row['year_added'] - row['release_year']) <= 1:
        return 'Original/New Release'
    else:
        return 'Licensed/Library'

df['content_source_type'] = df.apply(estimate_original_status, axis=1)


# ==========================================
# 4. EXPORT & VERIFICATION
# ==========================================

# Save the newly transformed dataset with engineering features
df.to_csv('netflix_titles_engineered.csv', index=False)

print("Execution Completed Successfully!")
print("\n--- Duration Category Metrics ---")
print(df['duration_category'].value_counts())
print("\n--- Content Source Heuristic Metrics ---")
print(df['content_source_type'].value_counts())