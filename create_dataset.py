import pandas as pd

# Locations of input/output data
data_dir = "~/Desktop/projects/cooking/"
mean_ratings_filename = "recipes_with_mean_ratings.csv"

print("Loading data...")
recipes_raw = pd.read_csv(data_dir + "RAW_recipes.csv")
interactions_raw = pd.read_csv(data_dir + "RAW_interactions.csv")

print("Processing data...")
if 'id' in recipes_raw:
    recipes_raw = recipes_raw.rename(columns={'id':'recipe_id'})
recipes_with_ratings = recipes_raw.merge(interactions_raw[['recipe_id', 'rating']], on="recipe_id", how="left")
keep_columns = [col for col in list(recipes_with_ratings.columns) if col not in ['contributor_id', 'submitted', 'rating']]
recipes_with_mean_ratings = recipes_with_ratings.groupby(keep_columns)['rating'].agg(['mean', 'count']).reset_index()

# All recipes should be unique at this point
assert(len(recipes_with_mean_ratings) == recipes_with_mean_ratings.recipe_id.nunique())

print("Writing output data...")
recipes_with_mean_ratings.to_csv(data_dir + mean_ratings_filename, encoding='utf-8', index=False)

print("Done!")