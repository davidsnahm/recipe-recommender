import numpy as np
import pandas as pd
from sklearn import metrics


class ContentBasedModel:
    def __init__(self, recipes, interactions):
        self.recipes = recipes
        self.interactions = interactions
        self.recipe_id_to_row_idx_mapping = self._build_recipe_id_to_row_index_mapping()

    def _get_user_reviews(self, user_id):
        return set(self.interactions[self.interactions.user_id == user_id].recipe_id)

    def _build_recipe_id_to_row_index_mapping(self):
        mapping = {}
        recipe_ids = self.recipes.recipe_id
        for i in range(len(recipe_ids)):
            mapping[recipe_ids] = i
        return mapping

    def _build_user_profile(self, user_id, tfidf_matrix):
        user_reviewed_recipes = self._get_user_reviews(user_id)
        rows = [self.recipe_id_to_row_idx_mapping[recipe_id] for recipe_id in user_reviewed_recipes]
        return tfidf_matrix.iloc[rows].mean(axis=0)

    def _calc_similarity(self, user_id, tfidf_matrix):
        # calc cosine similarity
        curr_user_profile = self._build_user_profile(user_id, tfidf_matrix)
        similarities = metrics.pairwise.cosine_similarity(tfidf_matrix, curr_user_profile)
        # sort similarities
        return similarities

    def recommend(self, user_id, tfidf_matrix):
        similarities = self._calc_similarity(user_id, tfidf_matrix)
        ignore_recipes = self._get_user_reviews(user_id)
