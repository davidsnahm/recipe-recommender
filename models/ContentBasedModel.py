import numpy as np
import pandas as pd
from sklearn import metrics


class ContentBasedModel:
    def __init__(self, recipes, interactions):
        self.recipes = recipes
        self.interactions = interactions
        self.recipe_id_to_row_idx_mapping = self._build_recipe_id_to_row_index_mapping()

    def _get_user_reviews(self, user_id):
        """
        Returns a list of recipes the user has reviewed
        :param user_id:
        :return:
        """
        return self.interactions[self.interactions.user_id == user_id].recipe_id

    def _build_recipe_id_to_row_index_mapping(self):
        mapping = {}
        recipe_ids = self.recipes.id
        for i in range(len(recipe_ids)):
            mapping[recipe_ids[i]] = i
        return mapping

    def _build_user_profile(self, user_id, tfidf_matrix):
        user_reviewed_recipes = self._get_user_reviews(user_id)
        rows = [self.recipe_id_to_row_idx_mapping[recipe_id] for recipe_id in user_reviewed_recipes]
        return tfidf_matrix.iloc[rows].mean(axis=0)

    def _calc_similarity(self, user_id, tfidf_matrix, n=1000):
        # calc cosine similarity
        curr_user_profile = self._build_user_profile(user_id, tfidf_matrix)
        similarities = metrics.pairwise.cosine_similarity(tfidf_matrix, curr_user_profile.values.reshape(1, -1))
        # sort similarities
        indices = similarities.flatten().argsort()[-n::-1]
        return indices

    def recommend(self, user_id, tfidf_matrix, k=10):
        """
        Returns the indices of the k most similar recipes to the user's tastes, excluding recipes
        the user has already reviewed
        :param user_id:
        :param tfidf_matrix:
        :param k: The number of recipes to recommend
        :return:
        """
        most_similar_recipes_indices = self._calc_similarity(user_id, tfidf_matrix)
        ignore_recipes_indices = set(self._get_user_reviews(user_id))
        top_k_similar_recipes_indices = [idx for idx in most_similar_recipes_indices if idx not in ignore_recipes_indices][:k]
        return top_k_similar_recipes_indices

