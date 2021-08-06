import numpy as np
import pandas as pd


class PopularityModel:
    def __init__(self, df):
        self.df = df

    def recommend(self, user_id, interactions_raw, num_recs=10):
        """
        Recommend recipes to the user based on the most popular recipes they haven't rated yet.
        :param num_recs: Number of recommended recipes to return
        :param interactions_raw: DataFrame containing all user interaction data
        :param user_id: User id of user who will receive recommendations
        :return: Returns list of the indices of recommended recipes
        """
        rec_recipes = []
        k = num_recs
        user_ratings = self.get_user_ratings(user_id, interactions_raw)
        for i in range(len(self.df)):
            recipe_id = self.df.iloc[i].recipe_id
            if recipe_id not in user_ratings:
                rec_recipes.append(recipe_id)
                k -= 1
                if k == 0:
                    break
        if k > 0:
            print("Returning as many recommendations as possible...")
        return rec_recipes

    def get_user_ratings(self, user_id, interactions_raw):
        """
        Get a set of the recipes that the given user has already reviewed
        :param user_id: ID of the user that we are providing recommendations for
        :param interactions_raw: DataFrame containing all user interaction data
        :return: Set of recipes reviewed by the user
        """
        user_ratings = set(interactions_raw[interactions_raw["user_id"] == user_id].recipe_id.unique())
        return user_ratings
