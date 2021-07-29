import numpy as np
import pandas as pd


class PopulationModel:
    def __init__(self, data):
        # TODO: add groupby/sort mean rating
        self.data = data

    def recommend(self, user_id, user_ratings, num_recs=10):
        """
        Recommend recipes to the user based on the most popular recipes they haven't rated yet.
        :param num_recs: Number of recommended recipes to return
        :param user_ratings: The recipes the user has already rated
        :param user_id: User id of user who will receive recommendations
        :return: Returns list of the indices of recommended recipes
        """
        rec_recipes = []
        k = num_recs
        for i in range(len(self.data)):
            if self.data.recipe not in user_ratings:
                rec_recipes.append(self.data.recipe)
                k -= 1
                if k == 0:
                    break
        if k > 0:
            print("Returning as many recommendations as possible...")
        return rec_recipes
