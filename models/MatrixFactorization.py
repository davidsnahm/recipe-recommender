import numpy as np

class MatrixFactorization:

    def __init__(self, num_iters, num_features, user_reg=0.0, recipe_reg=0.0):
        self.num_iters = num_iters
        self.num_features = num_features
        self.user_reg = user_reg
        self.recipe_reg = recipe_reg


    def fit(self, X_train, X_test):
        """
        Fit the model
        :param X_train:
        :param X_test:
        :return: Returns training loss and test loss as lists
        """
        assert len(X_train.shape) == 2
        num_users, num_recipes = X_train.shape[0], X_train.shape[1]
        user_features = np.random((num_users, self.num_features))
        recipe_features = np.random((num_recipes, self.num_features))
        train_loss, test_loss = [], []
        for iter in range(self.num_iter):
            recipe_features = self.wals_step(user_features, recipe_features, self.recipe_reg)
            user_features = self.wals_step(recipe_features, user_features, self.user_reg)
            if iter % 5 == 0:
                train_loss.append(self.calc_loss(X_train, user_features, recipe_features))
                test_loss.append(self.calc_loss(X_test, user_features, recipe_features))
        return train_loss, test_loss

    def calc_loss(self, X, user_features, recipe_features):
        mask = np.nonzero(X)
        X_pred = np.dot(user_features, recipe_features.T)
        X_masked = X[mask]
        return np.linalg.norm(X_masked - X_pred[mask]) ** 2 + self.user_reg * np.linalg.norm(user_features) ** 2 + self.recipe_reg * np.linalg.norm(recipe_features) ** 2

    """
    || A - UV* ||F +  l1 * || U ||F + l2 * || V ||F
    """
    def wals_step(self, fixed_features, var_features, reg):
        return 0