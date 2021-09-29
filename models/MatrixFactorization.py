class MatrixFactorization:
    
    def __init__(self, n_iter):
        self.n_iter = n_iter
        
    """
    || A - UV* ||F +  l1 * || U ||F + l2 * || V ||F
    """
    def wals_step(self, fixed_matrix, var_matrix):
        