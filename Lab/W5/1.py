from fractions import Fraction
from sympy import Matrix, Symbol, solve
import numpy as np
import time

class matrix:
    def __init__(self, matrix):
        self.matrix = np.array(matrix)  # Use NumPy array for matrix operations
        self.m = len(matrix)
        self.n = len(matrix[0])
        
    def if_square(self):
        return self.m == self.n
    
    def CharacteristicPolynomial(self):
        eq = np.polynomial.Polynomial.fromroots(np.linalg.eigvals(self.matrix))
        for i in range(len(eq.coef)):
            eq.coef[i] = Fraction(eq.coef[i]).limit_denominator()
        return eq
    
    def calEigenValue(self):
        eq = self.CharacteristicPolynomial()
        x = Symbol('x')
        ans = solve(eq(x), x)
        return ans
    
    def calEigenVector(self):
        eigenvalues = self.calEigenValue()
        eigenvectors = []
        
        # Convert to SymPy Matrix
        A = Matrix(self.matrix)
        
        for eigenvalue in eigenvalues:
            # A - λI
            A_minus_lambda_I = A - eigenvalue * Matrix.eye(self.m)
            
            # Find eigenvectors for each eigenvalue
            null_space = A_minus_lambda_I.nullspace()
            
            if null_space:
                for vec in null_space:
                    # Normalize and convert to float
                    eigenvectors.append(np.array(vec.evalf()).astype(float))
        
        # Remove duplicate eigenvectors (due to numerical precision)
        unique_eigenvectors = []
        for v in eigenvectors:
            if not any(np.allclose(v, u) for u in unique_eigenvectors):
                unique_eigenvectors.append(v)
        
        return unique_eigenvectors
    
    def calDiagTothePower(self, n):
        return [[self.matrix[i][j]**n for i in range(self.m)] for j in range(self.n)]
    
    def Diagonalize(self):
        if not self.if_square():
            return "Matrix is not square and cannot be diagonalized."
        
        eigenvectors = self.calEigenVector()
        
        # Check if the number of eigenvectors is sufficient
        if len(eigenvectors) < self.m:
            return "Matrix is not diagonalizable because there are not enough linearly independent eigenvectors."
        
        # Form the matrix P using eigenvectors
        P = np.column_stack(eigenvectors)
        # Check if P is invertible
        if np.linalg.matrix_rank(P) < self.m:
            return "Matrix is not diagonalizable because P is not invertible."
        # Compute the inverse of P
        P_inv = np.linalg.inv(P)
        # Compute the diagonal matrix D
        D = np.dot(np.dot(P_inv, self.matrix), P)
        
        return P, D, P_inv

# Example usage:
A = matrix(    [    [0,0,-2],
    [1,2,1],
    [1,0,3]])
P, D, P_inv = A.Diagonalize()
print("D:")
for row in D:
    print(row)
print("P:")
for row in P:
    print(row)
