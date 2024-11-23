import numpy as np #used uniquely to deal with operations in lists

def dot_product(vec1, vec2):
    """
    Dot product of two vectors.
    """ 
    return sum(x1 * x2 for x1, x2 in zip(vec1, vec2))

def check_orthogonality(vec1, vec2):
    """
    Checks for orthogonality given 2 vectors by computing dot product. Used in test cases for Gram-Schmidt.
    """
    if dot_product(vec1, vec2) == 0:
        return True
    else:
        return False

def gramschmidt(V):
    """
    Gram-Schmidt algorithm. Takes in a basis and returns an orthogonal basis.
    Based on pseudo-code HPS 7.3.
    """ 
    def projection(u, v): #helper for GS
        return (dot_product(u, v) / dot_product(u, u)) * np.array(u)

    U = [] #orthogonalized basis

    for v in V:
        temp = np.array(v)
        for u in U:
            temp = np.array(temp) - projection(u, v)

        if np.any(temp):
            U.append(temp.tolist())
            
    return U

def LLL_reduction(basis):
    """
    Computes LLL reduction algorithm on a given basis.
    Based on pseudo-code HPS 7.13.
    """
    n = len(basis)
    k = 1

    o_basis = gramschmidt(basis)

    def coef(i, j): #helper for LLL
        return (dot_product(basis[i], o_basis[j]) / dot_product(o_basis[j], o_basis[j]))

    while k < n:
        for j in range(k - 1, -1, -1):
            #Reduction step
            mu1 = coef(k,j)
            basis[k] = np.array(basis[k]) - ( round(mu1) * np.array(basis[j]) )
            o_basis = gramschmidt(basis)
        
        #Check for Lovász Condition
        mu2 = coef(k, k - 1)
        if ( dot_product(o_basis[k], o_basis[k]) >= ( 0.75 - pow(mu2,2) ) * dot_product(o_basis[k - 1], o_basis[k - 1]) ):
            k += 1
        else:
            #Swap step
            basis[k], basis[k - 1] = basis[k - 1], basis[k]
            o_basis = gramschmidt(basis)
            k = max(k - 1, 1)

    return basis