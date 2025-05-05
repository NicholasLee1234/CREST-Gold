# Final output seperated with namingaxis as they are quite different in function and is easier to understand
import numpy as np

def get_user_abstract(): # Easier to put in main function
    user_abstract = str(input("Type in your abstract: "))
    return user_abstract

# Softmax by definition
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Find by dot product to each pc
def find_probabilities(principal_components, embedded_abstract, n_components: int = 10):
    probability_vector = []
    for i in range(n_components):
        probability = abs(np.dot(embedded_abstract, principal_components[i]))
        probability_vector.append(probability)
    probability_vector = softmax(np.array(probability_vector))
    return probability_vector
