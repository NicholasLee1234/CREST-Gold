# Final output seperated with namingaxis as they are quite different in function and is easier to understand
import namingaxis
import user_input
from namingaxis import principal_components

# Main Workflow (what users see)
axis_dict = namingaxis.main() # Only need to name axis once, and is saved for the main function. Can be written to be saved in external file if needed in future. 
principal_components = namingaxis.principal_components

def main(n_components: int = 10):
    user_abstract = user_input.get_user_abstract()
    embedded_user_abstract = namingaxis.embed_input(user_abstract)
    probability_vector = user_input.find_probabilities(principal_components, embedded_user_abstract)
    for i in range(n_components):
        print(f"{axis_dict[f'Axis {i+1}']}: {probability_vector[i]:.2f}")
    # Categorise again
    restart = int(input("\nTry Another Paper? (Type 0 to exit, Type 1 to go again): "))
    if restart != 0:
        main() # Recursion to use function again

if __name__ == "__main__":
    main()
