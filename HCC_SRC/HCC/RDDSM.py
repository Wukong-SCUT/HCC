import torch

class Decomposition():
    def __init__(self, Matrix, device='cpu'):
        """
        Initialize the Decomposition class.

        Args:
            Matrix (torch.Tensor): The input matrix to be decomposed.
            device (str): The device to use ('cpu' or 'cuda').
        """
        self.Matrix = Matrix
        self.sub_space = []
        self.device = device

    def combine(self, list_of_lists):
        """
        Combine sublists where elements of length 1 are grouped together.

        Args:
            list_of_lists (list of lists): The list containing multiple sublists.

        Returns:
            list: The new combined list.
        """
        single_element_combined = []
        remaining_lists = []

        # Separate single-element lists and others
        for sublist in list_of_lists:
            if len(sublist) == 1:
                single_element_combined.extend(sublist)
            else:
                remaining_lists.append(sublist)

        if len(single_element_combined) > 0:
            new_list = remaining_lists + [single_element_combined]
        else:
            new_list = remaining_lists
        return new_list

    def is_subset(self, A, B):
        """
        Check if tensor A is a subset of tensor B.

        Args:
            A (torch.Tensor): The first tensor to check.
            B (torch.Tensor): The second tensor to check against.

        Returns:
            bool: True if A is a subset of B, otherwise False.
        """
        return torch.all((A == 0) | (A == B))

    def find_paradigm(self, sub_Matrix):
        """
        Find and group rows of the matrix with the same pattern.

        Args:
            sub_Matrix (torch.Tensor): The matrix to be processed.

        Returns:
            list: A list of sets, where each set contains indices of rows that share the same pattern.
        """
        element_indices = {}

        # Group rows with the same pattern
        for index, element in enumerate(sub_Matrix):
            element_tuple = tuple(element.tolist())  # Convert tensor to tuple for hashing
            if element_tuple in element_indices:
                element_indices[element_tuple].append(index)
            else:
                element_indices[element_tuple] = [index]

        keys = list(element_indices.keys())

        # Check subset relationships and merge indices
        for i in range(len(keys)):
            for j in range(len(keys)):
                if i != j and self.is_subset(torch.tensor(keys[i], device=self.device), torch.tensor(keys[j], device=self.device)):
                    element_indices[keys[i]].extend(element_indices[keys[j]])

        # Remove indices of subsets
        Paradigm_list = []
        indices_list = list(element_indices.values())

        # Remove intermediate subspaces that are subsets of others
        for i in range(len(indices_list)):
            is_subset = False
            for j in range(len(indices_list)):
                if i != j and set(indices_list[i]).issubset(set(indices_list[j])):
                    is_subset = True
                    break
            if not is_subset:
                Paradigm_list.append(set(indices_list[i]))

        return Paradigm_list

    def decomposition(self):
        """
        Perform decomposition of the matrix into subspaces based on paradigms.

        Returns:
            list: A list of combined subspaces after decomposition.
        """
        # Find the paradigms (patterns)
        Paradigm_list = self.find_paradigm(self.Matrix)
        for paradigm in Paradigm_list:
            paradigm_list = list(paradigm)
            self.sub_space.append(paradigm_list)

        # Convert sets to tuples and remove duplicates
        self.sub_space = {tuple(s) for s in self.sub_space}

        # Convert back to list of sets
        self.sub_space = [set(t) for t in self.sub_space]

        # Combine single elements
        self.sub_space = [list(s) for s in self.sub_space]
        self.sub_space = self.combine(self.sub_space)

        return self.sub_space




