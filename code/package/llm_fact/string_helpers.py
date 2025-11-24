"""
Helper functions for calculating edit distance.
"""


def edit_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.

    Ref: https://machinelearningknowledge.ai/ways-to-calculate-levenshtein-distance-edit-distance-in-python/#1_Using_Python_Code_from_Scratch

    Parameters:
    -----------
    - s1 (str): The first string to compare.
    - s2 (str): The second string to compare.

    Returns:
    - int: The edit distance between the two strings.
    """
    # Create a matrix to store distances between substrings
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the first column and the first row of the matrix
    for i in range(1, m + 1):
        dp[i][0] = i  # Cost of deleting characters from s1
    for j in range(1, n + 1):
        dp[0][j] = j  # Cost of inserting characters into s1

    # Compute the distance
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0  # No operation needed if characters are the same
            else:
                cost = 1  # Substitution cost
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,
            )  # Substitution

    return dp[m][n]


def find_closest_string(input_string, string_list):
    """
    Find the string in the list with the minimum edit distance compared to the input string.

    Parameters:
    -----------
    - input_string (str): The string to compare against the list.
    - string_list (list of str): The list of strings to compare.

    Returns:
    -----------
    - str: The string from the list that has the minimum edit distance to the input string.
    """
    # Initialize the minimum distance to a large number and the closest string as None
    min_distance = float("inf")
    closest_string = None

    # Iterate through each string in the list
    for string in string_list:
        # Calculate the edit distance between the input string and the current string
        distance = edit_distance(input_string, string)
        # If the current distance is less than the found minimum, update min_distance and closest_string
        if distance < min_distance:
            min_distance = distance
            closest_string = string

    return closest_string
