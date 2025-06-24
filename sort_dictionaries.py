# My Manual Implementation

#This implementation explicitly defines a lambda function for the key argument of list.sort().

def sort_dictionaries_manual(list_of_dicts, key_to_sort_by):
    """
    Sorts a list of dictionaries by a specific key using a lambda function.

    Args:
        list_of_dicts (list): A list of dictionaries.
        key_to_sort_by (str): The key to sort the dictionaries by.

    Returns:
        list: A new list of dictionaries sorted by the specified key.
    """
    
    return sorted(list_of_dicts, key=lambda d: d[key_to_sort_by])

# Example Usage:
data = [
    {'name': 'Lovemore', 'age': 30, 'city': 'Nairobi'},
    {'name': 'Bob', 'age': 25, 'city': 'Mombasa'},
    {'name': 'Charlie', 'age': 35, 'city': 'Eldorado'}
]

sorted_by_age = sort_dictionaries_manual(data, 'age')
print("Sorted by Age (Manual):")
for item in sorted_by_age:
    print(item)

sorted_by_name = sort_dictionaries_manual(data, 'name')
print("\nSorted by Name (Manual):")
for item in sorted_by_name:
    print(item)



# AI-Suggested Code (Simulated)
# An AI code completion tool like GitHub Copilot or Tabnine would likely suggest a concise and Pythonic approach,
#  often leveraging the "itemgetter" function from the operator module for performance and readability,
#  especially if it detects a common pattern of sorting by a dictionary key.

from operator import itemgetter

def sort_dictionaries_ai_suggestion(list_of_dicts, key_to_sort_by):
    """
    Sorts a list of dictionaries by a specific key using operator.itemgetter.

    Args:
        list_of_dicts (list): A list of dictionaries.
        key_to_sort_by (str): The key to sort the dictionaries by.

    Returns:
        list: A new list of dictionaries sorted by the specified key.
    """
    # Use itemgetter for potentially better performance and cleaner syntax
    return sorted(list_of_dicts, key=itemgetter(key_to_sort_by))

# Example Usage:
data = [
    {'name': 'Alice', 'age': 30, 'city': 'New York'},
    {'name': 'Bob', 'age': 25, 'city': 'Los Angeles'},
    {'name': 'Charlie', 'age': 35, 'city': 'Chicago'},
    {'name': 'David', 'age': 28, 'city': 'Houston'},
    {'name': 'Eve', 'age': 40, 'city': 'Miami'},
    {'name': 'Frank', 'age': 22, 'city': 'Seattle'},
    {'name': 'Grace', 'age': 31, 'city': 'Boston'},
    {'name': 'Heidi', 'age': 29, 'city': 'San Francisco'},
    {'name': 'Ivan', 'age': 45, 'city': 'Denver'}
]

sorted_by_age_ai = sort_dictionaries_ai_suggestion(data, 'age')
print("\nSorted by Age (AI Suggestion):")
for item in sorted_by_age_ai:
    print(item)

sorted_by_name_ai = sort_dictionaries_ai_suggestion(data, 'name')
print("\nSorted by Name (AI Suggestion):")
for item in sorted_by_name_ai:
    print(item)
