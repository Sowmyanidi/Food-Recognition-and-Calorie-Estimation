import json

# Manually create the class indices mapping
class_indices = {
    'Biryani': 0,
    'Butter Naan' : 1,
    'Chapathi' : 2,
    'Chicken Curry' : 3,
    'Dosa' : 4,
    'Ice Cream' : 5,
    'Idli' : 6,
    'Pav Bhaji' : 7,
    'Pizza' : 8,
    'Samosa' : 9
}

# Save class indices to JSON file
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)