import json

# File paths
#json_input_file_path = 'output/wlasl_989.json'  # Path to the original JSON data file
#new_gloss_handshapes_file_path = 'output/gloss_handshape_pairs_kmeans.txt'  # Path to the .txt file with new glosses and handshapes
#json_output_file_path = 'output/updated_wlasl_989.json'  # Path to the new JSON file to write the updated data

json_input_file_path = 'utils/semlex_800.json'  # Path to the original JSON data file
new_gloss_handshapes_file_path = 'output/gloss_handshape_pairs_kmeans.txt'  # Path to the .txt file with new glosses and handshapes
json_output_file_path = 'output/updated_semlex_989.json'  # Path to the new JSON file to write the updated data


# Function to read new glosses and handshapes from a .txt file
def read_new_gloss_handshapes(file_path):
    new_gloss_handshapes = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 2:
                new_gloss_handshapes.append((parts[0].strip(), parts[1].strip()))
    return new_gloss_handshapes

# Function to update handshapes in the JSON data
def update_handshapes(json_data, new_gloss_handshape_dict):
    for item in json_data:
        gloss = item.get('gloss')
        if gloss in new_gloss_handshape_dict:
            new_handshape = new_gloss_handshape_dict[gloss]
            for instance in item.get('instances', []):
                instance['Handshape'] = new_handshape

# Read new glosses and handshapes from the .txt file
new_gloss_handshapes = read_new_gloss_handshapes(new_gloss_handshapes_file_path)

# Convert list to a dictionary for faster lookup
new_gloss_handshape_dict = dict(new_gloss_handshapes)

# Load JSON data
with open(json_input_file_path, 'r') as file:
    data = json.load(file)

# Update handshapes in the loaded JSON data
update_handshapes(data, new_gloss_handshape_dict)

# Save the updated JSON data to a new file
with open(json_output_file_path, 'w') as file:
    json.dump(data, file, indent=4)

print("Handshapes updated and saved to new file successfully.")