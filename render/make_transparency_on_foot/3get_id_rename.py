import os

#check if the output directory exists
output_dir = 'images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#get the name of all the file folder in this directory
# Get the current directory
current_directory = os.getcwd()

# List all files and folders in the current directory
items = os.listdir(current_directory)
folder_list = []
# Print the names of the files and folders
for item in items:
    #print(item)
    #print(item[0])
    if item[0] == 'l' or item[0] == 'r':
        folder_list.append(item)
print(folder_list)

for folder in folder_list:


    dir = folder
    dir_parts = dir.split('_')
    direction = dir_parts[0]
    pos = dir_parts[1]
    camara = dir_parts[2]
    dir_full = dir + '/output'


    for filename in os.listdir(dir_full):
        parts = filename.split('_')
        id_left = parts[0][6:]
        id_right = parts[1][6:-4]
        extension = parts[1][-4:]
        original_filepath = os.path.join(dir_full, filename)
        if direction == 'l':
            new_filename = os.path.join(output_dir, id_left + '_' + pos + '_' + camara + extension)
        else:
            new_filename = os.path.join(output_dir, id_right + '_' + pos + '_' + camara + extension)
        os.rename(original_filepath, new_filename)