import os

# Path to the folder containing the PNG files
folder_path = 'background'

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Filter out only the PNG files
png_files = [file for file in files if file.endswith('.png')]

# Sort the files to ensure they are renamed in a consistent order
png_files.sort()

# Rename each file
for index, file in enumerate(png_files):
    new_name = f"{index}.png"
    old_file = os.path.join(folder_path, file)
    new_file = os.path.join(folder_path, new_name)
    os.rename(old_file, new_file)

print("Renaming complete!")