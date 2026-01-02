import os

dir = 'l_p0_c1'

left_id_list = []
right_id_list = []

for filename in os.listdir(dir):
    parts = filename.split('_')
    id_left = parts[0][6:]
    id_right = parts[1][6:-4]
    left_id_list.append(id_left)
    right_id_list.append(id_right)

#save the id list to a txt file
with open('left_id_list.txt', 'w') as f:
    for item in left_id_list:
        f.write("%s\n" % item)

with open('right_id_list.txt', 'w') as f:
    for item in right_id_list:
        f.write("%s\n" % item)
