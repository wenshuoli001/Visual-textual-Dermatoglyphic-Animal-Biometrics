import hou
import os
import time

texture_path = 'D:/houdini_project3/textures'
basecolor_texture_pram = hou.parm('/stage/materiallibrary1/mtlxmaterial/mtlximage1/file')
render_name_pram = hou.parm('/stage/karmarendersettings/picture')
render = hou.parm('/stage/usdrender_rop1/execute')
model_texture_pram = hou.parm('/mat/principledshader/basecolor_texture')
i = 0
for filename in os.listdir(texture_path):
    if filename.endswith('.png'):
        i += 1
        print(i)
        print(filename)
        #if filename is not exist in the render folder
        if not os.path.exists(f'D:/houdini_project3/render/{filename[:-4]}.png'):
            texture_file_path = f'{texture_path}/{filename}'
            basecolor_texture_pram.set(texture_file_path)  # set texture image
            model_texture_pram.set(texture_file_path)
            render_name_pram.set(f'D:/houdini_project3/render/{filename[:-4]}.png')  # set render image name
            render.pressButton()  # render image
            #time.sleep(2)  # wait for 60 seconds before starting the next render