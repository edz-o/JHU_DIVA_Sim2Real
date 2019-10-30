import os
import glob
import json
from tqdm import tqdm

activities = [
    'Closing', 
    'Closing_Trunk',
    'Entering',
    'Exiting', 
    'Open_Trunk',
    'Opening'
]

def mkdir(filename):
    folder = os.path.dirname(filename)
    if not os.path.isdir(folder):
        os.makedirs(folder)

def get_loc(data):
    x1, y1, x2, y2 = data
    x = (int(x1)+int(x2))/2
    y = (int(y1)+int(y2))/2
    w = abs((int(x1)-int(x2)))
    h = abs((int(y1)-int(y2)))
    return str(x)+' '+str(y)+' '+str(w)+' '+str(h)

def real_label_converter(mode):
    read_dir = "/data/wxy/1003_real_data/190207_DIVA_Union_CGT_images_label/{}/".format(mode)
    save_dir = "/data/wxy/1003_real_data/190207_DIVA_Union_CGT_gts/{}/".format(mode)
    for act in activities:
        print(act)
        paths = glob.glob(read_dir+act+"/*.json")
        for file in tqdm(paths):
            save_human = save_dir+act+'/'+file.split('/')[-1].split('.')[-2]+'/human/'
            save_car = save_dir+act+'/'+file.split('/')[-1].split('.')[-2]+'/car/'
            mkdir(save_human)
            mkdir(save_car)
            with open(file, 'r') as f:
                data = json.load(f)
                for i in range(len(data[1])):
                    # print(data[1][i][1])
                    file_human = open(save_human+'{:05d}.txt'.format(data[1][i][0]), 'w')
                    file_car = open(save_car+'{:05d}.txt'.format(data[1][i][0]), 'w')
                    if 'Person' in data[1][i][1].keys():
                        file_human.write(get_loc(data[1][i][1]['Person']))
                    else:
                        file_human.write("0.0 0.0 0.0 0.0")
                    if 'Vehicle' in data[1][i][1].keys():
                        file_car.write(get_loc(data[1][i][1]['Vehicle']))
                    else:
                        file_car.write("0.0 0.0 0.0 0.0")
                    file_human.close()
                    file_car.close()

def real_script_generator(mode):
    img_dir = "/data/wxy/1003_real_data/190207_DIVA_Union_CGT_images/{}/".format(mode)
    read_dir = "/data/wxy/1003_real_data/190207_DIVA_Union_CGT_images_label/{}/".format(mode)
    save_dir = '/data/wxy/diva_i3d/'
    sfile = open(save_dir+'{}.txt'.format(mode), 'w')
    for act in activities:
        paths = glob.glob(read_dir+act+"/*.json")
        for file in tqdm(paths):
            with open(file, 'r') as f:
                data = json.load(f)
                sfile.write(img_dir+act+'/'+file.split('/')[-1].split('.')[-2] \
                             +' {} {} {}\n'.format(data[1][0][0], data[1][-1][0], data[0]))
    sfile.close()

def sim_script_gengerator():
    img_dir = "/data/wxy/20191026_diva_sim_data/*/*/*/rgb/*.png"
    file = open("/data/wxy/diva_i3d/name_lists/sim_training.txt", 'w')
    imgs = glob.glob(img_dir)
    frame_dict = {}
    for name in tqdm(imgs):
        img = name.split('/')[-1]
        if name.split('/rgb/')[0]+"/rgb" not in frame_dict.keys():
            frame_dict[name.split('/rgb/')[0]+"/rgb"] = [int(img[:-4])]
        else:
            frame_dict[name.split('/rgb/')[0]+"/rgb"].append(int(img[:-4]))
    for path, frames in tqdm(frame_dict.items()):
        frames.sort()
        file.write("{} {:d} {:d} {}\n".format(path, frames[0], frames[-1], path.split('/')[-4]))
    
    file.close()

        



if __name__ == '__main__':
    # real_label_converter('validate')
    # real_script_generator('validate')
    sim_script_gengerator()
