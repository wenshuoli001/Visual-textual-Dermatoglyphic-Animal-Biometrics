# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle
class AmurTiger(BaseImageDataset):
    """
    Market1501
    Reference:
    ATRW: A Benchmark for Amur Tiger Re-identification in the Wild
    URL: https://cvwc2019.github.io/challenge.html

    Dataset statistics:
    # identities: 	92
    # images: 1887 (train) + 1762 (query) + 1762 (gallery)
    """
    dataset_dir = 'amurtiger'

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(AmurTiger, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)#获取文件夹路径
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()#检查文件夹是否存在
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> AmurTiger loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))



    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))#获取dir_path目录下所有.jpg文件的路径
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            #获取与图片路径同名的txt文件的路径
            txt_path = img_path.replace('.jpg', '.txt')
            #读取txt文件中的内容
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                caption = f.read().strip()
            # #if pid == -1: continue  # junk images are just ignored
            # print('debug!!!!!!!!!!!!!!')
            # print(img_path)
            # print(pid)
            # print(camid)
            #assert 0 <= pid <= 20  # pid == 0 means background
            #assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, caption, self.pid_begin + pid, camid, 1))
        return dataset

    # def _process_dir_test(self, dir_path, relabel=False):
    #     dataset = []
    #     img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
    #     for img_path in img_paths:
    #         dataset.append((img_path, 1, 1, 1))  # 相当于手动对原来的图片加上id和cam
    #     return dataset
