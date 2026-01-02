from PIL import Image, ImageFile
from torch import Tensor

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.simple_tokenizer import SimpleTokenizer#这个类用于将文本转换为 token 序列。


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> Tensor:#这个函数用于将文本转化为 token 序列。这个函数的主要用途是将输入的文本转换为一个固定长度的张量
    sot_token = tokenizer.encoder["<|startoftext|>"]#这个 token 表示文本的开始。tokenizer.encoder是一个字典，它将每个可能的单词或标记符映射到一个唯一的整数
    eot_token = tokenizer.encoder["<|endoftext|>"]#这个 token 表示文本的结束。
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]#对输入的 caption 进行编码，并在编码的开始和结束添加开始和结束的标记符，生成 tokens 列表.tokenizer.encode 方法将 caption 编码为一个整数列表

    result = torch.zeros(text_length, dtype=torch.long)#创建一个长度为 text_length 的全零张量
    if len(tokens) > text_length:#如果 tokens 的长度大于 text_length，则截断 tokens
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:#如果 truncate 为 False，则抛出异常
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)#将 tokens 转换为张量，并将张量的值赋给 result
    return result

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, text_length: int = 77, truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer = SimpleTokenizer()
        self.text_length = text_length
        self.truncate = truncate


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, caption, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        text_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length,
                          truncate=self.truncate)  # 将 caption 转换为 token 序列

        return img, text_tokens, pid, camid, trackid,img_path.split('/')[-1]