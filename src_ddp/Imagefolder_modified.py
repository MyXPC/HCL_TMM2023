from torchvision.datasets import VisionDataset  # 导入PyTorch视觉数据集基类
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载截断的图像文件
from PIL import Image  # 导入PIL图像处理库

import os
import os.path
import sys
import random  # 导入随机数模块



def has_file_allowed_extension(filename, extensions):
    """检查文件是否具有允许的扩展名

    Args:
        filename (string): 文件路径
        extensions (tuple of strings): 允许的扩展名元组（小写）

    Returns:
        bool: 如果文件以给定扩展名之一结尾，则返回True
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """检查文件是否为允许的图像扩展名

    Args:
        filename (string): 文件路径

    Returns:
        bool: 如果文件以已知图像扩展名结尾，则返回True
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None, number = None):
    """创建数据集 - 从目录结构中构建图像路径和标签的列表
    
    Args:
        dir: 数据集根目录
        class_to_idx: 类别名称到索引的映射字典
        extensions: 允许的文件扩展名列表
        is_valid_file: 自定义文件验证函数
        number: 每个类别最多使用的样本数量（None表示使用所有样本）
    
    Returns:
        list: 包含(图像路径, 类别索引)元组的列表
    """
    images = []
    dir = os.path.expanduser(dir)  # 扩展用户目录路径
    # 检查参数有效性：extensions和is_valid_file不能同时为None或同时不为None
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    # 遍历所有类别目录
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        # 遍历类别目录中的所有文件
        for root, _, fnames in sorted(os.walk(d)):
            if number!= None:
                if len(fnames) > number:
                    fnames = random.sample(fnames, number)  # 随机采样指定数量的文件
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])  # 创建(路径, 类别索引)元组
                    images.append(item)

    return images


class DatasetFolder(VisionDataset):
    """通用数据集加载器 - 按类别目录组织样本的数据集
    
    样本按以下方式组织: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): 根目录路径
        loader (callable): 根据路径加载样本的函数
        extensions (tuple[string]): 允许的扩展名列表
            extensions和is_valid_file不能同时传递
        transform (callable, optional): 对样本进行变换的函数/变换器
            例如，图像的``transforms.RandomCrop``
        target_transform (callable, optional): 对目标进行变换的函数/变换器
        is_valid_file (callable, optional): 检查图像文件是否有效的函数
            （用于检查损坏文件）extensions和is_valid_file不能同时传递
        cached (bool, optional): 是否将所有图像一次性加载到RAM中
        number (int, optional): 每个类别使用的样本数量限制

     Attributes:
        classes (list): 类别名称列表
        class_to_idx (dict): 包含(类别名称, 类别索引)项的字典
        samples (list): (样本路径, 类别索引)元组列表
        targets (list): 数据集中每个图像的类别索引值
    """

    def __init__(self, root, loader, extensions=None, transform=None, target_transform=None, is_valid_file=None, cached=False, number=None):
        super(DatasetFolder, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        self.cached=cached
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file, number)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        dict = {}
        for key in self.targets:
            dict[key] = dict.get(key, 0) + 1
        
        print('--------------------------------')

        if number != None:
            print('Using part of images, number of each class: ',number)
        else:
            print('Using all images')
        if cached == True:
            print('load all images once into RAM')
            self.images=[]
            for i,sample in enumerate(self.samples):
                path, target = sample
                if i%100 ==0:
                    sys.stdout.write('load {}\t{}\r '.format(i,path))
                    sys.stdout.flush()
                # image = self.loader(path)
                self.images.append(self.loader(path))
        print(os.path.basename(os.path.normpath(root)),'image number is ',self.__len__(),end=';\t')
        print('each class:',min(dict.values()),'-',max(dict.values()))
        print('--------------------------------')

    def _find_classes(self, dir):
        """
        在数据集中查找类别文件夹

        Args:
            dir (string): 根目录路径

        Returns:
            tuple: (classes, class_to_idx) 其中classes是相对于(dir)的类别名称，class_to_idx是字典

        Ensures:
            确保没有类别是另一个类别的子目录
        """
        if sys.version_info >= (3, 5):
            # Python 3.5及以上版本使用更快的os.scandir
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()  # 对类别名称进行排序
        class_to_idx = {classes[i]: i for i in range(len(classes))}  # 创建类别名称到索引的映射
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        获取指定索引的数据样本

        Args:
            index (int): 样本索引

        Returns:
            tuple: (sample, target, index) 其中sample是变换后的样本，target是目标类别索引
        """
        path, target = self.samples[index]  # 获取路径和目标标签

        if self.cached == False:
            # 如果未启用缓存，则从磁盘加载图像
            sample = self.loader(path)
        else:
            # 如果启用缓存，则从内存中获取图像
            sample = self.images[index]
        if self.transform is not None:
            sample = self.transform(sample)  # 应用数据变换
        if self.target_transform is not None:
            target = self.target_transform(target)  # 应用目标变换

        return sample, target, index  # 返回样本、目标和索引

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    """使用PIL加载图像文件
    
    Args:
        path: 图像文件路径
        
    Returns:
        PIL.Image: RGB格式的图像对象
    """
    # 以文件方式打开路径，避免ResourceWarning
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')  # 转换为RGB格式

def default_loader(path):
    """默认图像加载器 - 使用PIL加载器
    
    Args:
        path: 图像文件路径
        
    Returns:
        PIL.Image: RGB格式的图像对象
    """
    return pil_loader(path)

class Imagefolder_modified(DatasetFolder):
    """修改的图像文件夹加载器 - 按类别目录组织图像的数据集
    
    图像按以下方式组织: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): 根目录路径
        transform (callable, optional): 对PIL图像进行变换的函数/变换器
            返回变换后的版本，例如``transforms.RandomCrop``
        target_transform (callable, optional): 对目标进行变换的函数/变换器
        loader (callable, optional): 根据路径加载图像的函数
        is_valid_file (callable, optional): 检查图像文件是否有效的函数
            （用于检查损坏文件）
        cached (bool, optional): 是否将所有图像一次性加载到RAM中
        number (int, optional): 每个类别使用的样本数量限制

     Attributes:
        classes (list): 类别名称列表
        class_to_idx (dict): 包含(类别名称, 类别索引)项的字典
        imgs (list): (图像路径, 类别索引)元组列表
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, cached=False, number=None):
        super(Imagefolder_modified, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                                   cached=cached,
                                                   number=number)
        self.imgs = self.samples
