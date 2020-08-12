from PIL import Image,ImageFilter
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)

def get_size(file):
    # 获取文件大小:KB
    size = os.path.getsize(file)
    return size / 1024

def get_outfile(infile, outfile):
    if outfile:
        return outfile
    dir, suffix = os.path.splitext(infile)
    dir='EightAndDualAttention/'
    outfile = '{}out{}'.format(dir, suffix)
    return outfile

def compress_image(infile, outfile='', mb=20, step=10, quality=80):
    """不改变图片尺寸压缩到指定大小
    :param infile: 压缩源文件
    :param outfile: 压缩文件保存地址
    :param mb: 压缩目标，KB
    :param step: 每次调整的压缩比率
    :param quality: 初始压缩比率
    :return: 压缩文件地址，压缩文件大小
    """
    o_size = get_size(infile)
    if o_size <= mb:
        return infile
    outfile = get_outfile(infile, outfile)
    while o_size > mb:
        im = Image.open(infile)
        im.save(outfile, quality=quality)
        if quality - step < 0:
            break
        quality -= step
        o_size = get_size(outfile)
    return outfile

# def compress_image(im, o_size, mb=10, step=5, quality=95):
#     """不改变图片尺寸压缩到指定大小
#     :param infile: 压缩源图片
#     :param outfile: 压缩文件保存地址
#     :param mb: 压缩目标 倍数
#     :param step: 每次调整的压缩比率
#     :param quality: 初始压缩比率
#     :return: 压缩文件地址，压缩文件大小
#     """
#     mb=o_size/mb
#     if o_size <= mb:
#         return im
#     while o_size > mb:
#         im.save('1.jpg', quality=quality)
#         if quality - step < 0:
#             break
#         quality -= step
#         o_size = get_size('1.jpg')
#         im = Image.open('1.jpg')
#     return im

def resize_image(infile, outfile='', x_s=1376):
    """修改图片尺寸
    :param infile: 图片源文件
    :param outfile: 重设尺寸文件保存地址
    :param x_s: 设置的宽度
    :return:
    """
    im = Image.open(infile)
    x, y = im.size
    y_s = int(y * x_s / x)
    out = im.resize((x_s, y_s), Image.ANTIALIAS)
    outfile = get_outfile(infile, outfile)
    out.save(outfile)


if __name__ == '__main__':
    compress_image('/home/libiao/PycharmProjects/texture_filler2/275ILSVRC2012_test_00007007_1578321740.jpg')
    # resize_image(r'D:\learn\space.jpg')
    print(get_size('/home/libiao/PycharmProjects/texture_filler2/275ILSVRC2012_test_00007007_1578321740-out.jpg'))