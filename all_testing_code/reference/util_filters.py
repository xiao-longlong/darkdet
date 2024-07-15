import math
import cv2
import tensorflow as tf
import os
import sys
'''
output states:
    0: has rewards?
    1: stopped?
    2: num steps
    3:
'''
STATE_REWARD_DIM = 0
STATE_STOPPED_DIM = 1
STATE_STEP_DIM = 2
STATE_DROPOUT_BEGIN = 3

# wxl：为export文件配置路径，这个需要修改为合适的，但是没有调用。
def get_expert_file_path(expert):
  expert_path = 'data/artists/fk_%s/' % expert
  return expert_path

# From github.com/OlavHN/fast-neural-style
# wxl：正太分布中的标准化过程。
def instance_norm(x):
  epsilon = 1e-9
  mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
  return (x - mean) / tf.sqrt(var + epsilon)

# wxl：将states信息加入到net的最后一个维度的的第一个位置
def enrich_image_input(cfg, net, states):
  if cfg.img_include_states:
    print(("states for enriching", states.shape))
    states = states[:, None, None, :] + (net[:, :, :, 0:1] * 0)
    net = tf.concat([net, states], axis=3)
  return net


# based on https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary

# wxl：构建了一个字典类，此字典类继承自dict标准库。
# wxl：此字典类实现了三个隐式的属性调用方法，分别是访问属性，添加属性，删除属性，都可以像类中的属性一样直接操作，而服务的是字典键值。
# wxl：举例说明：
#       访问属性：obj.attr -> obj[attr]
#       添加属性：obj.attr = x -> obj[attr] = x
#       删除属性：del obj.attr -> del obj[attr] & value

class Dict(dict):
  """
    Example:
    m = Dict({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
  """

  def __init__(self, *args, **kwargs):
    super(Dict, self).__init__(*args, **kwargs)
    for arg in args:
      if isinstance(arg, dict):
        for k, v in arg.items():
          self[k] = v

    if kwargs:
      for k, v in kwargs.items():
        self[k] = v

  # wxl：访问Dict的属性。由于Dict是字典类型的类,其属性直接用键值处理，而不使用类的.方法访问。
  # wxl：其中的成员应该只有字典的键值，而不具有类的任何方法和属性。
  # wxl：这里是为Dict提供了两种访问属性的方法，通过键值访问，或者通过.访问。
  
  # wxl：obj.attr -> obj[attr]
  def __getattr__(self, attr):
    return self[attr]

  # wxl：obj.attr = x -> obj[attr] = x
  # wxl：底层还是调用的dict库中的方法实现的。
  def __setattr__(self, key, value):
    self.__setitem__(key, value)
  def __setitem__(self, key, value):
    super(Dict, self).__setitem__(key, value)
    self.__dict__.update({key: value})

  # wxl：del obj.attr时调用
  # wxl：底层通过dict库的方法实现的
  def __delattr__(self, item):
    self.__delitem__(item)
  def __delitem__(self, key):
    super(Dict, self).__delitem__(key)
    del self.__dict__[key]

# wxl：将image制作成grid。
# wxl：做法是将一个batch_size的image拼接，拼接的结果即可成为grid。
# wxl：同时每张小图片在边缘处有拓延。
def make_image_grid(images, per_row=8, padding=2):
  # wxl：带扩充的images是[batch_size, image_height, image_width, channels]
  npad = ((0, 0), (padding, padding), (padding, padding), (0, 0))
  images = np.pad(images, pad_width=npad, mode='constant', constant_values=1.0)
  # wxl：确保在一个batch_size下的图像数目可以被per_row整除。
  # wxl：想要得到多行数据，每一行上数据是满的。
  assert images.shape[0] % per_row == 0
  num_rows = images.shape[0] // per_row
  image_rows = []
  # wxl：将一个images组先h方向stack，再v方向stack一下，将多张图片拼接成了一个大图片
  for i in range(num_rows):
    image_rows.append(np.hstack(images[i * per_row:(i + 1) * per_row]))
  return np.vstack(image_rows)

# wxl：在这里image是形状是[height, width, channels]
# wxl：裁剪出图像中长边中间部分，使剩余部分与短边等长
# wxl：返回的是中间的部分
def get_image_center(image):
  if image.shape[0] > image.shape[1]:
    start = (image.shape[0] - image.shape[1]) // 2
    image = image[start:start + image.shape[1], :]

  if image.shape[1] > image.shape[0]:
    start = (image.shape[1] - image.shape[0]) // 2
    image = image[:, start:start + image.shape[0]]
  return image

# wxl：rotate_image函数的作用，如英文介绍所言，旋转图像并将图像中心与扩充后的图像中心重合
# wxl：空白的部分由0元素补充
def rotate_image(image, angle):
  """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

  # Get the image size
  # No that's not an error - NumPy stores image matricies backwards
  # wxl：numpy中图像是转置的，以元组形式得到图像中心
  image_size = (image.shape[1], image.shape[0])
  image_center = tuple(np.array(image_size) // 2)

  # Convert the OpenCV 3x2 rotation matrix to 3x3
  # wxl：通过getRotationMatrix2D方法得到3x2的旋转矩阵，并通过np.vstack得到3x3的旋转矩阵
  rot_mat = np.vstack(
      [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])

# wxl：3x3的矩阵中，前2x2的部分是需要的?
  rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

  # Shorthand for below calcs
  image_w2 = image_size[0] * 0.5
  image_h2 = image_size[1] * 0.5

  # Obtain the rotated coordinates of the image corners
  # wxl：.A是将matrix 转换成 array
  # wxl：.A[0]是将array的第一行取出，第一行是3D坐标在2D的投影
  # wxl：这里是计算出图像的四个角点转换后的结果
  rotated_coords = [
      (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
      (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
      (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
      (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
  ]

  # Find the size of the new image
  # wxl：取出旋转后的x,y的值，并且依据x，y是否大于0分类成，pos & neg两类
  # wxl：只要旋转，且取左上角为（0，0），就一定有小于0的坐标
  x_coords = [pt[0] for pt in rotated_coords]
  x_pos = [x for x in x_coords if x > 0]
  x_neg = [x for x in x_coords if x < 0]

  y_coords = [pt[1] for pt in rotated_coords]
  y_pos = [y for y in y_coords if y > 0]
  y_neg = [y for y in y_coords if y < 0]

  # wxl：取出四个边界坐标
  right_bound = max(x_pos)
  left_bound = min(x_neg)
  top_bound = max(y_pos)
  bot_bound = min(y_neg)

  # wxl：通过边界尺寸计算出新的图像宽高
  new_w = int(abs(right_bound - left_bound))
  new_h = int(abs(top_bound - bot_bound))


  # We require a translation matrix to keep the image centred
  # wxl：计算平移变换矩阵，保证原本图像旋转后的中心位置在新图像的中心
  trans_mat = np.matrix([[1, 0, int(new_w * 0.5 - image_w2)],
                         [0, 1, int(new_h * 0.5 - image_h2)], [0, 0, 1]])

  # Compute the tranform for the combined rotation and translation
  # wxl：矩阵变换的好处就是具有传递性，因此可以将多次矩阵变换的公式直接相乘，并取出前面的部分
  affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

  # Apply the transform

  # affine_mat = np.array([
  #   [cos_theta, -sin_theta, tx],
  #   [sin_theta, cos_theta, ty],
  #   [0, 0, 1]
  # ])
  # wxl：标准的旋转平移矩阵是3x3的，由于这里只在平面上对图像进行仿射操作，所以只需要2x3的矩阵。
  # x' = a11 * x + a12 * y + b1
  # y' = a21 * x + a22 * y + b2
  result = cv2.warpAffine(
      image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)

  return result

# wxl：计算的是一个特殊的矩形，以解决如下问题。
# wxl：在图像旋转时，旋转框也会跟着旋转，如果直接用其最小正外接矩形当作标注框，矩形框会偏大。
# wxl：此矩形是通过经验公式，结合矩形框的旋转角度 & 矩形框的宽高，设计出一种能几乎等效替代原本矩形框的正矩形框。
def largest_rotated_rect(w, h, angle):
  """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

  # wxl：math.floor(angle / (math.pi / 2))，角度除以pi/2，并取整，得到对应的象限
  # wxl：象限需要与3进行位运算，保证象限结果在0，1，2，3四种情况中
  quadrant = int(math.floor(angle / (math.pi / 2))) & 3

  # wxl：下面两行的综合作用：使角度是与x轴的夹角（有可能是x轴负半轴），且保证夹角是锐角
  # wxl：0，2象限，直接取对应的角度；1，3象限，取补角（大于pi的也取补角，取出来的是一个负的补角）
  sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
  alpha = (sign_alpha % math.pi + math.pi) % math.pi

  # wxl：求最小正外接矩形的尺寸
  bb_w = w * math.cos(alpha) + h * math.sin(alpha)
  bb_h = w * math.sin(alpha) + h * math.cos(alpha)

  # wxl：这里必然有问题，不能判断前后，结果都是pi/4
  # wxl：atan2是解算点（x,y）到原点（0，0）的角度（弧度制）
  gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

  # wxl：delta的物理意义是，pi - 斜框到正框的最小旋转角 - pi/4
  # wxl：根据物理意义，delta必定大于pi/2
  delta = math.pi - alpha - gamma

  # wxl：结果取长边
  length = h if (w < h) else w

  # wxl：长边作斜边的直角三角形的直角长边，下面的就是经验公式了
  d = length * math.cos(alpha)
  a = d * math.sin(alpha) / math.sin(delta)

  y = a * math.cos(gamma)
  x = y * math.tan(gamma)

  return (bb_w - 2 * x, bb_h - 2 * y)


def crop_around_center(image, width, height):
  """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """
  # wxl：统计输入的图片的尺寸
  image_size = (image.shape[1], image.shape[0])
  image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

  # wxl：保证需要剪裁的图像尺寸不能大于图像原本尺寸
  if (width > image_size[0]):
    width = image_size[0]

  if (height > image_size[1]):
    height = image_size[1]
  # wxl：沿着图像中心向外剪裁
  x1 = int(image_center[0] - width * 0.5)
  x2 = int(image_center[0] + width * 0.5)
  y1 = int(image_center[1] - height * 0.5)
  y2 = int(image_center[1] + height * 0.5)

  return image[y1:y2, x1:x2]


# angle: degrees
# wxl：将旋转和剪裁再次封装，剪裁是为了让旋转后图片不会过大
def rotate_and_crop(image, angle):
  image_width, image_height = image.shape[:2]
  image_rotated = rotate_image(image, angle)
  image_rotated_cropped = crop_around_center(image_rotated,
                                             *largest_rotated_rect(
                                                 image_width, image_height,
                                                 math.radians(angle)))
  return image_rotated_cropped


def lrelu(x, leak=0.2, name="lrelu"):
  with tf.variable_scope(name):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


# clamps to 0, 1 with leak
def double_lrelu(x, leak=0.1, name="double_lrelu"):
  with tf.variable_scope(name):
    return tf.minimum(tf.maximum(leak * x, x), leak * x - (leak - 1))


# clamp to lower, upper; leak is RELATIVE
def leaky_clamp(x, lower, upper, leak=0.1, name="leaky_clamp"):
  with tf.variable_scope(name):
    x = (x - lower) / (upper - lower)
    return tf.minimum(tf.maximum(leak * x, x), leak * x -
                      (leak - 1)) * (upper - lower) + lower


class Tee(object):

  def __init__(self, name):
    self.file = open(name, 'w')
    self.stdout = sys.stdout
    self.stderr = sys.stderr
    sys.stdout = self
    sys.stderr = self

  def __del__(self):
    self.file.close()

  def write(self, data):
    self.file.write(data)
    self.stdout.write(data)
    self.file.flush()
    self.stdout.flush()

  def write_to_file(self, data):
    self.file.write(data)

  def flush(self):
    self.file.flush()


def rgb2lum(image):
  image = 0.27 * image[:, :, :, 0] + 0.67 * image[:, :, :,
                                                  1] + 0.06 * image[:, :, :, 2]
  return image[:, :, :, None]


def tanh01(x):
  return tf.tanh(x) * 0.5 + 0.5


def tanh_range(l, r, initial=None):

  def get_activation(left, right, initial):

    def activation(x):
      if initial is not None:
        bias = math.atanh(2 * (initial - left) / (right - left) - 1)
      else:
        bias = 0
      return tanh01(x + bias) * (right - left) + left

    return activation

  return get_activation(l, r, initial)


def merge_dict(a, b):
  ret = a.copy()
  for key, val in list(b.items()):
    if key in ret:
      assert False, 'Item ' + key + 'already exists'
    else:
      ret[key] = val
  return ret


def lerp(a, b, l):
  return (1 - l) * a + l * b


def read_tiff16(fn):
  import tifffile
  import numpy as np
  img = tifffile.imread(fn)
  if img.dtype == np.uint8:
    depth = 8
  elif img.dtype == np.uint16:
    depth = 16
  else:
    print("Warning: unsupported data type {}. Assuming 16-bit.", img.dtype)
    depth = 16
  # wxl：将0-2^depth 转换到 0-1 的范围。
  return (img * (1.0 / (2**depth - 1))).astype(np.float32)

# wxl：exec是一个动态执行函数，动态执行某一python代码。
# wxl：如下exec操作的是动态执行from xx import xx代码。
# wxl：exec的第二个参数是用字典指定命名空间。
# wxl：命名空间的含义是代码会再命名空间中运行，代码定义的变量会在命名空间对应的字典中存储。
# wxl：chatgpt如是说
# wxl：在执行from config_example import cfg时，Python会导入config_example模块中的cfg对象，并在当前的全局命名空间中创建一个名为cfg的变量。由于这个命名空间是由scope字典指定的，所以cfg变量会被添加到scope字典中。
def load_config(config_name):
  scope = {}
  exec ('from config_%s import cfg' % config_name, scope)
  return scope['cfg']


# ======================================================================================================================
# added by Hao He
# ======================================================================================================================
def get_artist_batch(folder, size=128, num=64):
  import os
  js = os.listdir(folder)
  np.random.shuffle(js)
  imgs = np.zeros((num, size, size, 3))
  for i, jpg in enumerate(js[:num]):
    img = cv2.imread(folder + '/' + jpg)
    img = get_image_center(img) / 255.
    imgs[i] = cv2.resize(img, dsize=(size, size))
  return imgs

# wxl：数据集图片，打乱，裁剪出中心，尺缩，拼成大图
def show_artist_subnails(folder, size=128, num_row=8, num_column=8):
  imgs = get_artist_batch(folder, size, num_row * num_column)
  return make_image_grid(imgs, per_row=num_row)


def np_tanh_range(l, r):

  def get_activation(left, right):

    def activation(x):
      return np.tanh(x) * (right - left) + left

    return activation

  return get_activation(l, r)


class WB2:
  # wxl：将特征图从(-∞,+∞)归一化到指定范围。
  def filter_param_regressor(self, features):
    log_wb_range = np.log(5)
    color_scaling = np.exp(
        np_tanh_range(-log_wb_range, log_wb_range)(features[:, :3]))
    # There will be no division by zero here unless the WB range lower bound is 0
    return color_scaling

  # wxl：一幅图像，求出其流明。同一幅图像，乘积参数以得到处理后图像，并求出其流明。
  # wxl：处理后图像，归一化到原图像流明，（通过除自身流明，乘原图像流明实现）
  # wxl：下面的函数的作用，将图像用参数处理，并将图像亮度归一化到和原图像一致
  def process(self, img, param):
    lum = (img[:, :, :, 0] * 0.27 + img[:, :, :, 1] * 0.67 +
           img[:, :, :, 2] * 0.06 + 1e-5)[:, :, :, None]
    tmp = img * param[:, None, None, :]
    tmp = tmp / (tmp[:, :, :, 0] * 0.27 + tmp[:, :, :, 1] * 0.67 +
                 tmp[:, :, :, 2] * 0.06 + 1e-5)[:, :, :, None] * lum
    return tmp

# wxl：函数的作用是将制定的图像进行降级处理
def degrade_images_in_folder(
    folder,
    dst_folder_suffix,
    LIGHTDOWN=True,
    UNBALANCECOLOR=True,):
  # wxl：可以看出输出的文件夹是放在输入的文件夹下的
  import os
  js = os.listdir(folder)
  dst_folder = folder + '-' + dst_folder_suffix

  # wxl：创建目标文件夹
  try:
    os.mkdir(dst_folder)
  except:
    print('dir exist!')
  print('in ' + dst_folder)


  num = 3
  for j in js:
    img = cv2.imread(folder + '/' + j) / 255.
    # wxl：函数的作用是将图像变暗，一张图像会得到4张图像，前三张是通过gamma变换得来的，最后一张是通过除最大值得来的
    if LIGHTDOWN:
      for _ in range(num - 1):
        # wxl：这一步处理的结果，整体上是将图像变暗的，但是通过绘制曲线会发现，个别点可能出现变亮的情况
        out = pow(img, np.random.uniform(0.4, 0.6)) * np.random.uniform(
            0.25, 0.5)
        # wxl：将处理的结果保存
        cv2.imwrite(dst_folder + '/' + ('L%d-' % _) + j, out * 255.)
      # wxl：这里处理是依据最亮点将图像变暗，结果一定是变暗的。
      out = img * img
      out = out * (1.0 / out.max())
      # wxl：将结果保存下来，并记为最大数字的结果
      cv2.imwrite(dst_folder + '/' + ('L%d-' % num) + j, out * 255.)
    
    # wxl：
    if UNBALANCECOLOR:
      filter = WB2()
      # wxl：列表的用法，作用是将列表扩充几倍
      outs = np.array([img] * num)
      # wxl：feature.shape = 3x3,每个元素的范围是0~1
      features = np.abs(np.random.rand(num, 3))
      # wxl：将feature的结果 作为在WB2中处理outs的依据
      for _, out in enumerate(
          filter.process(outs, filter.filter_param_regressor(features))):
        # print out.max()
        # wxl：处理的结果还需要缩放再保存
        out /= out.max()
        out *= np.random.uniform(0.7, 1)
        cv2.imwrite(dst_folder + '/' + ('C%d-' % _) + j, out * 255.)

# wxl：将特征与图像堆叠起来并保存结果
def vis_images_and_indexs(images, features, dir, name):
  # indexs = np.reshape(indexs, (len(indexs),))
  # print('visualizing images and indexs: ', images.shape, indexs.shape)
  id_imgs = []
  # wxl：feature是字符形式。
  # wxl：将feature放置在合适的白图片位置
  for feature in features:
    img = np.ones((64, 64, 3))
    cv2.putText(img,
                str(feature), (4, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.25,
                (1.0, 0.0, 0.0))
    
    id_imgs.append(img)
  id_imgs = np.stack(id_imgs, axis=0)
  # print('id imgs: ', id_imgs.shape)

  # wxl：沿着第一个维度堆叠起来
  vis_imgs = np.vstack([images, id_imgs])
  image = make_image_grid(vis_imgs, per_row=images.shape[0])
  vis_dir = dir
  try:
    os.mkdir(vis_dir)
  except:
    pass
  cv2.imwrite(os.path.join(vis_dir, name + '.png'), image[:, :, ::-1] * 255.0)

# wxl：有不同种类的数据集，从文件中读取信息，或直接返回一个信息列表
def read_set(name):
  if name == 'u_test':
    fn = 'data/folds/FiveK_test.txt'
    need_reverse = False
  elif name == 'u_amt':
    fn = 'data/folds/FiveK_test_AMT.txt'
    need_reverse = False
  elif name == '5k':  # add by hao
    return list(range(1, 5001))
  elif name == '2k_train':
    fn = 'data/folds/FiveK_train_first2k.txt'
    need_reverse = False
  elif name == '2k_target':
    fn = 'data/folds/FiveK_train_second2k.txt'
    need_reverse = False
  else:
    assert False, name + ' not found'

  l = []
  ln = 0
  with open(fn, 'r') as f:
    for i in f:
      if i[0] != '#':
        try:
          i = int(i)
          ln += 1
          l.append(i)
        except Exception as e:
          print(e)
          pass
  if need_reverse:
    l = list(set(range(1, 5001)) - set(l))
  return l


'''
    util_image.py
    Copyright (c) 2014     Zhicheng Yan (zhicheng.yan@live.com)
        modified 2017  by Yuanming Hu  (yuanmhu@gmail.com)
        note that some of the color space conversions are NOT exact, like gamma 1.8 or 2.2
'''

import numpy as np
from skimage import color
import tifffile as tiff


class UtilImageError(Exception):
  pass


''' undo gamma correction '''

# wxl：ppRGB图像的线性化，使用幂指数函数进行gamma校正
def linearize_ProPhotoRGB(pp_rgb, reverse=False):
  if not reverse:
    gamma = 1.8
  else:
    gamma = 1.0 / 1.8
  pp_rgb = np.power(pp_rgb, gamma)
  return pp_rgb

# wxl：这个函数叫做白点校正，将图像从一种亮度映射成为另外一种亮度
# wxl：映射的原则是，将原本的三个通道通过不同的加权方式，变为新的三个通道
def XYZ_chromatic_adapt(xyz, src_white='D65', dest_white='D50'):
  if src_white == 'D65' and dest_white == 'D50':
    M = [[1.0478112, 0.0228866, -0.0501270], \
         [0.0295424, 0.9904844, -0.0170491], \
         [-0.0092345, 0.0150436, 0.7521316]]
  elif src_white == 'D50' and dest_white == 'D65':
    M = [[0.9555766, -0.0230393, 0.0631636], \
         [-0.0282895, 1.0099416, 0.0210077], \
         [0.0122982, -0.0204830, 1.3299098]]
  else:
    raise ValueError('invalid pair of source and destination white reference %s,%s') \
          % (src_white, dest_white)
  M = np.array(M)
  sp = xyz.shape
  assert sp[2] == 3
  # wxl：以第一种M为例。
  # wxl：channel'[1] = 1.0478112 * channel[1] + 0.0228866 * channel[2] - 0.0501270 * channel[3]
  # wxl：channel'[2] = 0.0295424 * channel[1] + 0.9904844 * channel[2] - 0.0170491 * channel[3]
  # wxl：channel'[3] = -0.0092345 * channel[1] + 0.0150436 * channel[2] + 0.7521316 * channel[3]
  xyz = np.transpose(np.dot(M, np.transpose(xyz.reshape((sp[0] * sp[1], 3)))))
  return xyz.reshape((sp[0], sp[1], 3))


# pp_rgb float in range [0,1], linear ProPhotoRGB
# refernce white is D50
# wxl：将PPRGB颜色空间的图像转变为XYZ颜色空间，具体来说是将不同通道的加权映射
def ProPhotoRGB2XYZ(pp_rgb, reverse=False):
  if not reverse:
    M = [[0.7976749, 0.1351917, 0.0313534], \
         [0.2880402, 0.7118741, 0.0000857], \
         [0.0000000, 0.0000000, 0.8252100]]
  else:
    M = [[1.34594337, -0.25560752, -0.05111183], \
         [-0.54459882, 1.5081673, 0.02053511], \
         [0, 0, 1.21181275]]
  M = np.array(M)
  sp = pp_rgb.shape
  xyz = np.transpose(
      np.dot(M, np.transpose(pp_rgb.reshape((sp[0] * sp[1], sp[2])))))
  return xyz.reshape((sp[0], sp[1], 3))


''' normalize L channel so that minimum of L is 0 and maximum of L is 100 '''

# wxl：将图像从原本的最小到最大的区间映射到，0~100
def normalize_Lab_image(lab_image):
  h, w, ch = lab_image.shape[0], lab_image.shape[1], lab_image.shape[2]
  assert ch == 3
  lab_image = lab_image.reshape((h * w, ch))
  L_ch = lab_image[:, 0]
  L_min, L_max = np.min(L_ch), np.max(L_ch)
  #     print 'before normalization L min %f,Lmax %f' % (L_min,L_max)
  scale = 100.0 / (L_max - L_min)
  lab_image[:, 0] = (lab_image[:, 0] - L_min) * scale
  #     print 'after normalization L min %f,Lmax %f' %\
  (np.min(lab_image[:, 0]), np.max(lab_image[:, 0]))
  return lab_image.reshape((h, w, ch))


''' white reference 'D65' '''

# wxl：将图像使用上述多种方法转换，为合适的xyz图像并
def read_tiff_16bit_img_into_XYZ(tiff_fn, exposure=0):
  pp_rgb = tiff.imread(tiff_fn)
  pp_rgb = np.float64(pp_rgb) / (2**16 - 1.0)
  if not pp_rgb.shape[2] == 3:
    print('pp_rgb shape', pp_rgb.shape)
    raise UtilImageError('image channel number is not 3')
  pp_rgb = linearize_ProPhotoRGB(pp_rgb)
  pp_rgb *= np.power(2, exposure)
  xyz = ProPhotoRGB2XYZ(pp_rgb)
  xyz = XYZ_chromatic_adapt(xyz, src_white='D50', dest_white='D65')
  return xyz

# wxl：将图像转换到LAB颜色空间
def ProPhotoRGB2Lab(img):
  if not img.shape[2] == 3:
    print('pp_rgb shape', img.shape)
    raise UtilImageError('image channel number is not 3')
  img = linearize_ProPhotoRGB(img)
  xyz = ProPhotoRGB2XYZ(img)
  lab = color.xyz2lab(xyz)
  return lab

# wxl：将图像直接转到LAB颜色空间
def linearProPhotoRGB2Lab(img):
  if not img.shape[2] == 3:
    print('pp_rgb shape', img.shape)
    raise UtilImageError('image channel number is not 3')
  xyz = ProPhotoRGB2XYZ(img)
  lab = color.xyz2lab(xyz)
  return lab
