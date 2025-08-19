import shutil  # 导入shutil模块，用于文件和目录的复制、移动、删除等操作
from skimage import io  # 从skimage库导入io模块，用于图像的读取和保存等IO操作
import os.path  # 导入os.path模块，提供处理文件路径的函数
import torch  # 导入torch模块，即PyTorch深度学习框架，用于构建和训练神经网络
import pickle  # 导入pickle模块，用于对象的序列化和反序列化，方便数据的存储和读取
import random  # 导入random模块，提供生成随机数的功能，用于数据随机处理等场景
import numpy as np  # 导入numpy库并简写为np，用于进行科学计算，处理多维数组和矩阵运算
from tqdm import tqdm  # 从tqdm库导入tqdm，用于创建进度条，方便查看循环等操作的执行进度
# import face_alignment  # 注释掉的代码：从face_alignment库导入相关功能，可能用于人脸对齐处理
from PIL import Image  # 从PIL库导入Image模块，用于图像处理，如打开、保存、转换图像格式等
import glob  # 导入glob模块，用于查找符合特定模式的文件路径名
import cv2  # 导入cv2模块，即OpenCV库，用于计算机视觉相关操作，如图像处理、视频分析等
from torch.utils.data import DataLoader, Dataset  # 从torch.utils.data导入DataLoader和Dataset，用于构建自定义数据集和数据加载器
import csv  # 导入csv模块，用于处理CSV格式的文件，进行数据的读写操作
import sys  # 导入sys模块，用于访问Python解释器的相关变量和功能，如系统路径等
sys.path.append("..")  # 将当前脚本的父目录添加到系统路径中，以便导入父目录下的模块
curr_path = os.path.abspath(os.path.dirname(__file__))  # 获取当前脚本文件的绝对路径F:\code\FreeAvatar-1\free_avatar\dataset，其中__file__当前执行脚本的文件名，os.path.dirname(__file__)获取当前脚本文件所在的目录路径
sys.path.append(os.path.join(curr_path, ".."))  # 将当前脚本父目录的父目录F:\code\FreeAvatar-1\free_avatar添加到系统路径中，扩展模块搜索路径
from utils.tools import load_rigs_to_cache  # 从utils.tools模块导入load_rigs_to_cache函数，用于加载rigs数据到缓存

# 计算数据的均值和标准差(未调用，且没有返回值）
def statistic(data):
    std = np.std(data) # data的标准差
    mean = np.mean(data) # data的平均值
    return

# 定义ABAWDataset2类，继承自PyTorch的Dataset类，用于构建自定义数据集
class ABAWDataset2(Dataset):
    # 初始化方法，用于设置数据集的基本参数和加载数据
    # 参数说明：
    # - root_path: 数据根路径
    # - character: 角色名称（如L36_233、L36_230等）
    # - data_split: 数据集分割（如'train'、'test'）
    # - only_render: 是否只使用渲染数据
    # - transform: 图像预处理变换
    # - random_flip: 是否随机翻转
    # - do_norm: 是否进行归一化
    # - use_ldmk: 是否使用 landmarks（人脸关键点）
    # - return_rigs: 是否返回rig数据（可能是面部驱动参数）
    # - n_rigs: rig参数的维度
    # - faceware_ratio: Faceware数据的采样比例
    # - img_postfix: 图像文件后缀（如'.jpg'）
    def __init__(self, root_path, character,data_split, only_render=False, transform=None, random_flip=False, do_norm=True, 
                 use_ldmk=False, return_rigs=False, n_rigs=-1,faceware_ratio=0.1, img_postfix='.jpg'):
        self.character = character  # 存储角色名称
        self.return_rigs = return_rigs  # 标记是否返回rig数据
        self.n_rigs = n_rigs  # 存储rig参数维度
        expr_fuxi_root = '/project/qiuf/Expr_fuxi/images_ttg_2024'  # 伏羲表情数据路径
        # 构建faceware数据根路径（将root_path中的'/images'替换为'/faceware'）
        self.faceware_root = os.path.join(root_path.replace('/images', '/faceware'))
        self.faceware_root_230 = '/project/qiuf/DJ01/L36_230/faceware'  # 230角色的faceware数据路径
        ttg_dy_root = '/project/qiuf/Expr_fuxi/images_ttg'  # ttg动态数据路径
        # 音频驱动的伏羲人脸数据路径
        fuxi_audio_data ='/project/qiuf/L36_drivendata/anrudong_2023_08_08_29_23_10_03_05_speaking_30fps/crop_face_processed'
        # 需要排除的文件夹列表（可能是无效或不需要的数据）
        exclude_folder = [ 'L36Randsample', 'bichiyin_zuhe', 'L36face234_ZHY_PM_c101_230097_xia_51566_1',
                          'L36face234_ZHY_PM_c102_234011_banxiangzi_51560_1'] # 'qiufeng011'

        # 若角色是L36系列（233/234/230等），加载对应的真实数据路径
        if character.lower() in ['l36_233', 'l36_234','l36_230', 'l36_230_61']:
            # 基础真实数据路径列表（包含多个来源的人脸数据）
            real_data = [
            '/data/Workspace/Rig2Face/data/yinxiaonv3',
            '/data/Workspace/Rig2Face/data/jiayang01',
            '/data/Workspace/Rig2Face/data/dongbu_kuazhang',
            '/data/Workspace/Rig2Face/data/singing',
            '/data/Workspace/Rig2Face/data/blackbro',
            '/data/Workspace/Rig2Face/data/Donbu_yinsu_31115_3_zhy',
            '/data/Workspace/Rig2Face/data/wjj01',
            '/data/Workspace/Rig2Face/data/qiufeng02',
            '/data/Workspace/Rig2Face/data/linjie_expr_test',
            '/project/qiuf/Expr_fuxi/images_old_50fps/C0016',
            '/project/qiuf/L36_drivendata/L36_233_20231106_43th/crop_face/L36face233_CYY_PM_bxy02_52252_1',
            '/project/qiuf/L36_drivendata/L36_233_20231106_43th/crop_face/L36face233_CYY_PM_bj_wjl_52251_1',
            '/project/qiuf/L36_drivendata/L36_233_20231106_43th/crop_face/L36face233_CYY_PM_bj_mm_52247_1',
            '/project/qiuf/L36_drivendata/L36_233_20231106_43th/crop_face/L36face233_CYY_PM_bj_lss_52250_1',
            '/project/qiuf/DJ01/L36/faceware/qiufeng011',
            '/project/qiuf/DJ01/L36/faceware/linjie_expr_test3',
            '/project/qiuf/DJ01/L36/faceware/linjie_expr_test3',
            '/project/qiuf/DJ01/L36/faceware/linjie_expr_test3',
            '/project/qiuf/DJ01/L36/faceware/wjj01',
            '/project/qiuf/DJ01/L36/faceware/wjj01',
            '/project/qiuf/DJ01/L36/faceware/qiufeng02',
            '/project/qiuf/DJ01/L36/faceware/qiufeng02',
            '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c05_02_55201_1',
            '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c05_02_55201_1',
            '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c05_02_55201_1',
            '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c08_55212_1',
            '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c08_55212_1',
            '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c08_55212_1',
            ]
            # 若角色是230或230_61，额外添加对应的真实数据路径（当前注释掉，可能暂不使用）
            if character.lower() in ['l36_230', 'l36_230_61']:
                real_data += [
                            #   '/project/qiuf/DJ01/L36_230/faceware/230_first',
                            #   '/project/qiuf/DJ01/L36_230/faceware/230_second'
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_huazhang_60069_2',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_huazhang_60069_2',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#233_fanghedeng02_59838_1',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#233_fanghedeng02_59838_1',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_qujia_nanwanjia_60071_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_qujia_nanwanjia_60071_4',
                            #   '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_qujia_nanwanjia_60071_4',
                              ]
            # 若角色是233或234，将faceware_root添加到真实数据路径                  
            elif character.lower() in ['l36_233', 'l36_234']:
                real_data += [self.faceware_root]
            
            # 若需要加入faceware数据，按比例采样文件夹并添加到真实数据
            # 233faceware
            # faceware_folder = []
            if faceware_ratio:
                # 按1/faceware_ratio的间隔采样faceware文件夹（控制数据量）
                faceware_folder = os.listdir(self.faceware_root)[::int(1//faceware_ratio)]
                
                # faceware_folder += [y for x in os.walk(self.faceware_root) for y in glob.glob(os.path.join(x[0], '*.jpg'))][::int(1//faceware_ratio)]
                # 将采样的文件夹路径添加到real_data
                real_data += [os.path.join(self.faceware_root, e) for e in faceware_folder]
            # 230数据
            # self.faceware_folder_230 = self.load_faceware_230()
            # real_data += list(self.faceware_folder_230.values())

            # JHM大脸男的数据加一些
            # JHM_root = '/project/qiuf/DJ01/L36_230/faceware/230_second'
            # JHM_folder = [os.path.join(JHM_root, e) for e in os.listdir(JHM_root) if '_JHM_' in e and not e.endswith('.mp4')][::3]
            # real_data += JHM_folder

            # 新的女生数据加一些
            # GIRL_root = '/project/qiuf/Expr_fuxi/luoshen_data_clip/crop_face'
            # GIRL_folder = [os.path.join(GIRL_root, e) for e in os.listdir(GIRL_root) if 'player1_' in e][::10]
            # real_data += GIRL_folder

            # fuxi expr data
            # expr_fuxi_data = os.listdir(expr_fuxi_root)
            # real_data += [os.path.join(expr_fuxi_root, fo) for fo in expr_fuxi_data]

            # 伏羲音频驱动数据：按1/3比例采样并添加到真实数据
            # 伏羲audio data
            # read_data += fuxi_audio_data
            expr_audio_data = os.listdir(fuxi_audio_data)[::3]
            real_data += [os.path.join(fuxi_audio_data, fo) for fo in expr_audio_data] 

            # cartoon 
            # cartoon_root = '/project/qiuf/expr-capture/data2/crop_face'
            # cartoon_data = os.listdir(cartoon_root)[::]
            # real_data += [os.path.join(cartoon_root, fo) for fo in cartoon_data] *100
            
        self.root_path = root_path  # 存储数据根路径
        self.transform = transform  # 存储图像预处理变换
        self.use_ldmk = use_ldmk  # 标记是否使用人脸关键点
        imgs_list = []  # 初始化图像路径列表
        self.rigs_list = []  # 初始化rig数据路径列表
        render_folders, render_folders_230 = [], []  # 渲染数据文件夹列表
        # 获取rig数据文件夹数量（用于构建图像列表文件名）
        n_rig_fold = len(os.listdir(root_path.replace('/images', '/rigs')))
        # 构建训练/测试图像列表文件路径（包含动作信息）
        image_list_action = os.path.join(root_path.replace('/images', ''), f'images_{data_split}_list_actions{n_rig_fold}.txt')
        image_list_action_230 = os.path.join(root_path.replace('/images', ''), f'images_{data_split}_list_actions_230.txt')
        # 若图像列表文件存在，读取并添加到imgs_list
        if os.path.exists(image_list_action):
            with open(image_list_action, 'r') as f:
                lines = f.readlines() # 读取所有行
            # lines = list(set(lines))
            # 提取渲染文件夹（去重）
            render_folders += list(set([l.split('/')[0].split('_202')[0] for l in lines]))
            # 按1/5比例采样图像路径并添加到列表（控制数据量）
            imgs_list += [os.path.join(root_path, l.strip()) for l in lines][::5]
        # if os.path.exists(image_list_action_230):
        #     with open(image_list_action_230, 'r') as f:
        #         lines_230 = f.readlines()
        #     render_folders_230 += list(set([l.split('/')[0] for l in lines_230]))
        #     imgs_list += [os.path.join(root_path.replace('/images', '/images_retarget_from_230'), l.strip()) for l in lines_230]
        # 若未通过列表文件获取图像，直接遍历root_path下的文件夹收集图像
        if not imgs_list:
            # 如果没有另外指定list action
            folders = os.listdir(root_path) # 获取根路径下的所有文件夹
            for folder in folders:
                # 若为文件夹，遍历其中的图像文件
                if os.path.isdir(os.path.join(root_path, folder)):
                    images = os.listdir(os.path.join(root_path, folder))
                    imgs_list += [os.path.join(root_path, folder, imgname) for imgname in images]
        
        # load real data
        # 若不只使用渲染数据，加载真实数据到图像列表
        if not only_render:
            for folder in real_data:
                # 收集文件夹下所有jpg和png图像（递归遍历子文件夹）
                images = [y for x in os.walk(folder) for y in glob.glob(os.path.join(x[0], '*.jpg'))]
                images += [y for x in os.walk(folder) for y in glob.glob(os.path.join(x[0], '*.png'))]
                # 对特定路径的图像按比例采样（减少数据量）
                if self.faceware_root_230 in folder:
                    images = images[::10]  # 每10张取1张
                elif expr_fuxi_root in folder:
                    images = images[::10]  # 每10张取1张
                # 按训练/测试分割数据
                if data_split in ['train', 'training']:
                    imgs_list += images[:int(0.9*len(images))]  # 取前90%作为训练集
                elif data_split in ['test']:
                    imgs_list += images[int(0.9*len(images)):]  # 取后10%作为测试集
        
        # 建立真实数据与渲染数据的映射关系（用于对齐）
        # 建立real data和render data之间的联系
        self.real2render = self._get_render_data_from_real_data(imgs_list, character)
        # self.real2render = {}
        
        self.imgs_list = imgs_list # 存储最终的图像路径列表
        # 若不只使用渲染数据，添加ABAW数据集的数据（按30%比例混合）
        if not only_render:
            abaw_data = self.read_abaw_data(data_split) # 读取ABAW数据
            random.shuffle(abaw_data) # 随机打乱
            # 取ABAW数据的前30%与现有数据混合（控制比例）
            abaw_data = abaw_data[:int(len(self.imgs_list)*0.3)]
            self.imgs_list += abaw_data[::]


        
        # self.imgs_list += self.read_MEAD_DATA(data_split, '/project/qiuf/MEAD/videos_res_no_smooth_selected_0511.txt')
        # self.imgs_list += self.read_MEAD_DATA(data_split, )
        # cache image data
        # self.img2rec = build_img2rec(root_path)
        # if return_rigs:
        #     self.rigs = load_rigs_to_cache(self.root_path.replace('/images', '/rigs'), n_rig=n_rigs)
        # self.rigs = load_rigs_to_cache(self.root_path.replace('/images', '/rigs'), n_rig=n_rigs)
        # rigs_array =  np.array(list(self.rigs.values()))
        # 过滤掉包含exclude_folder中文件夹的图像路径
        self.imgs_list = [e for e in self.imgs_list if e.split('/')[-2] not in exclude_folder]
        # 区分对称和非对称数据（可能与角色表情对称性相关）
        # 对称数据：不含'linjie'、'qiufeng'、'wjj'的图像
        symm = [e for e in self.imgs_list if 'linjie' not in e and 'qiufeng' not in e and 'wjj' not in e ]
        # 非对称数据：包含上述关键词的图像
        dissymm = [e for e in self.imgs_list if 'linjie' in e or 'qiufeng' in e or 'wjj' in e ]
        # 调整数据比例：对称数据每2张取1张，保留所有非对称数据
        self.imgs_list = symm[::2] + dissymm 
        # self.imgs_list = [e for e in self.imgs_list if 'linjie' in e and 'faceware' in e]
        # self.imgs_list = [e for e in self.imgs_list if 'linjie' in e]
        
        # 加载rig数据到缓存（从根路径的rigs文件夹）
        # 去除部分zhoumei数据
        self.rigs = load_rigs_to_cache(self.root_path.replace('/images', '/rigs'), n_rig=n_rigs)
        # 根据rig参数筛选特定表情的图像（增强特定特征的数据）
        # 筛选"zhoumei"相关表情（rig参数14>0.4）
        # imgs_list_zhoumei = [l for l in self.imgs_list if f"{l.split('/')[-2]}/{l.split('/')[-1].replace('0000.'+l.split('/')[-1].split('.')[-1], 'txt').replace(l.split('/')[-1].split('.')[-1], 'txt')}" in self.rigs and self.rigs[f"{l.split('/')[-2]}/{l.split('/')[-1].replace('0000.'+l.split('/')[-1].split('.')[-1], 'txt').replace(l.split('/')[-1].split('.')[-1], 'txt')}"][14]>0.3]
        imgs_list_zhoumei = [l for l in self.imgs_list if l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt') in self.rigs and 
                             self.rigs[l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt')][14]>0.4]
        # 筛选"minzui"相关表情（rig参数38>0.4）
        imgs_list_minzui = [l for l in self.imgs_list if l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt') in self.rigs and 
                             self.rigs[l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt')][38]>0.4]
        # 筛选"frownmouth"相关表情（rig参数36>0.4）
        imgs_list_frownmouth = [l for l in self.imgs_list if l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt') in self.rigs and 
                             self.rigs[l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt')][36]>0.4]
        # 筛选"blink_left"相关表情（rig参数0-1 < -0.1）
        imgs_list_blink_left = [l for l in self.imgs_list if l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt') in self.rigs and 
                             (self.rigs[l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt')][0]-self.rigs[l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt')][1]) < -0.1]
        # 筛选"press"相关表情（rig参数0-38 > 0.8）
        imgs_list_press = [l for l in self.imgs_list if l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt') in self.rigs and 
                             (self.rigs[l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt')][0]-self.rigs[l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt')][38]) > 0.8]

        imgs_list_remove_zhoumei = [l for l in self.imgs_list if l not in imgs_list_zhoumei and l not in imgs_list_press]
        # 筛选"mouth_pucker"相关表情（rig参数40>0.8）
        imgs_list_mouth_pucker = [l for l in self.imgs_list if l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt') in self.rigs and 
                             self.rigs[l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt')][40]>0.8]
        # 调整最终图像列表：移除部分zhoumei数据，按比例添加各类表情数据（增强特定特征）
        self.imgs_list = imgs_list_remove_zhoumei+imgs_list_zhoumei[::3] + imgs_list_minzui*3+imgs_list_minzui*10+imgs_list_frownmouth*10 + imgs_list_blink_left*2+imgs_list_press+ imgs_list_mouth_pucker*10
        # 针对L36_230_61角色，额外增强嘴部左右移动的表情数据
        if character == 'L36_230_61':
            # 嘴部右移（rig参数61-60 < -0.3）
            imgs_list_mouthright = [l for l in self.imgs_list if l.replace(self.root_path+'/','').replace('0000.png', 'txt') in self.rigs and 
                                (self.rigs[l.replace(self.root_path+'/','').replace('0000.png', 'txt')][61]-self.rigs[l.replace(self.root_path+'/','').replace('0000.png', 'txt')][60]) < -0.3 ]
            # 嘴部左移（rig参数61-60 > 0.3）
            imgs_list_mouthleft = [l for l in self.imgs_list if l.replace(self.root_path+'/','').replace('0000.png', 'txt') in self.rigs and 
                                (self.rigs[l.replace(self.root_path+'/','').replace('0000.png', 'txt')][61]-self.rigs[l.replace(self.root_path+'/','').replace('0000.png', 'txt')][60]) > 0.3 ]
            # 再次添加嘴部收缩表情数据
            imgs_list_mouth_pucker = [l for l in self.imgs_list if l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt') in self.rigs and 
                             self.rigs[l.replace(self.root_path+'/','').replace(f'0000{img_postfix}', 'txt')][40]>0.8]                    
            # 增强嘴部左右移动数据（各复制5次）
            self.imgs_list = self.imgs_list+ imgs_list_mouthright*5+imgs_list_mouthleft*5


# [l for l in self.imgs_list if l.replace(self.root_path+'/','').replace('0000.png', 'txt') in self.rigs and 
#                              (self.rigs[l.replace(self.root_path+'/','').replace('0000.png', 'txt')][61]-self.rigs[l.replace(self.root_path+'/','').replace('0000.png', 'txt')][60]) > 0.3 ]

        # self.bs_detector = FaceDetectorMediapipe()
        # self.bs = self._get_bs_cache(character, data_split)
        self.bs = {} # 初始化人脸关键点缓存（bs可能指blendshape或facial landmarks）
        # self.imgs_list = self.imgs_list[:500]
        # self.imgs_list = [e for e in self.imgs_list if 'L36face233_ZXS_PM_233_nielian_c05_02_55201_1' in e]
        # 打印当前分割的数据集大小
        print(f'{data_split} data: {len(self.imgs_list)}')
    
    # 缓存人脸关键点数据（避免重复计算）
    def _get_bs_cache(self, character, data_split):
        # 缓存文件路径（按角色和数据分割命名）
        bs_cache_pkl = f'/project/qiuf/DJ01/{character}/bs_20231228_{data_split}.pkl'
        bs_cache = {} # 存储关键点缓存的字典

        # 若缓存文件存在，直接加载
        if os.path.exists(bs_cache_pkl):
            with open(bs_cache_pkl, 'rb') as f:
                bs_cache = pickle.load(f)

        # 筛选未缓存的图像路径
        imgs_list_rest = [im for im in self.imgs_list if im not in bs_cache]
        if not imgs_list_rest: # 若所有图像都已缓存，直接返回
            return bs_cache
        
        # 对未缓存的图像计算关键点并缓存
        for img_path in tqdm(imgs_list_rest):  # tqdm显示进度条
            # 从图像中检测人脸关键点（bs_detector是人脸检测器）
            bs, _, has_face = self.bs_detector.detect_from_PIL(Image.open(img_path).convert('RGB'))
            if has_face:  # 若检测到人脸，存储关键点
                bs_cache[img_path] = np.array(bs)
            else:  # 未检测到人脸，存储全0数组
                bs_cache[img_path] = np.zeros(52)    
        # 保存缓存到文件
        with open(bs_cache_pkl, 'wb') as f:
            pickle.dump(bs_cache, f)
        return bs_cache

    # 建立真实数据到渲染数据的映射（用于数据对齐）
    def _get_render_data_from_real_data(self, imgs_list, character):
        real2render = {}  # 存储映射关系的字典
        # 处理L36_233/230/230_61角色
        if character in ['L36_233', 'L36_230', 'L36_230_61']:
            character = character.replace('_233', '')  # 统一角色名称格式
            save_pkl = f'/project/qiuf/DJ01/{character}/real2render_20231229.pkl'  # 映射缓存路径

            # 若缓存存在，直接加载
            if os.path.exists(save_pkl):
                with open(save_pkl, 'rb') as f:
                    real2render = pickle.load(f)
                return real2render
            # 遍历图像列表，构建映射
            for img in imgs_list:
                # 确定渲染数据根路径（根据真实数据路径）
                if self.faceware_root in img:
                    render_root = self.root_path
                elif self.faceware_root_230 in img:
                    render_root = f'/project/qiuf/DJ01/{character}/images_retarget_from_230'
                else:
                    continue # 跳过不匹配的路径
                # if 'linjie_expr_test' in img:
                #     print(img)
                # 解析图像的文件夹和文件名，构建对应的渲染图像路径
                vid_folder, imgname = img.split('/')[-2], img.split('/')[-1]
                img_index = int(imgname.split('.')[0].split('_')[0]) # 提取图像索引
                imgname_render = f'{img_index:06d}.0000.jpg' # 渲染图像文件名格式

                # 构建渲染图像路径（替换faceware为images）
                imgpath = os.path.join(os.path.dirname(img).replace('faceware/', 'images/'), imgname_render)
                _imgpath = imgpath.replace('.jpg', '.png') # 尝试png格式
                # 若路径存在，记录映射
                if os.path.exists(imgpath):
                    real2render[img] = imgpath
                elif os.path.exists(_imgpath):
                    real2render[img] = _imgpath
            # 保存映射到缓存文件
            with open(save_pkl, 'wb') as f:
                pickle.dump(real2render, f)
        # 处理L36_234角色（逻辑类似，缓存路径不同）
        elif character=='L36_234':
            save_pkl = f'/project/qiuf/DJ01/{character}/real2render_20231122.pkl'
            if os.path.exists(save_pkl):
                with open(save_pkl, 'rb') as f:
                    real2render = pickle.load(f)
                return real2render
            for img in imgs_list:
                if self.faceware_root in img:
                    render_root = self.root_path
                    vid_folder, imgname = img.split('/')[-2], img.split('/')[-1]
                    img_index = imgname.split('.')[0]
                    imgname_render = f'{img_index}.0000.png'
                    imgpath = os.path.join(render_root,vid_folder, imgname_render)
                    if os.path.exists(imgpath):
                        real2render[img] = imgpath
            with open(save_pkl, 'wb') as f:
                pickle.dump(real2render, f)
        return real2render
    
    # 加载230角色的faceware数据路径（返回文件夹路径字典）
    def load_faceware_230(self):
        folders = ['/project/qiuf/DJ01/L36_230/faceware/230_first', 
                   '/project/qiuf/DJ01/L36_230/faceware/230_second']
        faceware_230 = {}
        for fold in folders:
            videos = os.listdir(fold)  # 获取文件夹下的视频文件夹
            for v in videos:
                faceware_230[v] = os.path.join(fold, v)  # 存储{视频名:路径}
        return faceware_230
    
    # 读取MEAD数据集数据（按训练/测试分割）
    def read_MEAD_DATA(self, data_split, list_path_selected=''):
        root = '/project/qiuf/MEAD/images'  # MEAD数据集图像根路径
        folders = os.listdir(root)[::5]  # 每5个文件夹取1个（减少数据量）
        # 按训练/测试分割文件夹
        if data_split in ['train' or 'training']:
            folders = folders[:int(len(folders)*0.9)]  # 前90%为训练
        else:
            folders = folders[int(len(folders)*0.9):]  # 后10%为测试
        img_list=[]
        # 收集图像路径（每10张取1张）
        for folder in folders:
            imgnames = os.listdir(os.path.join(root, folder))
            img_list += [os.path.join(root, folder, im) for im in imgnames][::10]
        # 效果不错的视频，做强制对齐
        # 处理选中的视频（用于强制对齐）
        self.MEAD_selected={}
        if list_path_selected:
            with open(list_path_selected, 'r') as f:
                MEAD_selected = f.readlines()
            for line in MEAD_selected:
                self.MEAD_selected[line.split('_2022')[0]] = line.strip()
            # self.MEAD_selected = [ms.strip() for ms in MEAD_selected]
        return img_list

    # 读取ABAW数据集数据（情感/动作单元标注数据）
    def read_abaw_data(self, data_split):
        root = '/data/data/ABAW/crop_face_jpg'  # ABAW数据集裁剪后的人脸路径
        path_pkl = f'/data/Workspace/Rig2Face/data/abaw_images_{data_split}_large.pkl'  # 缓存路径
        # 若缓存存在，直接加载
        if os.path.exists(path_pkl):
            with open(path_pkl, 'rb') as f:
                img_list = pickle.load(f)
            return img_list[:40000]  # 限制最大数量

        # 按数据分割选择标注文件
        if data_split == 'train':
            data_file = '/project/ABAW2_dataset/gzh/aff_in_the_wild/data_init/Third ABAW Annotations/AU_Set/ABAW3_new_AU_training.csv'
        else:
            data_file = '/project/ABAW2_dataset/gzh/aff_in_the_wild/data_init/Third ABAW Annotations/AU_Set/ABAW3_new_AU_validation.csv'
        # 读取csv文件获取图像路径
        with open(data_file, newline='') as f:
            data_list = list(csv.reader(f, delimiter=' '))
        # 构建图像路径列表（过滤不存在的路径）
        img_list = [os.path.join(root, d[0].split(',')[1]) for d in data_list[1:]]
        img_list = [ad for ad in img_list if os.path.exists(ad)][:50000] # 限制最大数量
        # 保存到缓存
        with open(path_pkl, 'wb') as f:
            pickle.dump(img_list, f)
        return img_list

    # 重载Dataset的__getitem__方法，返回索引对应的样本数据
    def __getitem__(self, index):
        img_path = self.imgs_list[index] # 获取当前索引的图像路径

        # 读取图像（处理可能的读取错误）
        try:
            img = Image.open(img_path).convert('RGB')  # 打开并转为RGB格式
        except:
            print(f'reading img error:{img_path}')  # 打印错误路径
            img_path = self.imgs_list[20]  #  fallback到第20张图像
            img = Image.open(img_path).convert('RGB')

        # 标记是否为渲染数据和是否需要像素级损失
        # todo:
        if self.root_path in img_path or 'render' in img_path :
            is_render = 1  # 是渲染数据
            do_pixel = 1   # 需要计算像素损失
        else:
            is_render = 0  # 是真实数据
            do_pixel = 0   # 不需要像素损失
        
        # 获取人脸关键点数据（从缓存或用0填充）
        if img_path in self.bs:
            bs = self.bs[img_path]
        else:
            # print('bs not exist')
            bs = np.zeros(52)    # 无缓存时用0填充
        # 标记角色ID（区分不同来源的数据）
        if 'Rig2Face/data' in img_path:
            role_id = 1  # Rig2Face数据
        elif 'aligned' in img_path:
            role_id = 0  # 对齐后的真实数据
        else:
            role_id = 2   # render data  # 渲染数据
        
        
        # 若存在真实数据到渲染数据的映射，加载目标渲染图像（用于对齐训练）
        # l36动捕数据强制和动画数据对齐。        
        if img_path in self.real2render:
            do_pixel = 1 # 1才会做像素loss # 强制开启像素损失
            target_path = self.real2render[img_path] # 目标渲染图像路径
            try:
                target = Image.open(target_path).convert('RGB') # 读取目标图像
            except:
                target = img # 读取失败时用原图像代替
        else:
            target = img # 无映射时目标为原图像
            
        # 应用图像预处理变换
        if self.transform:
            img = self.transform(img)
            target = self.transform(target)
        # 构建返回的样本字典
        data = {'img':img, 'target':target, 'is_render':is_render, 'ldmk':np.array([0]*136), 'role_id':role_id, 'do_pixel':do_pixel, 'bs':bs}
        # 若需要返回rig数据，添加到样本中
        if self.return_rigs:
            # 解析rig数据文件名（与图像路径对应）
            _img_path_split = img_path.split('/')[6:]
            imgname = _img_path_split[-1]
            try:
                vindex = int(imgname.split('.')[0])
                rigname = f'{vindex:06d}.txt'  # 格式化rig文件名
            except:
                rigname = imgname.replace('.jpg', '.txt')  # 替换后缀获取rig文件名
            rigname = '/'.join(_img_path_split[:-1]+[rigname])  # 构建rig文件路径
            
            # foldname, imgname = img_path.split('/')[-2:]
            # try:
            #     vindex = int(imgname.split('.')[0])
            #     rigname = f'{vindex:06d}.txt'
                
            # except:
            #     rigname = imgname.replace('.jpg', '.txt')
            # rigname = f"{foldname}/{rigname}"
            # 从缓存中获取rig数据
            if rigname in self.rigs:
                rigs = np.array(self.rigs[rigname])
                has_rig = 1  # 标记存在rig数据
                # print(rigname)
            else:
                assert do_pixel == 0, f'do pixel loss but no rigs:{rigname}'  # 断言：像素损失开启时必须有rig数据
                rigs = np.zeros(self.n_rigs)  # 无rig数据时用0填充
                has_rig = 0  # 标记不存在rig数据
            data['rigs'] = rigs  # 添加rig数据到样本
        data['has_rig'] = has_rig  # 标记是否有rig数据
        return data
    
    # 重载Dataset的__len__方法，返回数据集大小
    def __len__(self):
        return len(self.imgs_list)

# 定义ABAWDataset2_multichar类，继承自PyTorch的Dataset类，用于构建多角色的人脸数据集
class ABAWDataset2_multichar(Dataset):
    # 初始化方法，用于加载多个角色的数据集并进行预处理
    # 参数说明：
    # - characters: 字典，包含多个角色的配置（如数据路径、rig参数维度等）
    # - data_split: 数据集分割（如'train'、'test'）
    # - CHARACTER_NAME: 角色名称列表，用于映射角色ID
    # - transform: 图像预处理变换
    # - random_flip: 是否随机翻转（未实际使用）
    # - do_norm: 是否归一化（未实际使用）
    # - use_ldmk: 是否使用人脸关键点（未实际使用）
    # - return_rigs: 是否返回面部驱动参数（rig数据）
    def __init__(self, characters, data_split, CHARACTER_NAME, transform=None, random_flip=False, do_norm=True, use_ldmk=False, return_rigs=False):
        self.return_rigs = return_rigs  # 标记是否返回rig数据
        self.imgs_list = []  # 存储所有角色的图像路径及对应角色ID（格式：[[图像路径, 角色ID], ...]）
        self.real2render = {}  # 字典，存储每个角色的"真实图像-渲染图像"映射关系（键：角色名，值：映射字典）
        self.rigs = {}  # 字典，存储每个角色的rig参数（键：角色名，值：rig数据字典）
        self.render_path = []  # 存储所有角色的渲染图像路径，用于判断图像是否为渲染数据
        self.n_rigs = 0  # 记录所有角色中最大的rig参数维度（用于统一输出维度）
        self.CHARACTER_NAME = CHARACTER_NAME  # 角色名称列表，用于将角色ID映射为角色名
         # 遍历每个角色的配置，加载对应的数据
        for character, character_config in characters.items():    
            root_path, n_rigs = character_config['data_path'], character_config['n_rig']  # 角色的数据路径和rig维度
            self.render_path.append(root_path + '/')  # 记录该角色的渲染路径
            self.character = character  # 当前处理的角色名
            self.n_rigs = max(self.n_rigs, n_rigs)  # 更新最大rig维度（确保所有角色的rig参数输出维度一致）
            # 定义各类数据的路径
            expr_fuxi_root = '/project/qiuf/Expr_fuxi/images_ttg_2024'  # 伏羲表情数据路径
            self.faceware_root = os.path.join(root_path.replace('/images', '/faceware'))  # 该角色的faceware数据路径（动捕数据）
            self.faceware_root_230 = '/project/qiuf/DJ01/L36_230/faceware'  # 230角色的faceware数据路径
            ttg_dy_root = '/project/qiuf/Expr_fuxi/images_ttg'  # ttg动态数据路径
            fuxi_audio_data ='/project/qiuf/L36_drivendata/anrudong_2023_08_08_29_23_10_03_05_speaking_30fps/crop_face_processed'# 音频驱动的伏羲人脸数据路径
            # 需要排除的无效文件夹（过滤低质量或无关数据）
            exclude_folder = [ 'L36Randsample', 'bichiyin_zuhe', 'L36face234_ZHY_PM_c101_230097_xia_51566_1',
                            'L36face234_ZHY_PM_c102_234011_banxiangzi_51560_1'] # 'qiufeng011'
            # 若角色是L36系列（233/234/230等），加载对应的真实数据路径
            if character.lower() in ['l36_233', 'l36_234','l36_230', 'l36_230_61']:
                # 基础真实数据路径列表（包含多个来源的人脸数据）
                real_data = [
                '/data/Workspace/Rig2Face/data/yinxiaonv3',
                '/data/Workspace/Rig2Face/data/jiayang01',
                '/data/Workspace/Rig2Face/data/dongbu_kuazhang',
                '/data/Workspace/Rig2Face/data/singing',
                '/data/Workspace/Rig2Face/data/blackbro',
                '/data/Workspace/Rig2Face/data/Donbu_yinsu_31115_3_zhy',
                '/data/Workspace/Rig2Face/data/wjj01',
                '/data/Workspace/Rig2Face/data/qiufeng02',
                '/data/Workspace/Rig2Face/data/linjie_expr_test',
                '/project/qiuf/Expr_fuxi/images_old_50fps/C0016',
                '/project/qiuf/L36_drivendata/L36_233_20231106_43th/crop_face/L36face233_CYY_PM_bxy02_52252_1',
                '/project/qiuf/L36_drivendata/L36_233_20231106_43th/crop_face/L36face233_CYY_PM_bj_wjl_52251_1',
                '/project/qiuf/L36_drivendata/L36_233_20231106_43th/crop_face/L36face233_CYY_PM_bj_mm_52247_1',
                '/project/qiuf/L36_drivendata/L36_233_20231106_43th/crop_face/L36face233_CYY_PM_bj_lss_52250_1',
                '/project/qiuf/DJ01/L36/faceware/qiufeng011',
                '/project/qiuf/DJ01/L36/faceware/linjie_expr_test3',
                '/project/qiuf/DJ01/L36/faceware/linjie_expr_test3',
                '/project/qiuf/DJ01/L36/faceware/linjie_expr_test3',
                '/project/qiuf/DJ01/L36/faceware/wjj01',
                '/project/qiuf/DJ01/L36/faceware/wjj01',
                '/project/qiuf/DJ01/L36/faceware/qiufeng02',
                '/project/qiuf/DJ01/L36/faceware/qiufeng02',
                '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c05_02_55201_1',
                '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c05_02_55201_1',
                '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c05_02_55201_1',
                '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c08_55212_1',
                '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c08_55212_1',
                '/project/qiuf/DJ01/L36/faceware/L36face233_ZXS_PM_233_nielian_c08_55212_1',
                ]
                # 若角色是L36_230，额外添加其专属的真实数据路径
                if character.lower() == 'l36_230':
                    real_data += ['/project/qiuf/DJ01/L36_230/faceware/230_first',
                                '/project/qiuf/DJ01/L36_230/faceware/230_second'
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_huazhang_60069_2',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_huazhang_60069_2',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#233_fanghedeng02_59838_1',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#233_fanghedeng02_59838_1',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_wangtian_60064_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_qujia_nanwanjia_60071_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_qujia_nanwanjia_60071_4',
                                '/project/qiuf/DJ01/L36_230/faceware/L36_230#230_233_qujia_nanwanjia_60071_4',
                                ]
                # 若角色是L36_233或L36_234，将其faceware路径添加到真实数据    
                elif character.lower() in ['l36_233', 'l36_234']:
                    real_data += [self.faceware_root]
                
                # 加载该角色的faceware文件夹下的所有数据，补充到真实数据
                # 233faceware
                faceware_folder = os.listdir(self.faceware_root) # 获取faceware文件夹列表
                # faceware_folder = [y for x in os.walk(self.faceware_root) for y in glob.glob(os.path.join(x[0], '*.jpg'))]
                real_data += [os.path.join(self.faceware_root, e) for e in faceware_folder] # 构建完整路径并添加
                # 230数据
                # self.faceware_folder_230 = self.load_faceware_230()
                # real_data += list(self.faceware_folder_230.values())

                # JHM大脸男的数据加一些
                # JHM_root = '/project/qiuf/DJ01/L36_230/faceware/230_second'
                # JHM_folder = [os.path.join(JHM_root, e) for e in os.listdir(JHM_root) if '_JHM_' in e and not e.endswith('.mp4')][::3]
                # real_data += JHM_folder

                # 加载伏羲音频驱动数据（每3个取1个，控制数据量）
                # 伏羲audio data
                # read_data += fuxi_audio_data
                expr_audio_data = os.listdir(fuxi_audio_data)[::3]
                real_data += [os.path.join(fuxi_audio_data, fo) for fo in expr_audio_data]  # 构建路径并添加

                
            # 初始化当前角色的基础参数
            self.root_path = root_path  # 当前角色的数据根路径
            self.transform = transform  # 图像预处理变换
            self.use_ldmk = use_ldmk  # 是否使用人脸关键点（未实际使用）
            imgs_list = []  # 存储当前角色的图像路径列表
            render_folders, render_folders_230 = [], []  # 渲染文件夹列表（用于筛选渲染数据）
            # 构建当前角色的图像列表文件路径（包含动作信息）
            n_rig_fold = len(os.listdir(root_path.replace('/images', '/rigs')))# rig文件夹数量
            image_list_action = os.path.join(root_path.replace('/images', ''), f'images_{data_split}_list_actions{n_rig_fold}.txt') # 动作图像列表
            # 若图像列表文件存在，读取并添加到当前角色的图像列表
            if os.path.exists(image_list_action):
                with open(image_list_action, 'r') as f:
                    lines = f.readlines() # 读取所有行
                # lines = list(set(lines))
                # 提取渲染文件夹（去重）
                render_folders += list(set([l.split('/')[0].split('_202')[0] for l in lines]))
                # 添加图像路径（从列表文件中读取）
                imgs_list += [os.path.join(root_path, l.strip()) for l in lines]
            
            # 处理230角色的图像列表文件（若存在）
            image_list_action_230 = os.path.join(root_path.replace('/images', ''), f'images_{data_split}_list_actions_230.txt')
            if os.path.exists(image_list_action_230):
                with open(image_list_action_230, 'r') as f:
                    lines_230 = f.readlines() # 读取所有行
                render_folders_230 += list(set([l.split('/')[0] for l in lines_230]))
                # 添加重定向的230角色图像路径
                imgs_list += [os.path.join(root_path.replace('/images', '/images_retarget_from_230'), l.strip()) for l in lines_230]
            # 若未通过列表文件获取图像，直接遍历根路径下的文件夹收集图像
            if not imgs_list:
                # 如果没有另外指定list action
                folders = os.listdir(root_path) # 获取根路径下的文件夹
                for folder in folders:
                    if os.path.isdir(os.path.join(root_path, folder)): # 仅处理文件夹
                        images = os.listdir(os.path.join(root_path, folder)) # 文件夹下的图像
                        imgs_list += [os.path.join(root_path, folder, imgname) for imgname in images] # 构建完整路径
            
            # 加载真实数据（非渲染数据）并添加到当前角色的图像列表
            # load real data
            for folder in real_data:
                # 递归遍历文件夹，收集所有jpg和png图像
                images = [y for x in os.walk(folder) for y in glob.glob(os.path.join(x[0], '*.jpg'))]
                images += [y for x in os.walk(folder) for y in glob.glob(os.path.join(x[0], '*.png'))]
                 # 对特定路径的图像按比例采样（减少数据量）
                if self.faceware_root_230 in folder:
                    images = images[::10] # 每10张取1张
                elif expr_fuxi_root in folder:
                    images = images[::10]  # 每10张取1张
                # 按训练/测试分割数据
                if data_split in ['train', 'training']:
                    imgs_list += images[:int(0.9*len(images))] # 前90%作为训练集
                elif data_split in ['test']:
                    imgs_list += images[int(0.9*len(images)):] # 后10%作为测试集
            
            # 建立当前角色的"真实图像-渲染图像"映射关系，并存储到字典
            # 建立real data和render data之间的联系
            self.real2render[character]=self._get_render_data_from_real_data(imgs_list, character)
            # self.real2render = {}
            
            # 读取ABAW公开数据集的数据，按30%比例混合到当前角色的图像列表
            abaw_data = self.read_abaw_data(data_split)  # 读取ABAW数据
            random.shuffle(abaw_data)  # 随机打乱
            abaw_data = abaw_data[:int(len(imgs_list)*0.3)]  # 取30%与现有数据混合
            imgs_list += abaw_data[::]

            # 过滤掉包含无效文件夹的图像路径
            imgs_list = [e for e in imgs_list if e.split('/')[-2] not in exclude_folder]
            
            # 区分对称和非对称数据（可能与表情对称性相关）
            # 对称数据：不含'linjie'、'qiufeng'、'wjj'的图像
            symm = [e for e in imgs_list if 'linjie' not in e and 'qiufeng' not in e and 'wjj' not in e ]
            # 非对称数据：包含上述关键词的图像
            dissymm = [e for e in imgs_list if 'linjie' in e or 'qiufeng' in e or 'wjj' in e ]
            # 调整比例：对称数据每2张取1张，保留所有非对称数据
            imgs_list = symm[::2] + dissymm 
            # imgs_list = [e for e in imgs_list if 'linjie' in e and 'faceware' in e]
            # imgs_list = [e for e in imgs_list if 'linjie' in e]
            
            # 加载当前角色的rig参数到缓存，并存储到字典
            # 去除部分zhoumei数据
            self.rigs[character]=load_rigs_to_cache(self.root_path.replace('/images', '/rigs'), n_rig=n_rigs)

            # 根据rig参数筛选特定表情的图像，进行数据增强（解决样本不平衡问题）
            # 筛选"zhoumei"表情（rig参数14>0.3）
            # imgs_list_zhoumei = [l for l in imgs_list if f"{l.split('/')[-2]}/{l.split('/')[-1].replace('0000.'+l.split('/')[-1].split('.')[-1], 'txt').replace(l.split('/')[-1].split('.')[-1], 'txt')}" in self.rigs and self.rigs[f"{l.split('/')[-2]}/{l.split('/')[-1].replace('0000.'+l.split('/')[-1].split('.')[-1], 'txt').replace(l.split('/')[-1].split('.')[-1], 'txt')}"][14]>0.3]
            imgs_list_zhoumei = [l for l in imgs_list if l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt') in self.rigs[character] and 
                                self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][14]>0.3]
            # 筛选"minzui（抿嘴）"表情（rig参数38>0.6）
            imgs_list_minzui = [l for l in imgs_list if l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt') in self.rigs[character] and 
                                self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][38]>0.6]
            # 筛选"frownmouth（皱眉）"表情（rig参数36>0.3）
            imgs_list_frownmouth = [l for l in imgs_list if l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt') in self.rigs[character] and 
                                self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][36]>0.3]
            # 筛选"blink_left（左眨眼）"表情（rig参数0-1 < -0.1）
            imgs_list_blink_left = [l for l in imgs_list if l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt') in self.rigs[character] and 
                                (self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][0]-self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][1]) < -0.1]
            # 筛选"duzui（嘟嘴）"表情（rig参数40和41均>0.6）
            imgs_list_duzui = [l for l in imgs_list if l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt') in self.rigs[character] and 
                                self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][40]>0.6 and 
                                self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][41]>0.6 ]
            # 筛选"guzui（鼓嘴）"表情（根据rig维度选择参数64或48>0.6）
            imgs_list_guzui =  [l for l in imgs_list if l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt') in self.rigs[character] and 
                                self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][64 if n_rigs==67 else 48]>0.6]
            # 筛选"press（抿嘴压力）"表情（rig参数0-38 > 0.8）
            imgs_list_press = [l for l in imgs_list if l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt') in self.rigs[character] and 
                                (self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][0]-self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][38]) > 0.8]
            # 筛选"L36_230_61角色的微笑表情"（rig参数54>0.6且19<0.1）
            imgs_list_smile =  [l for l in imgs_list if character=='L36_230_61' and l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt') in self.rigs[character] and 
                                self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][54]>0.6 and self.rigs[character][l.replace(self.root_path+'/','').replace('0000.jpg', 'txt').replace('0000.png', 'txt')][19]<0.1]
            # 调整最终图像列表：移除部分重复样本，按比例增强特定表情样本
            imgs_list_remove_zhoumei = [l for l in imgs_list if l not in imgs_list_zhoumei and l not in imgs_list_press]
            imgs_list = imgs_list_remove_zhoumei+imgs_list_zhoumei[::5] + imgs_list_minzui*5 + imgs_list_frownmouth*10 + imgs_list_blink_left*2 + \
                imgs_list_press[::3] + imgs_list_duzui*10 + imgs_list_guzui * 5 +  imgs_list_smile *5
            # 将当前角色的图像路径与角色ID（CHARACTER_NAME中的索引）关联，添加到全局图像列表
            self.imgs_list+=[[e, CHARACTER_NAME.index(character)] for e in imgs_list]
            # self.bs_detector = FaceDetectorMediapipe()
            # self.bs = self._get_bs_cache(character, data_split)
        self.bs = {} # 初始化人脸关键点缓存（未实际使用）
            # self.imgs_list = self.imgs_list[:500]
            # self.imgs_list = [e for e in self.imgs_list if 'L36face233_ZXS_PM_233_nielian_c05_02_55201_1' in e]
        # self.imgs_list = self.imgs_list[:80]
        # 打印当前分割的数据集总大小
        print(f'{data_split} data: {len(self.imgs_list)}')
    
    # 缓存人脸关键点数据（避免重复计算），与单角色类逻辑一致
    def _get_bs_cache(self, character, data_split):
        bs_cache_pkl = f'/project/qiuf/DJ01/{character}/bs_20231228_{data_split}.pkl' # 缓存文件路径
        bs_cache = {} # 存储关键点的字典

        # 若缓存存在，直接加载
        if os.path.exists(bs_cache_pkl):
            with open(bs_cache_pkl, 'rb') as f:
                bs_cache = pickle.load(f)

        # 筛选未缓存的图像
        imgs_list_rest = [im for im in self.imgs_list if im not in bs_cache]
        if not imgs_list_rest:
            return bs_cache
        
        # 计算并缓存关键点
        for img_path in tqdm(imgs_list_rest):
            bs, _, has_face = self.bs_detector.detect_from_PIL(Image.open(img_path).convert('RGB'))
            if has_face:
                bs_cache[img_path] = np.array(bs)
            else:
                bs_cache[img_path] = np.zeros(52)    
        # 保存缓存
        with open(bs_cache_pkl, 'wb') as f:
            pickle.dump(bs_cache, f)
        return bs_cache

    # 为单个角色建立"真实图像-渲染图像"的映射关系（用于数据对齐）
    def _get_render_data_from_real_data(self, imgs_list, character):
        real2render = {} # 存储映射关系的字典
        # 处理L36_233/230/230_61角色
        if character in ['L36_233', 'L36_230', 'L36_230_61']:
            character = character.replace('_233', '') # 统一角色名格式
            save_pkl = f'/project/qiuf/DJ01/{character}/real2render_20231229.pkl' # 映射缓存路径

            # 若缓存存在，直接加载
            if os.path.exists(save_pkl):
                with open(save_pkl, 'rb') as f:
                    real2render = pickle.load(f)
                return real2render
            # 遍历图像列表，构建映射
            for img in imgs_list:
                # 确定渲染数据根路径
                if self.faceware_root in img:
                    render_root = self.root_path
                elif self.faceware_root_230 in img:
                    render_root = f'/project/qiuf/DJ01/{character}/images_retarget_from_230'
                else:
                    continue # 跳过不匹配的路径
                # if 'linjie_expr_test' in img:
                #     print(img)
                # 解析图像索引，构建对应的渲染图像路径
                vid_folder, imgname = img.split('/')[-2], img.split('/')[-1]
                img_index = int(imgname.split('.')[0].split('_')[0]) # 提取图像索引
                imgname_render = f'{img_index:06d}.0000.jpg' # 渲染图像文件名

                # 构建路径（将faceware替换为images）
                imgpath = os.path.join(os.path.dirname(img).replace('faceware/', 'images/'), imgname_render)
                if os.path.exists(imgpath):
                    real2render[img] = imgpath  # 记录映射关系
            # 保存映射到缓存
            with open(save_pkl, 'wb') as f:
                pickle.dump(real2render, f)
        # 处理L36_234角色（逻辑类似，缓存路径不同）
        elif character=='L36_234':
            save_pkl = f'/project/qiuf/DJ01/{character}/real2render_20231122.pkl'
            if os.path.exists(save_pkl):
                with open(save_pkl, 'rb') as f:
                    real2render = pickle.load(f)
                return real2render
            for img in imgs_list:
                if self.faceware_root in img:
                    render_root = self.root_path
                    vid_folder, imgname = img.split('/')[-2], img.split('/')[-1]
                    img_index = imgname.split('.')[0]
                    imgname_render = f'{img_index}.0000.png'
                    imgpath = os.path.join(render_root,vid_folder, imgname_render)
                    if os.path.exists(imgpath):
                        real2render[img] = imgpath
            with open(save_pkl, 'wb') as f:
                pickle.dump(real2render, f)
        return real2render
    
    # 加载L36_230角色的faceware动捕数据路径（返回{视频名:路径}字典）
    def load_faceware_230(self):
        folders = ['/project/qiuf/DJ01/L36_230/faceware/230_first', 
                   '/project/qiuf/DJ01/L36_230/faceware/230_second']
        faceware_230 = {}
        for fold in folders:
            videos = os.listdir(fold) # 获取视频文件夹
            for v in videos:
                faceware_230[v] = os.path.join(fold, v)  # 存储路径
        return faceware_230
    
    # 读取MEAD公开数据集的图像路径（按训练/测试分割）
    def read_MEAD_DATA(self, data_split, list_path_selected=''):
        root = '/project/qiuf/MEAD/images' # MEAD图像根路径
        folders = os.listdir(root)[::5] # 每5个文件夹取1个（控制数据量）
         # 按训练/测试分割
        if data_split in ['train' or 'training']:
            folders = folders[:int(len(folders)*0.9)] # 前90%为训练
        else:
            folders = folders[int(len(folders)*0.9):] # 后10%为测试
        img_list=[]
        # 收集图像路径（每10张取1张）
        for folder in folders:
            imgnames = os.listdir(os.path.join(root, folder))
            img_list += [os.path.join(root, folder, im) for im in imgnames][::10]
        # 处理选中的视频（用于强制对齐，未实际使用）
        self.MEAD_selected={}
        
        if list_path_selected:
            with open(list_path_selected, 'r') as f:
                MEAD_selected = f.readlines()
            for line in MEAD_selected:
                self.MEAD_selected[line.split('_2022')[0]] = line.strip()
            # self.MEAD_selected = [ms.strip() for ms in MEAD_selected]
        return img_list

    # 读取ABAW公开数据集的图像路径（支持缓存）
    def read_abaw_data(self, data_split):
        root = '/data/data/ABAW/crop_face_jpg' # ABAW裁剪后的人脸路径
        path_pkl = f'/data/Workspace/Rig2Face/data/abaw_images_{data_split}_large.pkl' # 缓存路径
        # 若缓存存在，直接加载
        if os.path.exists(path_pkl):
            with open(path_pkl, 'rb') as f:
                img_list = pickle.load(f)
            return img_list[:40000] # 限制最大数量

        # 选择训练/测试对应的标注文件
        if data_split == 'train':
            data_file = '/project/ABAW2_dataset/gzh/aff_in_the_wild/data_init/Third ABAW Annotations/AU_Set/ABAW3_new_AU_training.csv'
        else:
            data_file = '/project/ABAW2_dataset/gzh/aff_in_the_wild/data_init/Third ABAW Annotations/AU_Set/ABAW3_new_AU_validation.csv'
        # 读取csv文件获取图像路径
        with open(data_file, newline='') as f:
            data_list = list(csv.reader(f, delimiter=' '))
        img_list = [os.path.join(root, d[0].split(',')[1]) for d in data_list[1:]] # 构建路径
        img_list = [ad for ad in img_list if os.path.exists(ad)][:50000] # 过滤无效路径并限制数量
        # 保存到缓存
        with open(path_pkl, 'wb') as f:
            pickle.dump(img_list, f)
        return img_list

    # 重载__getitem__方法，返回指定索引的样本数据（包含多角色信息）
    def __getitem__(self, index):
        img_path, role_id = self.imgs_list[index]  # 获取图像路径和角色ID
        character = self.CHARACTER_NAME[role_id]  # 根据角色ID获取角色名
        # 读取图像（处理可能的读取错误）
        try:
            img = Image.open(img_path).convert('RGB') # 打开并转为RGB
        except:
            print(f'reading img error:{img_path}') # 打印错误路径
            # _imgname = img_path.split('/').split('.')[0]
            # img_path = img_path.replace(_imgname, f'{int(_imgname)-1:06d}')
            img_path, role_id = self.imgs_list[128] #  fallback到第128个样本
            img = Image.open(img_path).convert('RGB')

        # todo:
        # 解析当前图像对应的rig参数文件名
        _img_path_split = img_path.split('/')[6:] # 分割路径，提取关键部分
        imgname = _img_path_split[-1] # 图像文件名
        try:
            vindex = int(imgname.split('.')[0])  # 提取图像索引
            rigname = f'{vindex:06d}.txt' # 格式化rig文件名
        except:
            rigname = imgname.replace('.jpg', '.txt') # 替换后缀获取rig文件名
        rigname = '/'.join(_img_path_split[:-1]+[rigname]) # 构建完整rig路径
            
        # 标记是否为渲染数据和是否计算像素损失
        is_render = 0  # 默认为非渲染数据
        do_pixel = 0    # 默认为不计算像素损失
        # 若图像是渲染数据且存在对应的rig参数，标记为渲染数据并开启像素损失
        if 'render' in img_path and rigname in self.rigs[character]:
            is_render = 1
            do_pixel = 1 
        else:
            # 检查图像路径是否包含任何角色的渲染路径，且存在rig参数
            for render_path in self.render_path:
                if render_path in img_path and rigname in self.rigs[character]:
                    is_render = 1
                    do_pixel = 1    
        
        # 获取人脸关键点（从缓存或用0填充，未实际使用）
        if img_path in self.bs:
            bs = self.bs[img_path]
        else:
            bs = np.zeros(52)    

        # 若存在真实图像到渲染图像的映射，加载目标渲染图像（用于对齐训练）
        # l36动捕数据强制和动画数据对齐。        
        if img_path in self.real2render[character]:
            do_pixel = 1 # 1才会做像素loss # 强制开启像素损失
            target_path = self.real2render[character][img_path]  # 目标渲染图像路径
            target = Image.open(target_path).convert('RGB') # 读取目标图像
            # target = img 

        else:
            target = img # 无映射时目标为原图像
        
        # 应用图像预处理变换
        if self.transform:
            img = self.transform(img)
            target = self.transform(target)
        # 构建样本字典，包含图像、目标图像及辅助信息
        data = {'img':img, 'target':target, 'is_render':is_render, 'ldmk':np.array([0]*136), 'role_id':role_id, 'do_pixel':do_pixel, 'bs':bs}
        # 若需要返回rig参数，添加到样本中
        if self.return_rigs:
            rigs = np.zeros(self.n_rigs) # 初始化rig数组（维度为最大rig维度）
            # for render_path in self.render_path:
            #     img_path = img_path.replace(render_path, '')
            # _img_path_split = img_path.split('/')[6:]
            # imgname = _img_path_split[-1]
            # try:
            #     vindex = int(imgname.split('.')[0])
            #     rigname = f'{vindex:06d}.txt'
            # except:
            #     rigname = imgname.replace('.jpg', '.txt')
            # rigname = '/'.join(_img_path_split[:-1]+[rigname])
             # 若存在对应的rig参数，填充到数组（不足部分补0）
            if rigname in self.rigs[character]:
                
                _rigs = np.array(self.rigs[character][rigname])
                rigs[:len(_rigs)] = _rigs # 按实际长度填充，保证输出维度统一
                has_rig = 1 # 标记存在rig参数
                # print(rigname)
            else:
                # 断言：若开启像素损失，必须存在rig参数
                if do_pixel==1:
                    print(1)
                assert do_pixel == 0, f'do pixel loss but no {character} rigs:{rigname}'
                has_rig = 0 # 标记不存在rig参数
            data['rigs'] = rigs # 添加rig参数
            data['has_rig'] = has_rig # 标记是否存在rig参数
        return data
    
    # 重载__len__方法，返回数据集总样本数
    def __len__(self):
        return len(self.imgs_list)

# 对输入图像进行中心裁剪，再将裁剪后的图像缩放回原始尺寸
def center_crop_restore(img):
    # input: 256*256*3 # （说明预期输入为256x256的三通道图像）
    w, h = 211, 211 # 定义中心裁剪的目标宽度和高度（211x211）

    # 判断图像通道是否在第一维（常见于PyTorch的[C, H, W]格式）
    if img.shape[0] == 3: 
        ori_shape = img.shape # 记录原始图像形状（用于后续恢复尺寸）

        # 计算中心裁剪的起始x坐标（宽度方向中心偏移）
        x = int(img.shape[1] / 2 - w / 2)
        # 计算中心裁剪的起始y坐标（高度方向中心偏移）
        y = int(img.shape[2] / 2 - h / 2)
        # 执行中心裁剪：截取通道维度不变，高度从y到y+h，宽度从x到x+w
        crop_img = img[:, y:y+h,x:x+w]
        # 将裁剪后的图像缩放回原始尺寸（使用numpy的resize函数）
        img = np.resize(crop_img, ori_shape)
        return img # 返回处理后的图像
    else:
        # 图像通道在最后一维的情况（常见于OpenCV的[H, W, C]格式）
        ori_shape = img.shape # 记录原始图像形状

        # 计算中心裁剪的起始x坐标（宽度方向）
        x = int(img.shape[0] / 2 - w / 2)
        # 计算中心裁剪的起始y坐标（高度方向）
        y = int(img.shape[1] / 2 - h / 2)
        # 执行中心裁剪：截取高度从y到y+h，宽度从x到x+w，通道维度不变
        crop_img = img[y:y+h, x:x+w]
        # 将裁剪后的图像缩放回原始尺寸（使用OpenCV的resize函数）
        img = cv2.resize(crop_img, (ori_shape[0], ori_shape[1]))
        return img  # 返回处理后的图像

# 读取图像并进行一系列预处理
def read_image(img_path, mode='rgb', flip=False, size=None, center_crop=True, div=False):
    # 参数说明：
    # - img_path: 图像文件路径
    # - mode: 图像通道模式，'rgb'表示转换为RGB格式，默认'rgb'
    # - flip: 是否水平翻转图像，False表示不翻转
    # - size: 缩放尺寸，若为整数则将图像缩放到(size, size)，None表示不缩放
    # - center_crop: 是否进行中心裁剪并还原尺寸，True表示执行（调用center_crop_restore函数）
    # - div: 是否将像素值归一化到[0,1]，True表示除以255
    # 使用OpenCV读取图像（默认读取为BGR格式）
    img = cv2.imread(img_path)
    try:
        # 若指定模式为'rgb'，将BGR格式转换为RGB格式（因OpenCV默认读为BGR，与其他库可能存在差异）
        if mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        # 若转换失败（如图像读取错误），打印错误信息和对应路径
        print('read image error:', img_path)
    # 若需要水平翻转，使用OpenCV的flip函数（参数1表示水平翻转）
    if flip:
        img = cv2.flip(img, 1)
    # 若指定了缩放尺寸，将图像缩放到(size, size)的正方形
    if size:
        img = cv2.resize(img, (size, size))
    # 若需要中心裁剪，调用之前定义的center_crop_restore函数（裁剪中心区域后还原原始尺寸）
    if center_crop:
        img = center_crop_restore(img)
    # 若需要归一化，将像素值从[0,255]转换为[0,1]（除以255）
    if div:
        img = img / 255.
    # 返回预处理后的图像
    return img

# 读取特定格式的文本文件，并将内容解析为浮点数列表
def read_txt(path):
    # 参数path：文本文件的路径
    
    # 使用with语句打开文件（自动处理文件关闭，避免资源泄露）
    # 'r'表示只读模式
    with open(path, 'r') as f:
        # 读取文件所有行，存储为列表（每行作为一个元素）
        lines = f.readlines()
    
    # 处理读取到的行：
    # 1. 跳过第一行（lines[1:]），通常第一行为表头或注释
    # 2. 对每行进行清洗：
    #    - strip(' \n')：去除行首尾的空格和换行符
    #    - [: -1]：去掉行尾最后一个字符（可能是多余的逗号或分隔符）
    #    - replace(',', ' ')：将逗号替换为空格，统一分隔符
    #    - split(' ')：按空格分割为字符串列表（自动忽略多个连续空格）
    # 3. map(float, ...)：将分割后的字符串列表转换为浮点数列表
    # 4. 最终得到一个二维列表，每个子列表是一行的浮点数数据
    lines = [list(map(float, l.strip(' \n')[:-1].replace(',',' ').split(' '))) for l in lines[1:]]
    
    # 返回解析后的二维浮点数列表
    return lines

# 从指定根目录提取所有JPG图像，并复制到新的目录中（用于整理DJ01相关图像）
def extract_DJ01_images():
    from glob import glob # 导入glob模块，用于匹配文件路径
    # 定义原始图像的根目录（这里使用的是测试结果目录，注释掉的是另一个可能的源目录）
    # root = '/data/data/DJ01/DJ01/images'
    root = '/data/Workspace/Rig2Face/results/kanghui220220402-023345test/'
    # 递归遍历root目录下的所有子文件夹，收集所有.jpg图像的路径
    # os.walk(root)生成root下所有目录的路径、子目录名、文件名
    # glob(os.path.join(x[0], '*.jpg'))匹配每个目录下的所有.jpg文件
    # 最终images是一个包含所有JPG图像绝对路径的列表
    images = [y for x in os.walk(root) for y in glob(os.path.join(x[0], '*.jpg'))]
    # 对图像列表进行采样（控制提取的数量）：每3张取1张（减少数据量）
    # 注释掉的是另一种采样方式：前5000张每3张取1张 + 剩余所有每10张取1张
    # images = images[:5000:3] + images[::10]
    images = images[::3]
    # 定义保存提取图像的目标目录（在原始根目录下创建query_dj文件夹）
    save_root = os.path.join(root, 'query_dj')
    # 创建目标目录，exist_ok=True表示若目录已存在则不报错
    os.makedirs(save_root, exist_ok=True)
    # 遍历所有采样后的图像路径，使用tqdm显示进度条
    for im_source in tqdm(images):
        # 构建目标路径：保持原文件名，保存到save_root目录下
        target = os.path.join(save_root, im_source.split('/')[-1])
        # 将源图像文件复制到目标路径
        shutil.copyfile(im_source, target)
    # 函数无返回值（或可根据需要返回提取的图像数量等信息）
    return

# 将ABAW5数据集的Hubert特征按视频ID拆分并单独保存，便于DeBERT模型批量处理
# # 注：DeBERT是一种基于Transformer的预训练模型，此处用于处理音频或文本特征
def make_debert_batch():
    # 定义输入的特征文件路径（ABAW5挑战4的验证集Hubert特征，以pkl格式存储）
    file_feature = os.path.join('/data/data/ABAW5/challenge4/', 'val_hubert.pkl')
    # 定义保存拆分后特征的根目录（路径与输入文件相同，仅去掉.pkl后缀）
    save_root = file_feature.replace('.pkl','')
    # 创建保存目录，若目录已存在则不报错（exist_ok=True）
    os.makedirs(save_root,exist_ok=True)
    # 读取pkl格式的特征文件（该文件是一个字典，键为视频ID，值为对应视频的Hubert特征）
    with open(file_feature, 'rb') as f:
        features = pickle.load(f)
    # 遍历特征字典中的每个视频ID，使用tqdm显示处理进度（总进度为视频ID的数量）
    for vid_id in tqdm(features.keys(), total=len(features)):
        # 构建当前视频ID对应的特征保存路径（保存到save_root目录下，文件名为视频ID）
        save_path = os.path.join(save_root, vid_id)
        # 将当前视频ID的特征单独保存为pkl文件
        with open(save_path, 'wb') as f:
            pickle.dump(features[vid_id], f)     
            
# 读取L36角色的控制参数文件（通常是包含面部驱动参数的文本文件），并转换为NumPy数组
def read_L36_ctrls(path):
    # 参数path：控制参数文件的路径
    # 使用with语句以只读模式打开文件，确保文件操作完成后自动关闭
    with open(path, 'r') as f:
        # 读取文件所有行，存储为列表（每行作为一个元素）
        lines = f.readlines()
    # 处理每行数据：
    # 1. l.strip(' \n')：去除每行首尾的空格和换行符
    # 2. split(' ')：按空格分割为字符串列表（处理以空格分隔的数值）
    # 3. map(float, ...)：将字符串列表转换为浮点数列表
    # 4. 最终得到一个包含多行浮点数列表的二维列表
    lines = [list(map(float, l.strip(' \n').split(' '))) for l in lines]
    # 将处理后的二维列表转换为NumPy数组，并使用squeeze()去除维度为1的轴（简化数组结构）
    return np.array(lines).squeeze()

# 筛选L36角色中与动作相关的图像数据（侧重233等子角色），通过重采样平衡不同动作的样本数量，最终生成训练图像列表
# 解决动作样本分布不均问题（稀有动作样本少，常见动作样本多），提升模型对各类动作的学习效果
def filter_image_data_with_actions_233():
    # 设置随机种子，保证采样结果可复现
    random.seed(0)
    # 定义图像数据根目录（L36角色的图像文件夹）
    root = r'/project/qiuf/DJ01/L36/images'
    # 定义筛选后训练图像列表的保存路径
    save_path = '/project/qiuf/DJ01/L36/images_train_list_actions.txt'
    # 获取根目录下的所有文件夹名称
    folders = os.listdir(root)
    # 用于存储最终筛选出的图像路径列表
    data_list = []
    # 过滤出真正的文件夹（排除文件，只保留目录）
    folders = [f for f in folders if os.path.isdir(os.path.join(root, f))]
    # 打乱文件夹顺序（随机化数据顺序）
    random.shuffle(folders)
    # 取前90%的文件夹作为训练数据（剩下10%可作为验证/测试集）
    folders = folders[:int(len(folders)*0.9)]
    # 存储有效文件夹（包含符合条件的图像和对应的rig参数）
    avaible_folders = []
    # 注释说明未来可能的优化方向：通过统计不同维度参数的出现频次来优化重采样策略，当前仅关注嘴部动作
    # 加载L36角色的rig参数文件（面部驱动参数，用于判断动作）
    # TODO: 统计不同维度的出现频次来重采样数据。只算嘴
    with open(r'/project/qiuf/DJ01/L36/rigs71.pkl', 'rb') as f:
        rigs_data = pickle.load(f)
    # 对rig参数按文件名中的数字排序（确保时序一致性，如按帧号排序）
    # 排序依据：文件名中"/"分隔后的第二个部分（通常是帧号）转换为整数
    rigs_data = {k: v for k, v in sorted(rigs_data.items(), key=lambda x: int(x[0].split('.')[0].split('/')[1]))}
    # 存储筛选后的rig参数
    rigs = [] 
    # 筛选与233、CYY、mainline相关的rig参数（这些是目标子角色的动作数据）
    for name, rig in rigs_data.items():
        if '233_' in name or 'CYY_' in name or 'mainline_' in name:
        # if '233_chikouxiongceshi_30082_1' in name:
            rigs.append(rig)
    # 计算筛选出的rig参数的平均值（作为动作基准）
    rigs_avg = np.mean(np.array(rigs), axis=0)
    # 提取嘴部相关的rig参数（索引20以后，假设这部分对应嘴部动作），并找到最大值
    max_mouth = np.max(rigs_avg[20:])
    # 对嘴部rig参数的平均频率进行归一化（除以最大值）
    rigs_freq_normed = rigs_avg / max_mouth
    # 计算采样频率（稀有动作的采样频率更高，1/归一化频率）
    rigs_sample_freq = 1 / rigs_freq_normed
    # 统计无动作样本的数量
    count_no_action = 0
    # 控制采样频率的系数（让常见动作采样频率≈1，无动作≈0.1）
    coff_count = 5000 # 控制不同表情的采样频率，尽量让常见表情的采样频率=1. 无表情的采样频率=0.1
    # 提取所有rig参数中的嘴部部分（索引20以后），并每10个取1个（减少计算量）
    rigs_mouth = np.array(rigs)[:, 20:][::10]
    # 遍历每个文件夹（带进度索引e）
    for e, folder in enumerate(folders):
        # 筛选包含233、CYY或mainline的文件夹（只处理目标子角色的数据）
        if '233_' not in folder and 'CYY_' not in folder and 'mainline_' not in folder:
            continue
        # 打印当前处理进度（第e个/总文件夹数）和文件夹名称
        print(f'[{e}/{len(folders)}]:', folder)
        # 获取当前文件夹下的所有图像文件名
        imgnames = os.listdir(os.path.join(root, folder))
        # 尝试按图像文件名中的数字排序（确保按帧顺序处理）
        try:
           imgnames.sort(key=lambda x: int(x.split('.')[0]))
        except:
            # 排序失败则跳过该文件夹
            continue
        # 构建当前文件夹对应的rig参数文件夹路径（假设rig与images目录结构平行，仅文件夹名不同）
        rigs_folder = os.path.join(root.replace('images', 'rigs'), folder)
        # 检查rig参数文件夹是否存在，不存在则跳过
        if not os.path.exists(rigs_folder):
            print(f'{rigs_folder} do not has corresponding rigs folder!')
            continue
        # 检查图像数量与rig参数文件数量是否一致（确保每一帧都有对应的驱动参数）
        if len(imgnames) != len(os.listdir(rigs_folder)):
            print(f'{folder} images and rigs has different frames!')
            continue
        # 将有效文件夹加入列表
        avaible_folders.append(folder)
        # 存储当前文件夹的采样频率（未实际使用，仅作为记录）
        sample_freqs = []
        # 遍历当前文件夹下的每个图像（带进度条）
        for i, imgname in tqdm(enumerate(imgnames)):
            # 构建当前图像对应的rig参数文件名（从rigs_data字典中查找）
            rigname = os.path.join(folder, imgname.replace('.0000.jpg', '.txt'))
            # 尝试从rigs_data中获取当前图像的rig参数
            if rigname in rigs_data:
                rig = rigs_data[rigname]
            else:
                # 若未找到，直接从rig文件夹中读取rig参数文件（调用之前定义的read_L36_ctrls函数）
                rigname = os.path.join(rigs_folder, imgname.replace('.0000.jpg', '.txt'))
                rig = read_L36_ctrls(os.path.join(root, rigname))
            
            # 核心采样逻辑：基于当前rig与已有rig的相似性计算采样频率
            # 1. 计算当前嘴部rig参数与所有已有嘴部rig参数的欧氏距离（衡量动作相似度）
            # 计算当前rig和所有rigs的距离，小于阈值的认为是类似的动作。这个动作出现的次数越少，采样频率越高。 
            dist = np.linalg.norm(rigs_mouth - rig[20:], axis=1)
            # 2. 统计相似动作的数量（距离小于0.5的视为相似）
            cnt_similar = np.sum(dist<0.5) 
            # 3. 计算采样频率：相似动作越少（cnt_similar越小），采样频率越高（log放大差异）
            sample_freq = np.log(coff_count/cnt_similar)
            # sample_freqs.append(sample_freq)
            # 根据采样频率决定是否添加当前图像到数据列表
            if sample_freq < 1:
                # 采样频率小于1时，按概率采样（随机数小于采样频率则保留）
                if np.random.randn() < sample_freq:
                    data_list.append(f'{folder}/{imgname}\n')
            else:
                # 采样频率大于等于1时，重复添加对应次数（稀有动作多采样）
                for _ in range(int(sample_freq)):
                    data_list.append(f'{folder}/{imgname}\n')
            
            # 以下是注释掉的其他采样策略（可能是早期尝试）：
            # 策略1：基于单个rig维度的动作判断
            # 给每一个rig维度都附一个采样频率，然后判断每个rig是否有动作，然后取最大值进行重复采样。
            # 但是这样独立地统计每一个rig的频率并不能很好地表征动作出现的频率。
            # rig_action_index = np.where(rig>0.2)[0]
            # rig_action_index_mouth = np.array([i for i in rig_action_index if i > 19])
            # if rig_action_index_mouth.shape[0]>0:
            #     sample_n = np.max(rigs_sample_freq[rig_action_index_mouth])
            # else:
            #     count_no_action += 1
            #     sample_n = 0.1 # 如果没动作 采样频率是0.1
            # if sample_n < 1:
            #     if np.random.randn() < sample_n:
            #         data_list.append(f'{folder}/{imgname}\n')
            # else:
            #     for _ in range(int(sample_n)):
            #         data_list.append(f'{folder}/{imgname}\n')
            
            # 策略2：基于rig参数最大值判断动作
            # 根据rig的最大值，来判断是否有动作，有动作的进行采样，没动作的进行0.1频率的重采样
            # if np.max(rig[2:]) > 0.3:
            #     data_list.append(f'{folder}/{imgname}\n')
            #     # print('avaiable image', imgname)
            # elif np.random.randn() < 0.1:
            #     data_list.append(f'{folder}/{imgname}\n')
    # 以下为统计信息（未实际使用，仅用于调试）
    sample_freqs = np.array(sample_freqs)
    sample_freqs_log = np.log(sample_freqs)
    # 打印无动作样本的数量
    print('samples with no actions', count_no_action)
    # 将rig参数转换为数组（便于后续分析）
    rigs = np.array(rigs)
    # 打印有效文件夹列表
    print('avaible folder: ', avaible_folders)
    # 打印最终筛选出的样本总数
    print('total data:', len(data_list))
    # 将筛选后的图像路径列表写入文本文件
    with open(save_path, 'w') as f:
        f.writelines(data_list)

# 筛选L36角色的测试集图像数据（侧重动作相关样本），通过重采样平衡不同动作的样本数量，生成标准化测试图像列表
# 与filter_image_data_with_actions_233的主要区别：专注于测试集，代码更精简，采样逻辑更稳定
def filter_image_data_with_actions_233_clean():
    # 设置随机种子，确保采样结果可复现
    random.seed(0)
    # 定义图像数据根目录（L36角色的图像文件夹）
    root = r'/project/qiuf/DJ01/L36/images'
    # 定义筛选后测试图像列表的保存路径（文件名含666标识，与训练集区分）
    save_path = '/project/qiuf/DJ01/L36/images_test_list_actions666.txt'
    # 获取根目录下的所有文件夹名称
    folders = os.listdir(root)
    # 用于存储最终筛选出的图像路径列表
    data_list = []
    # 过滤出真正的文件夹（排除文件，只保留目录）
    folders = [f for f in folders if os.path.isdir(os.path.join(root, f))]
    # 打乱文件夹顺序（随机化数据分布）
    random.shuffle(folders)
    # 取后10%的文件夹作为测试集（与训练集的90%划分对应）
    folders = folders[int(len(folders)*0.9):]  # 训练集90%， 测试集10%
    # 以下为注释掉的扩展测试集方案（可手动添加特定文件夹到测试集）
    # folders += ['qiufeng011', 'wjj01', 'qiufeng02', 'qiufeng03', 'wjj01_blink', 'qiufeng02_blink', 
                # 'qiufeng03_blink', 'L36face233_ZXS_PM_233_nielian_c05_02_55201_1', 'L36face233_ZXS_PM_233_nielian_c08_55212_1']
    # folders = folders[int(len(folders)*0.9):] # 训练集90%， 测试集10%
    # 存储有效文件夹（包含符合条件的图像和对应的rig参数）
    avaible_folders = []
    # 加载L36角色的rig参数文件（使用rigs666.pkl，与训练集的rigs71.pkl区分）
    # 统计不同维度的出现频次来重采样数据。只算嘴
    with open(r'/project/qiuf/DJ01/L36/rigs666.pkl', 'rb') as f:
        rigs_data = pickle.load(f)
    # 对rig参数按文件名中的数字排序（确保时序一致性，如按帧号排序）
    # 排序依据：文件名中"/"分隔后的第二个部分（通常是帧号）转换为整数
    rigs_data = {k: v for k, v in sorted(rigs_data.items(), key=lambda x: int(x[0].split('.')[0].split('/')[1]))}
    # 提取所有rig参数值，每10个取1个（减少计算量），转换为NumPy数组
    rigs = np.array(list(rigs_data.values()))[::10]
    # rigs = []

    # 控制采样频率的系数（让常见动作采样频率≈1，无表情≈0.1）
    coff_count = 5000 # 控制不同表情的采样频率，尽量让常见表情的采样频率=1. 无表情的采样频率=0.1
    # 提取所有rig参数中的嘴部部分（索引19以后，假设这部分对应嘴部动作）
    rigs_mouth = np.array(rigs)[:, 19:]
    # 遍历每个测试集文件夹（带进度索引e）
    for e, folder in enumerate(folders):
        # 打印当前处理进度（第e个/总文件夹数）和文件夹名称
        print(f'[{e}/{len(folders)}]:', folder)
        # 获取当前文件夹下的所有图像文件名
        imgnames = os.listdir(os.path.join(root, folder))  
        # 尝试按图像文件名中的数字排序（确保按帧顺序处理）
        try:
           imgnames.sort(key=lambda x: int(x.split('.')[0]))
        except:
            # 排序失败则跳过该文件夹
            continue
        # 构建当前文件夹对应的rig参数文件夹路径（假设rig与images目录结构平行）
        rigs_folder = os.path.join(root.replace('images', 'rigs'), folder)
        # 检查rig参数文件夹是否存在，不存在则跳过
        if not os.path.exists(rigs_folder):
            print(f'{rigs_folder} do not has corresponding rigs folder!')
            continue
        # 检查图像数量与rig参数文件数量是否一致（确保每一帧都有对应的驱动参数）
        if len(imgnames) != len(os.listdir(rigs_folder)):
            print(f'{folder} images and rigs has different frames!')
            continue
        # 将有效文件夹加入列表
        avaible_folders.append(folder)
        
        # 遍历当前文件夹下的每个图像（带进度条）
        for i, imgname in tqdm(enumerate(imgnames)):
            # 构建当前图像对应的rig参数文件名（从rigs_data字典中查找）
            rigname = os.path.join(folder, imgname.replace('.0000.jpg', '.txt'))
            # 尝试从rigs_data中获取当前图像的rig参数
            if rigname in rigs_data:
                rig = rigs_data[rigname]
            else:
                # 若未找到，直接从rig文件夹中读取rig参数文件（调用read_L36_ctrls函数）
                rigname = os.path.join(rigs_folder, imgname.replace('.0000.jpg', '.txt'))
                rig = read_L36_ctrls(os.path.join(root, rigname))
            
            # 核心采样逻辑：基于当前rig与已有rig的相似性计算采样频率
            # 1. 计算当前嘴部rig参数与所有已有嘴部rig参数的欧氏距离（衡量动作相似度）
            # 计算当前rig和所有rigs的距离，小于阈值的认为是类似的动作。这个动作出现的次数越少，采样频率越高。 
            dist = np.linalg.norm(rigs_mouth - rig[19:], axis=1)
            # 2. 统计相似动作的数量（距离小于0.5的视为相似）
            cnt_similar = np.sum(dist<0.5) 
            # 3. 计算采样频率：相似动作越少（cnt_similar越小），采样频率越高（log放大差异）
            sample_freq = np.log(coff_count/cnt_similar)
            # 限制采样频率范围（最小0.01，最大10），避免极端值影响
            sample_freq = min(max(0.01, sample_freq), 10) # 控制在10倍和0.1倍之间。 
            # 根据采样频率决定是否添加当前图像到数据列表
            if sample_freq < 1:
                # 采样频率小于1时，按概率采样（随机数小于采样频率则保留）
                if np.random.randn() < sample_freq:
                    data_list.append(f'{folder}/{imgname}\n')
            else:
                # 采样频率大于等于1时，重复添加对应次数（稀有动作多采样）
                for _ in range(int(sample_freq)):
                    data_list.append(f'{folder}/{imgname}\n')
    # 打印有效文件夹列表（用于验证测试集包含的文件夹）
    print('avaible folder: ', avaible_folders)
    # 打印最终筛选出的测试样本总数
    print('total data:', len(data_list))
    # 将筛选后的测试图像路径列表写入文本文件
    with open(save_path, 'w') as f:
        f.writelines(data_list)

# 筛选L36_230_61虚拟角色的图像数据，基于动作重采样平衡样本，并划分训练集和测试集，生成对应的图像列表文件
# 与233版本的区别：针对L36_230_61角色，支持同时生成训练集和测试集列表，路径处理更灵活
def filter_image_data_with_actions_230_clean():
    # 设置随机种子，确保采样和划分结果可复现
    random.seed(0)
    
    # 定义L36_230_61角色的图像数据根目录
    root = r'/project/qiuf/DJ01/L36_230_61/images'
    # rig参数文件夹的标识（用于构建rig文件路径，与文件名中的数字对应）
    rig_folders_n = '321'

    # 获取根目录下的所有文件夹名称
    folders = os.listdir(root)
    # 用于存储筛选和重采样后的图像路径列表
    data_list = []
    # 以下为注释掉的指定文件夹方案（可手动指定需要处理的文件夹）
    # folders = ['230_first', '230_second', 'L36_230#230_233_qujia_nanwanjia_60071_4', 'L36_230#230_233_huazhang_60069_2',
    #            'L36_230#233_fanghedeng02_59838_1', 'L36_230#230_233_wangtian_60064_4']
    # 打乱文件夹顺序（随机化数据分布）
    random.shuffle(folders)
    # 以下为注释掉的数据集划分方案（直接取后10%作为测试集，当前已改用基于有效文件夹的划分）
    # folders = folders[int(len(folders)*0.9):] # 训练集90%， 测试集10%
    # folders += ['qiufeng011', 'wjj01', 'qiufeng02', 'qiufeng03', 'wjj01_blink', 'qiufeng02_blink', 
                # 'qiufeng03_blink', 'L36face233_ZXS_PM_233_nielian_c05_02_55201_1', 'L36face233_ZXS_PM_233_nielian_c08_55212_1']
    # folders = folders[int(len(folders)*0.9):] # 训练集90%， 测试集10%
    
    # 存储有效文件夹（包含符合条件的图像和对应的rig参数）
    avaible_folders = []
    # 加载L36_230_61角色的rig参数文件（路径为root替换images为rigs，加上rig_folders_n标识）
    # 统计不同维度的出现频次来重采样数据。只算嘴
    with open(root.replace('images', 'rigs')+str(rig_folders_n)+'.pkl', 'rb') as f:
        rigs_data = pickle.load(f)
    # 以下为注释掉的rig参数排序逻辑（当前未启用，可能因数据已按序存储）
    # rigs_data = {k: v for k, v in sorted(rigs_data.items(), key=lambda x: int(x[0].split('.')[0].split('/')[1]))}
    # 提取所有rig参数值，每10个取1个（减少计算量），转换为NumPy数组
    rigs = np.array(list(rigs_data.values()))[::10]
    # rigs = []
    
    # 控制采样频率的系数（让常见动作采样频率≈1，无表情≈0.1）
    coff_count = 5000 # 控制不同表情的采样频率，尽量让常见表情的采样频率=1. 无表情的采样频率=0.1
    # 提取所有rig参数中的嘴部部分（索引19以后，对应嘴部动作参数）
    rigs_mouth = np.array(rigs)[:, 19:]
    # 遍历每个文件夹（带进度索引e）
    for e, folder in enumerate(folders):
        # 打印当前处理进度（第e个/总文件夹数）和文件夹名称
        print(f'[{e}/{len(folders)}]:', folder)
        # 递归遍历当前文件夹及其子文件夹，收集所有jpg和png图像路径
        imgnames = [y for x in os.walk(os.path.join(root, folder)) for y in glob.glob(os.path.join(x[0], '*.jpg'))]
        imgnames += [y for x in os.walk(os.path.join(root, folder)) for y in glob.glob(os.path.join(x[0], '*.png'))]
        # 构建当前文件夹对应的rig参数文件夹路径（root中images替换为rigs）
        rigs_folder = os.path.join(root.replace('images', 'rigs'), folder)
        # 检查rig参数文件夹是否存在，不存在则跳过
        if not os.path.exists(rigs_folder):
            print(f'{rigs_folder} do not has corresponding rigs folder!')
            continue
         # 以下为注释掉的图像与rig数量校验（当前未启用，可能因子文件夹结构复杂）
        # rigs_files =  [y for x in os.walk(rigs_folder) for y in glob.glob(os.path.join(x[0], '*.txt'))]
        # if len(imgnames) != len(rigs_files):
        #     print(f'{folder} images and rigs has different frames!')
        #     continue
        # 收集当前文件夹下所有图像的父文件夹（去重），作为有效文件夹
        avaible_folders+=(list(set([e.split('/')[-2] for e in imgnames])))
        
        # 遍历当前文件夹下的每个图像（带进度条）
        for i, imgname in tqdm(enumerate(imgnames)):
            # 构建当前图像对应的rig参数文件名（替换后缀，去除root路径，得到相对路径）
            rigname = imgname.replace('.0000.jpg', '.txt').replace(root, '').strip('/')
            rigname = imgname.replace('.0000.png', '.txt').replace(root, '').strip('/')  # 同时处理png格式
            # 尝试从rig_data中获取当前图像的rig参数
            if rigname in rigs_data:
                rig = rigs_data[rigname]
            else:
                # 若未找到，尝试直接从rig文件夹读取rig参数文件
                try:
                    rigname = os.path.join(rigs_folder, imgname.replace('.0000.jpg', '.txt'))
                    rig = read_L36_ctrls(os.path.join(root, rigname))
                except:
                    # 读取失败则打印错误并跳过
                    print(f'{rigname} not exists')            
                    continue
            # 计算当前rig和所有rigs的距离，小于阈值的认为是类似的动作。这个动作出现的次数越少，采样频率越高。 
            # 核心采样逻辑：基于动作相似度计算采样频率
            # 1. 计算当前嘴部rig参数与所有已有嘴部rig参数的欧氏距离（衡量动作相似度）
            dist = np.linalg.norm(rigs_mouth - rig[19:], axis=1)
            # 2. 统计相似动作的数量（距离小于0.5的视为相似）
            cnt_similar = np.sum(dist<0.5) 
            # 3. 计算采样频率：相似动作越少，采样频率越高（log放大差异）
            sample_freq = np.log(coff_count/cnt_similar)
            # 限制采样频率范围（0.01~10），避免极端值
            sample_freq = min(max(0.01, sample_freq), 10) # 控制在10倍和0.1倍之间。 
            # 根据采样频率添加图像路径到数据列表（存储相对路径）
            if sample_freq < 1:
                # 采样频率小于1时，按概率采样
                if np.random.randn() < sample_freq:
                    data_list.append(imgname.replace(root, '').strip('/')+'\n')
            else:
                # 采样频率大于等于1时，重复添加对应次数
                for _ in range(int(sample_freq)):
                    data_list.append(imgname.replace(root, '').strip('/')+'\n')
            

    # 划分测试集：取有效文件夹的后10%作为测试文件夹
    test_folder = avaible_folders[int(len(avaible_folders)*0.9):]
    # 筛选出属于测试文件夹的图像路径，作为测试集列表
    test_list = [e for e in data_list if e.split('/')[-2] in test_folder]
    # 定义测试集列表保存路径（含root和rig标识）
    save_path = f'{root}_test_list_actions{rig_folders_n}.txt'
    # 写入测试集列表文件
    with open(save_path, 'w') as f:
        f.writelines(test_list)
    
    # 划分训练集：取有效文件夹的前90%作为训练文件夹
    train_folder = avaible_folders[:int(len(avaible_folders)*0.9)]
    # 以下为注释掉的训练集扩展方案（未启用）
    # train_folder += folders[2:]
    # 筛选出属于训练文件夹的图像路径，作为训练集列表
    train_list = [e for e in data_list if e.split('/')[-2] in train_folder]
    # 定义训练集列表保存路径
    save_path = f'{root}_train_list_actions{rig_folders_n}.txt'
    # 写入训练集列表文件
    with open(save_path, 'w') as f:
        f.writelines(train_list)
    # 打印总数据量
    print('total data:', len(data_list))
    # 打印训练集数据量和对应的文件夹
    print('train data:', len(train_list))
    print('train folder: ', train_folder)
    # 打印测试集数据量和对应的文件夹
    print('test data:', len(test_list))
    print('test folder: ', test_folder)

# 筛选L36角色中“从230重定向”的图像数据，基于动作重采样生成测试集列表，专注于处理跨角色重定向后的样本
# 注：“retarget_from_230”表示这些图像是从L36_230角色重定向到L36角色的，需针对性处理
def filter_image_data_with_actions_230():
    # 设置随机种子，确保采样结果可复现
    random.seed(0)
    # 定义图像数据根目录（L36角色中从230重定向的图像文件夹）
    root = r'/project/qiuf/DJ01/L36/images_retarget_from_230'
    # 定义筛选后测试图像列表的保存路径
    save_path = '/project/qiuf/DJ01/L36/images_test_list_actions_230.txt'
    # 获取根目录下的所有文件夹名称
    folders = os.listdir(root)
    # 用于存储筛选和重采样后的图像路径列表
    data_list = []
    # 过滤出真正的文件夹（排除文件，只保留目录）
    folders = [f for f in folders if os.path.isdir(os.path.join(root, f))]
    # 打乱文件夹顺序（随机化数据分布）
    random.shuffle(folders)
    # 取后10%的文件夹作为测试集（与训练集的90%划分对应）
    folders = folders[int(len(folders)*0.9):]  # 训练集90%， 测试集10%
    # 存储有效文件夹（包含符合条件的图像和对应的rig参数）
    avaible_folders = []
    # 统计不同维度的出现频次来重采样数据。只算嘴
    # 构建rig参数根目录路径（将images替换为rigs）
    rigs_root = root.replace('images', 'rigs')
    # 定义rig参数pkl文件路径（与rig根目录同名，方便查找）
    rigs_pkl = rigs_root + '.pkl'
    # 检查rig参数pkl文件是否存在，存在则直接加载
    if os.path.exists(rigs_pkl):        
        with open(rigs_pkl, 'rb') as f:
            rigs_data = pickle.load(f)
        # 对rig参数按文件名中的数字排序（确保时序一致性）
        rigs_data = {k: v for k, v in sorted(rigs_data.items(), key=lambda x: int(x[0].split('.')[0].split('/')[1]))}
    else:
        # 若pkl文件不存在，则遍历rig文件夹读取所有rig参数并生成pkl文件（首次运行时执行）
        rigs_data = {}
        # 获取rig根目录下的所有文件夹
        rig_folders = os.listdir(rigs_root)
        # 遍历每个rig文件夹（带进度条）
        for rig_folder in tqdm(rig_folders, total=len(rig_folders)):
            # 获取当前rig文件夹下的所有文件
            rig_files = os.listdir(os.path.join(rigs_root, rig_folder))
            # 遍历每个rig文件
            for rigf in rig_files:
                # 构建rig文件完整路径
                rig_path = os.path.join(rigs_root, rig_folder, rigf)
                # 读取rig参数（调用read_L36_ctrls函数）
                rig = read_L36_ctrls(rig_path)
                # 存储到rig_data字典（键为相对路径）
                rigs_data[os.path.join(rig_folder, rigf)] = rig
        # 将读取的rig参数保存为pkl文件（避免下次重复读取）
        with open(rigs_pkl, 'wb') as f:
            pickle.dump(rigs_data, f)
    # 将rig参数转换为NumPy数组
    rigs = np.array(list(rigs_data.values()))
    # 筛选出长度为61的rig参数（过滤无效或格式错误的参数）
    rigs = np.array([np.array(rig) for rig in rigs if len(rig) == 61])
    # 打印有效rig参数的形状（便于验证数据格式）
    print(rigs.shape)
    # 控制采样频率的系数（让常见动作采样频率≈1，无表情≈0.1）
    coff_count = 5000 # 控制不同表情的采样频率，尽量让常见表情的采样频率=1. 无表情的采样频率=0.1
    # 提取所有rig参数中的嘴部部分（索引20以后，对应嘴部动作参数），每10个取1个（减少计算量）
    rigs_mouth = np.array(rigs)[:, 20:][::10]
    # 遍历每个测试集文件夹（带进度索引e）
    for e, folder in enumerate(folders):
        # 打印当前处理进度（第e个/总文件夹数）和文件夹名称
        print(f'[{e}/{len(folders)}]:', folder)
        # 获取当前文件夹下的所有图像文件名
        imgnames = os.listdir(os.path.join(root, folder))
        # 尝试按图像文件名中的数字排序（确保按帧顺序处理）
        try:
           imgnames.sort(key=lambda x: int(x.split('.')[0]))
        except:
            # 排序失败则跳过该文件夹
            continue
        # 构建当前文件夹对应的rig参数文件夹路径
        rigs_folder = os.path.join(root.replace('images', 'rigs'), folder)
        # 检查rig参数文件夹是否存在，不存在则跳过
        if not os.path.exists(rigs_folder):
            print(f'{rigs_folder} do not has corresponding rigs folder!')
            continue
        # 检查图像数量与rig参数文件数量是否一致（确保每一帧都有对应的驱动参数）
        if len(imgnames) != len(os.listdir(rigs_folder)):
            print(f'{folder} images and rigs has different frames!')
            continue
        # 将有效文件夹加入列表
        avaible_folders.append(folder)
        # 遍历当前文件夹下的每个图像（带进度条）
        for i, imgname in tqdm(enumerate(imgnames)):
            # 构建当前图像对应的rig参数文件名（从rigs_data字典中查找）
            rigname = os.path.join(folder, imgname.replace('.0000.jpg', '.txt'))
            # 尝试从rigs_data中获取当前图像的rig参数
            if rigname in rigs_data:
                rig = rigs_data[rigname]
            else:
                # 若未找到，直接从rig文件夹中读取rig参数文件
                rigname = os.path.join(rigs_folder, imgname.replace('.0000.jpg', '.txt'))
                rig = read_L36_ctrls(os.path.join(root, rigname))
            # 计算当前rig和所有rigs的距离，小于阈值的认为是类似的动作。这个动作出现的次数越少，采样频率越高。 
            # 核心采样逻辑：基于动作相似度计算采样频率
            # 1. 计算当前嘴部rig参数与所有已有嘴部rig参数的欧氏距离（衡量动作相似度）
            dist = np.linalg.norm(rigs_mouth - rig[20:], axis=1)
            # 2. 统计相似动作的数量（距离小于0.5的视为相似）
            cnt_similar = np.sum(dist<0.5) 
            # 3. 计算采样频率：相似动作越少，采样频率越高（log放大差异）
            sample_freq = np.log(coff_count/cnt_similar)
            # 根据采样频率决定是否添加当前图像到数据列表
            if sample_freq < 1:
                # 采样频率小于1时，按概率采样
                if np.random.randn() < sample_freq:
                    data_list.append(f'{folder}/{imgname}\n')
            else:
                # 采样频率大于等于1时，重复添加对应次数
                for _ in range(int(sample_freq)):
                    data_list.append(f'{folder}/{imgname}\n')
        # break
    # 将rig参数转换为数组（便于后续分析）
    rigs = np.array(rigs)
    # 打印有效文件夹列表
    print('avaible folder: ', avaible_folders)
    # 打印最终筛选出的测试样本总数
    print('total data:', len(data_list))
    # 将筛选后的测试图像路径列表写入文本文件
    with open(save_path, 'w') as f:
        f.writelines(data_list)

# 检查Faceware动作捕捉数据与渲染图像是否配对（时序对应），并将配对的图像拼接保存用于可视化验证
# Faceware：一种面部动作捕捉系统，用于记录真实人脸表情数据；render：基于捕捉数据生成的虚拟角色渲染图像
# 核心目的：验证动作捕捉数据与渲染结果的时序一致性，确保每帧捕捉数据都对应正确的渲染图像
def check_is_faceware2render_paried():
    # 定义Faceware动作捕捉数据的根目录
    faceware_root = '/project/qiuf/DJ01/L36/faceware'
    # 定义虚拟角色渲染图像的根目录
    render_root = '/project/qiuf/DJ01/L36/images'
    # 获取Faceware根目录下的所有文件夹名称（每个文件夹对应一组序列）
    folders = os.listdir(faceware_root)
    # 定义拼接图像的保存根目录（用于可视化查看配对结果）
    save_root = '/project/qiuf/DJ01/L36/temp'
    # 遍历每个文件夹（带索引i）
    for i, fold in enumerate(folders):
        # 构建当前文件夹对应的Faceware数据路径
        faceware_fold = os.path.join(faceware_root, fold)
        # 构建当前文件夹对应的渲染图像路径
        render_fold = os.path.join(render_root, fold)
        # 构建当前文件夹对应的拼接图像保存路径
        save_fold = os.path.join(save_root, fold)
        # 创建保存目录，若已存在则不报错
        os.makedirs(save_fold, exist_ok=True)
    
        # 检查当前文件夹的Faceware数据和渲染图像是否都存在
        if os.path.exists(faceware_fold) and os.path.exists(render_fold):
            # 获取当前Faceware文件夹下的所有图像文件名（动作捕捉帧）
            imgnames = os.listdir(faceware_fold)
            # 打印当前处理进度（第i个/总文件夹数）
            print(f'{i}/{len(folders)}')
            # 遍历当前文件夹下的每个Faceware图像（带进度条）
            for imgname in tqdm(imgnames, total=len(imgnames)):
                # 构建Faceware图像的完整路径
                imgpath_faceware = os.path.join(faceware_fold, imgname)
                # 构建对应的渲染图像路径：从Faceware文件名提取后5位数字作为帧号，拼接为渲染图像文件名（.0000.jpg）
                # 例如：Faceware文件名为"xxx12345.csv"，则渲染图像为"12345.0000.jpg"
                imgpath_render = os.path.join(render_fold, '{}.0000.jpg'.format(int(imgname.split('.')[0][-5:])))
                # 构建拼接图像的保存路径
                imgpath_saved = os.path.join(save_fold, imgname)
                # 若拼接图像已存在，则跳过（避免重复处理）
                if os.path.exists(imgpath_saved):
                    continue
                # 读取Faceware图像和渲染图像
                img_faceware = cv2.imread(imgpath_faceware)
                img_render = cv2.imread(imgpath_render)
                try:
                    # 将两张图像都缩放到256×256，然后垂直拼接（上半部分为Faceware数据，下半部分为渲染结果）
                    img = np.concatenate((cv2.resize(img_faceware, (256, 256)), cv2.resize(img_render, (256, 256))), axis=0)
                    # 保存拼接后的图像
                    cv2.imwrite(imgpath_saved, img)
                except:
                    # 若读取或拼接失败（如图像不存在、尺寸不匹配），则跳过
                    continue

# 更新训练/测试用的图像路径列表，修复其中无效的路径（图像不存在的情况）
# 核心目的：确保列表中的图像路径均可访问，避免模型训练/测试时因文件不存在导致报错
def update_action_list_for_training():
    # 定义旧的图像路径列表文件（需要更新的原始列表）
    old_path = '/project/qiuf/DJ01/L36/images_test_list_actions.txt'
    # 定义图像文件的根目录（用于拼接完整路径验证文件是否存在）
    images_root = '/project/qiuf/DJ01/L36/images'
    # 读取旧列表文件中的所有路径
    with open(old_path, 'r') as f:
        lines = f.readlines()  # lines是包含所有图像路径的列表，每行一个路径
    # 用于存储更新后的有效路径
    new_lines = []
    # 遍历旧列表中的每个路径（带进度条，显示处理进度）
    for line in tqdm(lines, total=len(lines)):
        # 拼接完整的图像路径（根目录 + 列表中的相对路径），并去除行尾的换行符等空白字符
        full_path = os.path.join(images_root, line.strip())
        # 检查该图像文件是否存在
        if not os.path.exists(full_path):
            # 若文件不存在，尝试修正路径：在文件名前添加"_old/"（假设旧文件可能被移动到该子目录）
            imgname = line.split('/')[-1]  # 提取文件名（如"12345.0000.jpg"）
            line = line.replace('/' + imgname, '_old/' + imgname)  # 替换路径，例如"folder/123.jpg"→"folder_old/123.jpg"
        # 将（可能修正过的）路径添加到新列表
        new_lines.append(line)
    # 将更新后的路径列表写入新文件（在旧文件名后添加"_new"标识）
    with open(old_path.replace('.txt', '_new.txt'), 'w') as f:
        f.writelines(new_lines)
    # 函数无返回值（主要通过写入新文件输出结果）
    return

# 统一格式化指定目录下的JPG图像文件名，使其按规范命名（6位数字索引 + .0000.jpg后缀）
# 核心目的：标准化图像文件名格式，便于后续按序号排序或关联其他时序数据（如rig参数、动作帧）
def format_images():
    # 定义需要格式化的图像根目录（这里是L36_230角色下的230_second子文件夹）
    root = '/project/qiuf/DJ01/L36_230/images/230_second'
    # 递归遍历root目录及其所有子文件夹，收集所有.jpg图像的路径
    # 注：虽然变量名是txtfiles，但实际处理的是.jpg文件（可能是命名笔误）
    txtfiles = [y for x in os.walk(root) for y in glob.glob(os.path.join(x[0], '*.jpg'))]
    # 遍历所有收集到的图像文件（带进度条，显示处理总进度）
    for txtfile in tqdm(txtfiles, total=len(txtfiles)):
        try:
            # 从文件名中提取数字索引（假设原文件名是纯数字，如"123.jpg"→提取123）
            # 处理逻辑：分割文件名（取最后一段）→按"."分割→取第一段（数字部分）→转换为整数
            index = int(txtfile.split('/')[-1].split('.')[0])
        except:
            # 若提取索引失败（如文件名含非数字字符），打印错误文件路径并中断处理
            print(txtfile)
            break
        # 定义原文件路径（待重命名的文件）
        src = txtfile
        # 构建新的文件路径：保持原目录，文件名格式化为"6位数字.0000.jpg"（如123→000123.0000.jpg）
        # f'{index:06d}'表示将索引补零至6位数字（确保序号对齐，便于排序）
        dst = os.path.join(os.path.dirname(src), f'{index:06d}.0000.jpg')
        # 执行重命名操作（将原文件改名为新格式的文件名）
        os.rename(src, dst)
    # 函数无返回值（主要通过文件重命名完成操作）
    return

# 向现有训练集列表中添加新的图像数据，扩展训练集规模
# 核心目的：将指定文件夹中的新图像数据合并到已有训练列表，生成更新后的训练集文件
# 函数名可能存在笔误，应为add_new_data_to_train_set（添加新数据到训练集）
def add_new_data_to_transet():  
    # 定义图像数据的根目录（L36_230_61角色的图像文件夹）
    root = r'/project/qiuf/DJ01/L36_230_61/images'
    # 获取根目录下的所有文件夹名称（后续被手动指定的文件夹列表覆盖）
    folders = os.listdir(root)
    # 手动指定需要添加的新数据文件夹（这里重复10次同一文件夹，可能是为了重复采样该文件夹数据以增加权重）
    folders = ['ziva_L36_230_61_processed', 'ziva_L36_230_61_processed','ziva_L36_230_61_processed', 'ziva_L36_230_61_processed', 'ziva_L36_230_61_processed', 
               'ziva_L36_230_61_processed', 'ziva_L36_230_61_processed','ziva_L36_230_61_processed', 'ziva_L36_230_61_processed', 'ziva_L36_230_61_processed']
    # 读取已有的训练集图像列表文件（旧训练集）
    with open('/project/qiuf/DJ01/L36_230_61/images_train_list_actions321.txt', 'r') as f:
        imgs_list = f.readlines()  # 存储旧训练集中的所有图像路径
    # 遍历每个需要添加的文件夹
    for fold in folders:
        # 递归查找当前文件夹及其子文件夹中所有的.png图像路径
        imgs = [y for x in os.walk(os.path.join(root, fold)) for y in glob.glob(os.path.join(x[0], '*.png'))]
        # 将新图像的路径转换为相对路径（去除根目录前缀），并添加换行符，然后合并到旧训练集列表
        imgs_list += [l.replace(root + '/', '') + '\n' for l in imgs]
    # 将合并后的训练集列表写入新文件（文件名后缀从321改为322，标识更新版本）
    with open('/project/qiuf/DJ01/L36_230_61/images_train_list_actions322.txt', 'w') as f:
        f.writelines(imgs_list)
    # 函数无返回值（主要通过写入新文件输出结果）
    return

# 主程序入口：当该脚本被直接运行时，执行以下代码；若作为模块导入则不执行
if __name__ == '__main__':
    # 导入PyTorch的图像变换工具（用于数据预处理）
    import torchvision.transforms.transforms as transforms
    # 以下为注释掉的函数调用（可根据需要取消注释执行对应功能）
    # format_images()  # 调用图像格式化函数（统一文件名）
    # exit()  # 执行后退出程序（用于单独测试某个功能）
    # filter_image_data_with_actions_233_clean()  # 调用233角色的图像筛选函数
    # add_new_data_to_transet()  # 调用训练集扩展函数
    # 加载rig参数到缓存（指定路径、rig参数维度为61、版本666）
    load_rigs_to_cache('/project/qiuf/DJ01/L36/rigs', n_rig=61, version_old=666)
    # exit()  # 执行到此处退出（用于单独测试load_rigs_to_cache功能）
    # filter_image_data_with_actions_230_clean()  # 调用230角色的图像筛选函数
    exit()  # 退出程序（当前配置下，执行完load_rigs_to_cache后即退出）
    # 以下为注释掉的其他功能代码
    # update_action_list_for_training()  # 调用训练列表更新函数
    # exit()
    # 递归查找指定目录下的所有jpg图像并打印数量（用于数据统计）
    # root = '/project/qiuf/L36_drivendata'
    # images = [y for x in os.walk(root) for y in glob.glob(os.path.join(x[0], '*.jpg'))]
    # print(len(images))

    # 定义图像预处理管道： resize到256×256 → 转为Tensor → 标准化（均值0.5，标准差0.5）
    transform2 = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 定义图像数据根目录
    data_path = '/project/qiuf/DJ01/L36/images'
    # 初始化ABAW数据集（自定义数据集类）：指定路径、角色l36、测试集、预处理方法、不随机翻转、不使用 landmarks
    train_dataset = ABAWDataset2(root_path=data_path, character='l36', data_split='test', transform=transform2, random_flip=False, use_ldmk=False)
    # 创建数据加载器：批量大小4，打乱数据顺序
    data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
    )
    # 遍历数据加载器，打印前51个批次的索引和数据（用于验证数据加载是否正常）
    for i, inputs in enumerate(data_loader):
        print(i, inputs)
        if i > 50:  # 只打印前51个批次（索引0到50）
            break
