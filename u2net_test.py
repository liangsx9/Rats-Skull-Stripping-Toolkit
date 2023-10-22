import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB


###nii2jpg###

import nibabel as nib
import numpy as np
import imageio
import os

def read_niifile(niifile):  # 读取niifile文件
    img = nib.load(niifile)  # 下载niifile文件（提取文件）
    img_fdata = img.get_fdata()  # 获取niifile数据
    img90 = np.rot90(img_fdata) #旋转90度
    #return img_fdata
    return img90


def save_fig(file):  # 保存为图片
    fdata = read_niifile(file)  # 调用上面的函数，获得数据
    (y, x, z) = fdata.shape  # 获得数据shape信息：（长，宽，维度-切片数量）
    for k in range(z):
        silce = fdata[:, :, k]
        #silce = fdata[k, :, :]  # 三个位置表示三个不同角度的切片
        imageio.imwrite(os.path.join(savepicdir, '{}.jpg'.format(k)), silce)
        # 将切片信息保存为jpg格式





#print(list(list_nii))
def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield f

base = os.path.join(os.getcwd(), 'nii') # nii文件的路径
output = os.path.join(os.getcwd(), 'test_data', 'test_images') # 保存jpg的路径
for i in findAllFile(base):
    NiiFileName = i
    dir = os.path.join(base,i)
    savepicdir = (output)
    save_fig(dir)
    #os.mkdir(savepicdir) #新建文件夹，重命名为nii文件名称
    #save_fig(dir)

#####nii2jpg_end#####



# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.jpg')

def main():

    # --------- 1. get image path and name ---------
    model_name='u2netp'#u2net

    #设定数据路径
    import os
    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + '.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP")
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7


    ###jpg2nii###
    
    import nibabel as nib  
    from PIL import Image  
    import numpy as np  
    import os  
      
    # 指定 PNG 图像所在的文件夹路径  
    png_folder = os.path.join(os.getcwd(), 'test_data', model_name + '_results')
    # 获取文件夹中所有的 PNG 图像文件名 
    png_files = [f for f in os.listdir(png_folder) if f.endswith('.jpg')]  
    # 将文件名重新排序，避免出现1,10,2,20排序的情况
    png_files.sort(key=lambda x:int(x.split('.')[0]))
      
    # 确定图像大小和通道数  
    img_shape = None  
    img_channels = None  
    for f in png_files:  
        img = Image.open(png_folder + os.sep + f)  
        if img_shape is None:  
            img_shape = img.size  
        if img_channels is None:  
            img_channels = img.mode  
        if img_shape != img.size or img_channels != img.mode:  
            raise ValueError("All images must have the same size and number of channels.")  
      
    # 创建一个空的 numpy 数组，用于存储所有 PNG 图像的数据  
    img_data = np.zeros(img_shape + (len(png_files),), dtype=np.uint8)  
      
    # 读取所有 PNG 图像并将其数据存储到 numpy 数组中  
    for i, f in enumerate(png_files):  
        img = Image.open(png_folder+ os.sep + f)  
        #90度旋转图像以正常显示
        img_array = np.array(img.rotate(-90,expand=1))  
        img_data[:,:,i] = img_array[:,:,0]
    
    # 二值化，消除边缘模糊
    img_data[img_data <= 128] = 0
    img_data[img_data > 128] = 1
    
      
    # 创建一个新的 NIfTI 图像对象，将 numpy 数组作为数据传递给该对象  
    nii_img = nib.Nifti1Image(img_data, np.eye(4))  
      
    # # 确定 NIfTI 图像的数据类型  
    # if img_channels == 'L':  
    #     data_type = np.uint8  
    # else:  
    #     data_type = np.uint8 if img_channels == 'RGB' else np.uint16  
      
    # # 创建一个新的 NIfTI 图像对象，将 numpy 数组作为数据传递给该对象  
    # img_data = np.transpose(img_data, axes=(1,0,2)) 
    # img_data = np.transpose(img_data, axes=(1,0,2)) 
    # nii_img = nib.Nifti1Image(img_data.astype(data_type), np.eye(4))  
      
    # 保存 NIfTI 图像  
    nib.save(nii_img, os.getcwd() + os.sep + 'results' + os.sep + 'mask_' + NiiFileName)
    
    #####jpg2nii_end######
    
    ####将nii与mask相乘####
    #将图像回正
    nii_input_data = np.rot90(np.rot90(np.rot90(read_niifile(dir))))
    
    mask_data = img_data
    # 将两个图像逐像素对应相乘  
    result = nii_input_data * mask_data  

    # 创建一个新的 NIfTI 图像对象，将结果保存为 NIfTI 文件  
    nii_result = nib.Nifti1Image(result, nii_img.affine)  
    nib.save(nii_result,  os.getcwd() + os.sep + 'results' + os.sep + 'segresult_' + NiiFileName)
    ####将nii与mask相乘_end####



if __name__ == "__main__":
    main()
