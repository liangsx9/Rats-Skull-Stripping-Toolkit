# Rats-Skull-Stripping-Toolkit
A python program for Rats Skull Stripping 

## 用途：
## Application：
本程序可以利用输入的脑影像，经过神经网络的处理，实现大鼠核磁影像的颅骨快速剥离，最终输出剥离后的大脑掩模及脑组织。省去了手动勾画的步骤，节省时间及精力。  
This program can utilize the input brain images and process them through a neural network to achieve rapid skull dissection of rat MRI images, ultimately outputting the stripped brain mask and brain tissue. Eliminated manual sketching steps, saving time and effort.  

## 安装方法：
## Installation：  
**运行环境**:  
**Operating environment**:  
需要python3.0及以上。  
Requires Python 3.0 and above.  
  
需要安装的包：  
Package to be installed:  
```
nibabel  
pillow  
numpy  
os  
imageio  
scikit-image  
torch  
torchvision  
glob2
```

## 文件夹结构：
## Folder structure：  
```
Rats Skull Stripping.  
                    │  data_loader.py  
                    │  u2net_test.py  
                    │  
                    ├─model  
                    │  │  u2net.py  
                    │  │  u2net_refactor.py  
                    │  │  init.py  
                    │  │  
                    │  └─__pycache__  
                    │  
                    ├─nii  
                    │      T2TurboRARE.nii  
                    │  
                    ├─results  
                    │      mask_T2TurboRARE.nii  
                    │      segresult_T2TurboRARE.nii  
                    │  
                    ├─saved_models  
                    │      u2netp.pth  
                    │      u2netp0.pth  
                    │  
                    ├─test_data  
                    │  ├─test_images  
                    │  │      0.jpg  
                    │  │      1.jpg  
                    │  │      ...  
                    │  │  
                    │  └─u2netp_results  
                    │          0.jpg  
                    │          1.jpg  
                    │          ...  
                    │  
                    └─__pycache__
```

## 使用方法：
## Usage method：
1.将需要剥离颅骨的大鼠nifti核磁影像文件放在nii文件夹下，当前版本推荐一次放一例数据。  
1.Place the nifti magnetic resonance imaging files of rats that require skull dissection in the nii folder. The current version recommends placing one case of data at a time.  
2.运行 u2net_test.py .  
2.Run the u2net_test.py。  

## 结果查看：
## Result viewing：
- 剥离结果为nifti文件，存放于 `results` 文件夹下，其中 `mask_`开头的文件为剥离出来的二值化掩膜，`segresult_`开头的文件为剥除颅骨的脑组织。
- The stripping result is a nifti file, stored in the `results` folder, where `mask_` The starting file is the stripped binary mask, `segresult_` The starting file is the brain tissue removed from the skull.  
- nii文件推荐使用mricro、mricron软件查看，也可在 `\test_data\u2netp_results` 文件夹下直接查看每一层的颅骨剥离结果，剥离前的逐层原始图像可在 `\test_data\test_images`文件夹下查看，可用于进行效果对比。
- It is recommended to use mricro or mricron software to view NII files, and can also be found in `\test_data\u2netp_results`  Directly view the skull dissection results of each layer in the results folder, and the original images of each layer before dissection can be found in the `\test_data\test_images` folder, which can be used for effect comparison.  

## 模型选择：
## Model selection：
`u2netp0.pth`代表利用13627张影像训练138200次后保存得到的权重。  
`u2netp0.pth` represents the weight saved after training 138200 times using 13627 images.  
`u2netp.pth`代表利用13627张影像训练584600次后保存得到的权重。  
`u2netp.pth` represents the weight saved after training 584600 times using 13627 images.  
需要更换模型时，将模型重命名为`u2netp.pth`并在 `saved_models` 文件夹下替换即可。  
When you need to replace the model, rename it to `u2netp.pth` and just replace it in the `saved_models` folder.  

## 剥离效果图：
## Peeling effect diagram：
![image](https://github.com/DDDRN/Rats-Skull-Stripping-Toolkit/assets/42291489/a758b460-f1ab-46fe-81ed-3bb61b169abc)
## 运行演示:
## Run demonstration:
![GIF](https://github.com/DDDRN/Rats-Skull-Stripping-Toolkit/assets/42291489/ffd8479d-efe9-4945-ba08-089c201535d0)

如果有意见或建议，请邮件联系：[sxliang@fjtcm.edu.cn](sxliang@fjtcm.edu.cn)  
If you have any comments or suggestions, please email:[sxliang@fjtcm.edu.cn](sxliang@fjtcm.edu.cn)  

如果此工具对您有帮助，请引用此文献：  
If this tool is helpful to you, please refer to this literature:  
>***Liang S, Yin X, Huang L, Huang J, Yang J, Wang X, Peng L, Zhang Y, Li Z, Nie B, Tao J. Automatic brain extraction for rat magnetic resonance imaging data using U^2-Net. Phys Med Biol. 2023 Oct 2;68(20). doi: >10.1088/1361-6560/acf641.PMID:37659398.***
          
  

