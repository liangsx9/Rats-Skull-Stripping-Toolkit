# Rats-Skull-Stripping-Toolkit
A python program for Rats Skull Stripping 

## Application：
This program can utilize the input brain images and process them through a neural network to achieve rapid skull dissection of rat MRI images, ultimately outputting the stripped brain mask and brain tissue. Eliminated manual sketching steps, saving time and effort.  

## Installation：  
**Operating environment**:  
Requires Python 3.0 and above.  
  
**Package to be installed**:  
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

## Usage method：
1. Place the nifti magnetic resonance imaging files of rats that require skull dissection in the nii folder. The current version recommends placing one case of data at a time.  
2. Run the u2net_test.py。  

## Result viewing：
- The stripping result is a nifti file, stored in the `results` folder, where `mask_` The starting file is the stripped binary mask, `segresult_` The starting file is the brain tissue removed from the skull.  
- It is recommended to use mricro or mricron software to view NII files, and can also be found in `\test_data\u2netp_results`  Directly view the skull dissection results of each layer in the results folder, and the original images of each layer before dissection can be found in the `\test_data\test_images` folder, which can be used for effect comparison.  

## Model selection：
`u2netp0.pth` represents the weight saved after training 138200 times using 13627 images.  
`u2netp.pth` represents the weight saved after training 584600 times using 13627 images.  
When you need to replace the model, rename it to `u2netp.pth` and just replace it in the `saved_models` folder.  

## Peeling effect diagram：
![277142873-a758b460-f1ab-46fe-81ed-3bb61b169abc](https://i2.100024.xyz/2023/10/29/pe405h.webp)  
## Run demonstration:
![277142896-ffd8479d-efe9-4945-ba08-089c201535d0](https://i2.100024.xyz/2023/10/29/pc8poy.gif)  

If you have any comments or suggestions, please email:[sxliang@fjtcm.edu.cn](sxliang@fjtcm.edu.cn)  

If this tool is helpful to you, please refer to this literature:  
>***Liang S, Yin X, Huang L, Huang J, Yang J, Wang X, Peng L, Zhang Y, Li Z, Nie B, Tao J. Automatic brain extraction for rat magnetic resonance imaging data using U^2-Net. Phys Med Biol. 2023 Oct 2;68(20). doi:10.1088/1361-6560/acf641.PMID:37659398
        
        
        
        
        
        
        
        ***
