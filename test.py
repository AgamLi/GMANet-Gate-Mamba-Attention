import os
import numpy as np
import argparse
import yaml
import torch
import torch.utils
from torch.utils.data import DataLoader, Dataset
from mydataset import ImageToImage2D, JointTransform2D, correct_dims
from pathlib import Path
import SimpleITK
import cv2
from tqdm import tqdm
import SimpleITK as sitk
from my_model.Model import Model


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()

    originSize = itkimage.GetSize()  # 获取原图size
    originSpacing = itkimage.GetSpacing()  # 获取原图spacing
    newSize = np.array(newSize, dtype='uint32')
    factor = originSize / newSize
    newSpacing = originSpacing * factor

    resampler.SetReferenceImage(itkimage)  # 指定需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    itkimgResampled.SetOrigin(itkimage.GetOrigin())
    itkimgResampled.SetSpacing(itkimage.GetSpacing())
    itkimgResampled.SetDirection(itkimage.GetDirection())
    return itkimgResampled


class Fetal_dataset(Dataset):
    def __init__(self, list_dir, transform=None):
        self.transform = transform  # using transform in torch!
        images = [SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(str(i))) for i
                  in list_dir[0]]
        labels = [SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(str(i))) for i in
                  list_dir[1]]
        self.images = np.array(images)
        self.labels = np.array(labels)


    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image, mask = correct_dims(self.images[idx].transpose((1, 2, 0)), self.labels[idx])
        sample = {}
        if self.transform:
            image, mask, low_mask = self.transform(image, mask)

        sample['image'] = image
        sample['low_res_label'] = low_mask.unsqueeze(0)
        sample['label'] = mask.unsqueeze(0)

        return sample


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='gmanet_psfh', help='model name')
    args = parser.parse_args()
    return args


def main():
    device = torch.device('cuda')


    # parse args and config
    args = parse_args()
    with open('config/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)


    # create pred and gt folders
    os.makedirs("./pred", exist_ok=True)
    os.makedirs("./gt", exist_ok=True)


    # load model
    ckpt = os.listdir(f"./checkpoints/")[-1]
    model = Model(num_classes=config['num_classes']).cuda()
    model.load_state_dict(torch.load(f"./checkpoints/{ckpt}"))
    print(f"Loaded model:{ckpt}")
    model.to(device)


    # Data loading code
    root_path = Path(config['data_dir'])
    image_files = np.array([(root_path / Path("image_mha") / Path(str(i).zfill(5) + '.mha')) for i in range(1, 5102)])
    label_files = np.array([(root_path / Path("label_mha") / Path(str(i).zfill(5) + '.mha')) for i in range(1, 5102)])
    
    with open(os.path.join(config['data_dir'], 'test.txt'), "r") as file:
        lines = file.readlines()
    test_index = [int(line.strip().split("/")[-1]) - 1 for line in lines]
    print('test images:', len(test_index))


    # transform for data augmentation
    tf_val = JointTransform2D(img_size=256, low_img_size=128, ori_size=256,
                                crop=None, p_flip=0.0, color_jitter_params=None, long_mask=True)
    
    
    # create test dataset and dataloader
    db_test = Fetal_dataset(transform=tf_val,
                            list_dir=(image_files[np.array(test_index)], label_files[np.array(test_index)])
                            )
    testloader = DataLoader(db_test, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)


    # test the model
    with torch.no_grad():
        model.eval()
        a=0
        for batch_idx, (datapack) in tqdm(enumerate(testloader), ncols=150, total=len(testloader), desc='Testing Progress', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed}, remaining: {remaining}]'):
            imgs = datapack['image'].to(dtype=torch.float32, device='cuda')
            masks = datapack['label'].to(dtype=torch.float32, device='cuda')  # (b,1,256,256)
            preds = model(imgs)
            for i in range(preds.shape[0]):
                pred = preds[i].argmax(dim=0).detach().cpu().numpy()
                label = masks[i].squeeze(0).long().detach().cpu().numpy()
                # pred = sitk.GetArrayFromImage(resize_image_itk(sitk.GetImageFromArray(pred), (256, 256)))
                # label = sitk.GetArrayFromImage(resize_image_itk(sitk.GetImageFromArray(label), (256, 256)))
                cv2.imwrite(f"./pred/{a + 1}.png", pred)
                cv2.imwrite(f"./gt/{a + 1}.png", label)
                a=a+1


if __name__ == '__main__':
    main()
