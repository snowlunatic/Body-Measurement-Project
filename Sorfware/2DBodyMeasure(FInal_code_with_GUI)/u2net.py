import os
from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT, ToTensorLab, SalObjDataset
from model import U2NET  # full size version 173.6 MB

# Normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    return (d - mi) / (ma - mi)

def save_output(image_name, pred, d_dir):
    predict = pred.squeeze().cpu().data.numpy()
    im = Image.fromarray(predict * 255).convert('RGB')

    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    img_name = os.path.basename(image_name)
    imidx = os.path.splitext(img_name)[0]
    if not os.path.exists(d_dir):
        os.makedirs(d_dir, exist_ok=True)
    imo.save(os.path.join(d_dir, f"{imidx}.png"))

def run_u2net_inference(image_dir, prediction_dir, model_path, num_images=2):
    """
    Run U2NET inference on the given image directory and save results.

    Parameters:
        image_dir (str): Path to the directory containing images.
        prediction_dir (str): Path to the directory where results will be saved.
        model_path (str): Path to the U2NET model file.
        num_images (int): Number of images to process. Default is 2.
    """
    # Get image paths
    img_name_list = glob.glob(os.path.join(image_dir, '*'))[:num_images]
    if not img_name_list:
        print(f"No images found in {image_dir}.")
        return

    print("Processing images:", img_name_list)

    # Dataloader
    test_salobj_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
    )
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Load model
    print("...load U2NET---173.6 MB")
    net = U2NET(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()

    # Inference
    for i_test, data_test in enumerate(test_salobj_dataloader):
        print("Inferencing:", img_name_list[i_test])

        inputs_test = data_test['image'].type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, _, _, _, _, _, _ = net(inputs_test)

        # Normalize and save
        pred = normPRED(d1[:, 0, :, :])
        save_output(img_name_list[i_test], pred, prediction_dir)

        del d1
