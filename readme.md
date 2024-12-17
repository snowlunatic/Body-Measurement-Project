# MIP2024 - Body Measurement Using a 2D camera for Home Fitness 

## 1. Introduction

This repository is for the project conducted in 2024-2 semester of Mechtronics Integration Project, "Body Measurement Using a 2D Camera for Home Fitness". It is composed of source files and software environment configuration. 

The purpose of the project is to come up with a prototype body measurement system that uses 2D camera which is affordable for individual needs. 

## 2. Environment Configuration

This project uses AI model and trains it so related software programs and frameworks are needed.

- PC: Geoforce RTX 3060 laptop (GPU for training: NVIDIA A100 or NVIDIA Tesla T4)
- Anaconda 
  - CUDA: version 11.8
  - cuDNN:  version 8.7.0
  - pyTorch: version 2.1.2
  - Python: version 3.8.20

## 2.1. Anaconda Installation

Install anaconda manually from the website: [Anaconda install link](https://www.anaconda.com/download/success)

After installing with ".exe" file open Anaconda Prompt

- Create environment

```
conda create -n [your env name] python=3.8.20
```

- Activate conda environment

```
conda activate [your env na]
```


- CUDA & cuDNN installation 

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Following this code will simply install cuda version 11.8 and also corresponding cuDNN.



- Check Install - with python code

```python
import torch
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("cuDNN Version:", torch.backends.cudnn.version())
```

This will print out the version installed.



## Additional Libraries

It is installed individually for the models requirement and GUI framework by "pip install"

- U2Net model requirements

```
pip install numpy==1.15.2
pip install scikit-image==0.14.0
pip install torch
pip install torchvision
pip install pillow==8.1.1
pip install opencv-python
pip install paddlepaddle
pip install paddlehub
pip install gradio
```

- Tkinter GUI framework requirements

```
pip install certifi==2021.5.30
pip install chardet==4.0.0
pip install idna==2.10
pip install jinja2==3.0.1
pip install markupsafe==2.0.1
pip install Pillow==10.1
pip install requests==2.25.1
pip install urllib3==1.26.6
```

If there is any requested libraries aside from the above, simply download with "pip install" or "conda install".



## 3. Dataset Installation

For synthetic SMPL dataset it's link was included in the thesis "Automatic Estimation of Anthropometric Human Body Measurements".
[SMPL Dataset Link](https://skeletex.xyz/portfolio/datasets)

For additional information who wants to know about BodyM dataset
It is a real human dataset with 14 values of body size for each pair of front and left-side image.
The data is on AWS, so to use it you must follow the instuction bellow.

**AWS CLI Installation and S3 Access Instructions**

1. **Install AWS CLI**
   Follow the official guide to install AWS CLI from the link:
   [Install AWS CLI on Windows](https://docs.aws.amazon.com/cli/v1/userguide/install-windows.html)
   This enables access to AWS services.
   Verify the installation using the command:

   ```
   aws --version
   ```

2. **Open Command Prompt**
   Open the command prompt or terminal window to proceed with AWS commands.

3. **View S3 Bucket Contents**
   To list the contents of the S3 bucket, use the following command:

   ```
   aws s3 ls --no-sign-request s3://amazon-bodym/
   ```

4. **Download Files from S3**

   Download file
   
   ```
   aws s3 cp --no-sign-request s3://amazon-bodym ~/Downloads/destination_filename
   ```

By following these steps, you can list and download files from the specified S3 bucket.



## 4. "Conv_BoDiEs" Model Architecture Code

The architecture of "Conv_BoDiEs" the body size estiamtion model was very simple and consist with 4 constitutional layers followed by max pooling.

![image](https://github.com/user-attachments/assets/b9acc35a-9a69-4bc3-906d-022ad1f217e0)

- Python code for the architecture

```python
class Conv_BoDiEs(nn.Module):
    def __init__(self):
        super(Conv_BoDiEs, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()

        # Calculate the flattened size
        # Input size is (b, 32, 12, 12) after conv4 and maxpool
        flattened_size = 32 * 12 * 12

        self.fc1 = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(128, 16)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
```



## 5. U-2-Net Pre-trained Model 

Github : [Link](https://github.com/xuebinqin/U-2-Net)

The github repository for the U-2-Net does not gives pre-trained model cause it's size is big. 

Instead you can find it from huggingface : [Link](https://huggingface.co/spaces/hylee/u2net_portrait/tree/main/U-2-Net/saved_models)



