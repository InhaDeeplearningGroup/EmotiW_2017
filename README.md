# EmotiW_2017

- Multi-modal Emotion Recognition using Semi-supervised Learning and Multiple Neural Networks in the Wild
- The Website is here [[EmotiW_2017]](https://sites.google.com/site/emotiwchallenge/)
- Contact: kdhht5022@gmail.com

### News
- Prepare dataset codes are uploaded (25 Nov, 2017).
- Image-based network's codes are uploaded (25 Nov, 2017).

### Citation
```
  @InProceedings{kimICMI2017,
  Author={Dae Ha Kim, Min Kyu Lee, Dong Yoon Choi, Byung Cheol Song},
  Title={Multi-modal Emotion Recognition using Semi-supervised Learning and Multiple neural Networks},
  Booktitle={19th ACM International Conference on Multimodal Interaction},  
  Year = {2017}
  }
```

### Environment
- This code is tested on Linux os (ubuntu 16.04) 64bit, Python 3.5, and Cuda-8.0 (cudnn-5.1) with NVIDIA GTX 1080 TI.

## Usage

**Step 1.
Install [Keras>=2.0.6](https://github.com/fchollet/keras) 
with [TensorFlow>=1.2.0](https://github.com/tensorflow/tensorflow) backend.**
```
pip install tensorflow-gpu==1.2.0
pip install keras==2.0.6
```

**Step 2. Clone this repository to local.**
```
git clone https://github.com/InhaDeeplearningGroup/EmotiW_2017.git
cd EmotiW_2017
```

**Step 3. Prepare dataset.**

We use AFEW train & validation dataset that use EmotiW challenge.

You can get AFEW dataset after the administrator's approval in EmotiW website.

Next you have to make emotion folders such as Angry, Disgust, etc.

Adjust the path of the emotion files in the `prepare_dataset.py` and save `.npz` format.

**Step 4. Train networks.**

This is only Image-based networks.

You have to adjust the path of the `.npz` files in the `c3da.py` and `s3dae.py`

And simply run :)


