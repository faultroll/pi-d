1. Download the VGG16 model from https://download.pytorch.org/models/vgg16-397923af.pth
1. Put vgg16-397923af.pth in pretrained dir
2. Download the RESIDE SOTS from https://sites.google.com/view/reside-dehaze-datasets/reside-standard
2. Uncompress the SOTS in same dir as train.py
2. Run dataset_reside.py to generate split_txt files
3. Run train.py to get ONNX model:
     python train.py
4. Run predict.py to see results:
     python predict.py
