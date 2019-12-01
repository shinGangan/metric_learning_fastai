#------------------------------------------------------------
#   coding:utf-8
#------------------------------------------------------------
#   Updata History
#   December  02  03:00, 2019 (Mon)
#------------------------------------------------------------

from fastai.vision import *
from torchvision import datasets, transforms
from torch import nn
import PIL
from tqdm import tqdm
#from dlcliche.image import *
import os

def prepare_full_MNIST_databunch(data_folder, tfms):
    train_ds = datasets.MNIST(data_folder, train=True, download=True,
                            transform=transforms.Compose([
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
    valid_ds = datasets.MNIST(data_folder, train=False,
                            transform=transforms.Compose([
                                transforms.Normalize((0.1307,), (0.3801,))
                            ]))

    def ensure_folder(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
    def have_already_been_done():
        return (data_folder/"images").is_dir()
    def build_images_folder(data_root, X, labels, dest_folder):
        images = data_folder/"images"
        for i, (x, y) in tqdm.tqdm(enumerate(zip(X, labels))):
            folder = images/dest_folder/f"{y}"
            ensure_folder(folder)
            x = x.numpy()
            image = np.stack([x for ch in range(3)], axis=-1)
            PIL.Image.fromarray(image).save(folder/f"img{y}_{i:06d}.png")

    if not have_already_been_done():
        build_images_folder(data_root=DATA, X=train_ds.train_data,
                            labels=train_ds.train_labels, dest_folder="train")
        build_images_folder(data_root=DATA, X=valid_ds.test_data,
                            labels=valid_ds.test_labels, dest_folder="valid")

    return ImageDataBunch.from_folder(data_folder/"images", ds_tfms=tfms)


def body_feature_model(model):
    try:
        body, head = list(model.org_model.children())
    except:
        body, head = list(model.children())
    return nn.Sequential(body, head[:-1])


def get_embeddings(embedding_model, data_loader, label_catcher=None, return_y=False):
    embs, ys = [], []
    for X, y in data_loader:
        if label_catcher:
            label_catcher.on_barch_begin(X, y, train=False)
        with torch.no_grad():
            out = embedding_model(X).cpu().detach().numpy()
            out = out.reshape((len(out), -1))
            embs.append(out)
        ys.append(y)
    
    embs = np.concatenate(embs)
    ys = np.concatenate(ys)
    if return_y:
        return embs, ys
    return embs
