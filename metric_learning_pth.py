#------------------------------------------------------------
#   coding:utf-8
#------------------------------------------------------------
#   Updata History
#   November  25  21:00, 2019 (Mon)
#------------------------------------------------------------

from fastai.vision import *
from torchvision import datasets, transforms
from torch import nn
import PIL
from tqdm import tqdm
from dlcliche.image import *

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import axes3d
#%matplotlib notebook

def prepare_full_MNIST_databunch(data_folder, tfms):
    train_ds = datasets.MNIST(data_folder, train=True, download=True,
                            transform=transforms.Compose([
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
    valid_ds = datasets.MNIST(data_folder, train=False,
                            transform=transforms.Compose([
                                transforms.Normalize((0.1307,), (0.3801,))
                            ]))

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

"""
    描画
"""
def show_2D_tSNE(latent_vecs, target, title="t-SNE viz"):
    latent_vecs = latent_vecs
    latent_vecs_reduced = TSNE(n_components=2, random_state=0).fit_transform(latent_vecs)
    plt.scatter(latent_vecs_reduced[:, 0], latent_vecs_reduced[:, 1],
                c=target, cmap="jet")
    plt.colorbar()
    plt.show()

def show_3D_tSNE(latent_vecs, target, title="3D t-SNE viz"):
    latent_vecs = latent_vecs
    tsne = TSNE(n_comonents=3, random_state=0).fit_transform(latent_vecs)
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter3D(tsne[:, 0], tsne[:, 1], tsne[:, 2],
                        c=target, cmap="jet")
    ax.set_title(title)
    plt.colorbar(scatter)
    plt.show()

def show_as_PCA(latent_vecs, target, title="PCA viz"):
    latent_vecs = latent_vecs
    latent_vecs_reduced = PCA(n_components=2).fit_transform(latent_vecs)
    plt.scatter(latent_vecs_reduced[:, 0], latent_vecs_reduced[:, 1],
                c=target, cmap="jet")
    plt.colorbar()
    plt.show()

"""

"""
DATA = Path("data")
data = prepare_full_MNIST_databunch(DATA, get_transforms(do_flip=False))

"""

"""
raw_x = np.array([a.data.numpy() for a in data.valid_ds.x])
raw_x = raw_x.reshape((len(raw_x), -1))
raw_y = np.array([int(y.obj) for y in data.valid_ds.y])

if False:
    LIMIT = 1000
    chosen_idxes = np.random.choice(list(range(len(raw_x))), LIMIT)
    raw_x = raw_x[chosen_idxes]
    raw_y = raw_y[chosen_idxes]

show_2D_tSNE(raw_x, raw_y, "Raw sample distributions (t-SNE)")