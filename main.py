#------------------------------------------------------------
#   coding:utf-8
#------------------------------------------------------------
#   Updata History
#   December  02  05:00, 2019 (Mon)
#------------------------------------------------------------

#  default
import sys

#  local
from visualize_data import *
from utils import *


"""
    Raw data distribution
"""
def raw_data_distribution():
    raw_x = np.array([a.data.numpy() for a in data.valid_ds.x])
    raw_x = raw_x.reshape((len(raw_x), -1))
    raw_y = np.array([int(y.obj) for y in data.valid_ds.y])

    if False:
        LIMIT = 1000
        chosen_idxes = np.random.choice(list(range(len(raw_x))), LIMIT)
        raw_x = raw_x[chosen_idxes]
        raw_y = raw_y[chosen_idxes]

    show_2D_tSNE(raw_x, raw_y, "Raw sample distributions (t-SNE)")


"""
    Conventional softmax model
"""
def conventional_softmax_model():
    def learner_conventional(train_data):
        learn = cnn_learner(train_data, models.resnet18, metrics=accuracy)
        learn.fit(1)
        learn.unfreeze()
        learn.fit(3)
        return learn

    learn = learner_conventional(data)
    embs = get_embeddings(body_feature_model(learn.model), data.valid_dl)
    show_2D_tSNE(embs, [int(y) for y in data.valid_ds.y], title="Simply trained ResNet18 (t-SNE)")


"""
    L2-constrained Softmax Loss
"""
class L2ConstraintedNet(nn.Module):
    def __init__(self, org_model, alpha=16, num_classes=1000):
        super().__init__()
        self.org_model = org_model
        self.alpha = alpha

    def forward(self, x):
        x = self.org_model(x)

        #  L2 softmax function部分
        l2 = torch.sqrt((x**2).sum())
        x = self.alpha * (x / l2)
        return x

def l2_constrained_softmax_model():
    def learner_L2ConstraintedNet(train_data):
        learn = cnn_learner(train_data, models.resnet18, metrics=accuracy)
        learn.model = L2ConstraintedNet(learn.model)
        learn.fit(1)
        learn.unfreeze()
        learn.fit(5)
        return learn

    learn = learner_L2ConstraintedNet(data)
    embs = get_embeddings(body_feature_model(learn.model), data.valid_dl)
    show_2D_tSNE(embs, [int(y) for y in data.valid_ds.y], title="L2 contrainted ResNet18 (t-SNE)")


if __name__ == "__main__":
    args = sys.argv

    DATA = Path("data")
    data = prepare_full_MNIST_databunch(DATA, get_transforms(do_flip=False))

    if args[1] == 1:
        raw_data_distribution()
    elif args[1] == 2:
        conventional_softmax_model()
    elif args[1] == 3:
        l2_constrained_softmax_model()
    else:
        pass