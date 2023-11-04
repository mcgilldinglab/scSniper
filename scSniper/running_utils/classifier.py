from .layers import *
class Classifier(nn.Module):
    def __init__(self, interlayers_dims, emb_dim, num_batch, num_class, apply_softmax=True, batchnorm=True,
                 activation=nn.LeakyReLU(), dropout=0):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_batch = num_batch
        self.num_class = num_class
        self.classifier = nn.ModuleDict()

        self.num_batch = 0

        self.classifier["emb"] = Linear_Layers_Classifier(emb_dim, interlayers_dims[0], num_batch)
        for i in range(len(interlayers_dims) - 1):
            self.classifier[str(interlayers_dims[i]) + "->" + str(interlayers_dims[i + 1])] = Linear_Layers(
                interlayers_dims[i], interlayers_dims[i + 1], num_batch, batchnorm=batchnorm, activation=activation,
                dropout=dropout)
        if apply_softmax:
            self.output = Linear_Layers_Classifier(interlayers_dims[-1], num_class, num_batch,
                                                   activation=nn.Softmax(dim=1), dropout=0, batchnorm=False)
        else:
            self.output = Linear_Layers_Classifier(interlayers_dims[-1], num_class, num_batch, activation=nn.Identity(),
                                                   dropout=0, batchnorm=False)

    def forward(self, emb, batch_onehot=None):
        output = emb
        for key in self.classifier.keys():
            output = self.classifier[key](output, batch_onehot)
        output = self.output(output, batch_onehot)
        return output