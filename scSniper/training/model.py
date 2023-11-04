from autoencoder import AE
from classifier import Classifier
from layers import *
class AE_Classifier(nn.Module):

    def __init__(self, encoder_dim_dict, decoder_dim_list, interlayers_dims, num_batch, num_class, batchnorm=True,
                 activation=nn.LeakyReLU(), dropout=0.1, dropout_classifier=0):
        super().__init__()
        self.num_batch = num_batch
        self.num_class = num_class

        self.AE = AE(encoder_dim_dict, decoder_dim_list, num_batch, batchnorm=batchnorm, activation=activation, dropout=dropout)
        self.Classifier = Classifier(interlayers_dims, self.AE.emb_dim, num_batch, num_class, batchnorm=batchnorm,
                                     activation=activation, dropout=dropout_classifier)
        self.loss_function = None

    def set_batch_onehot(self, batch_onehot):
        self.batch_onehot = batch_onehot

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def forward(self, x, *args, train_what="all_return_prob"):
        if train_what == "all":
            x_re, emb = self.AE(x, self.batch_onehot)
            output = self.Classifier(emb, self.batch_onehot)
            return x_re, emb, output
        elif train_what == "AE":
            x_re, emb = self.AE(x, self.batch_onehot)
            return x_re, emb
        elif train_what == "Classifier":
            output = self.Classifier(x, self.batch_onehot)
            return output
        elif train_what == "all_return_prob":
            if self.loss_function is None:
                if isinstance(x, tuple):
                    batch_onehot = self.batch_onehot.repeat(x[0].shape[0] // self.batch_onehot.shape[0], 1)
                    x_re, emb = self.AE(x, batch_onehot)
                elif args is not None:
                    args = list(args)
                    if isinstance(x, tuple):
                        x = list(x) + args
                    else:
                        x = [x] + args
                    batch_onehot = self.batch_onehot.repeat(x[0].shape[0] // self.batch_onehot.shape[0], 1)
                    x_re, emb = self.AE(x, batch_onehot)
                else:
                    x = [x]
                    batch_onehot = self.batch_onehot.repeat(x[0].shape[0] // self.batch_onehot.shape[0], 1)
                    x_re, emb = self.AE(x, batch_onehot)
                output = self.Classifier(emb, batch_onehot)
                return output
            else:
                args = list(args)
                ground_truth_patient_cat = args.pop()
                batch_onehot = args.pop()
                if isinstance(x, tuple):
                    x = list(x) + args
                else:
                    x = [x] + args
                x = self.set_uniform_device(*x)
                x = list(x)
                x[0], batch_onehot, ground_truth_patient_cat = self.set_uniform_device(x[0], batch_onehot,
                                                                                       ground_truth_patient_cat)
                x_re, emb = self.AE(x, batch_onehot)

                output = self.Classifier(emb, batch_onehot)
                loss = self.loss_function[0](output, ground_truth_patient_cat)
                return loss

    def set_uniform_device(self, *arg):
        arg = list(arg)
        # how to know if a tensor is on cuda or not?

        device = "cuda" if sum([i.is_cuda for i in arg]) > 0 else "cpu"
        for i in range(len(arg)):
            arg[i] = arg[i].to(device)
        return tuple(arg)

    def enable_Classifier_grad(self):
        self.AE.requires_grad_(False)
        self.AE.to("cpu")
        self.Classifier.requires_grad_(True)
        self.Classifier.to("cuda") if torch.cuda.is_available() else self.Classifier.to("cpu")

    def disable_regularizer_grad(self):
        for i in range(len(self.AE.first_layer)):
            self.AE.first_layer[i].regularzation.requires_grad_(False)
            # self.AE.first_layer[i].regularzation.to("cpu")

    def enable_AE_grad(self):
        self.AE.requires_grad_(True)
        self.AE.to("cuda") if torch.cuda.is_available() else self.AE.to("cpu")
        self.Classifier.requires_grad_(False)
        self.Classifier.to("cpu")

    def enable_all_grad(self):
        self.AE.requires_grad_(True)
        self.AE.to("cuda") if torch.cuda.is_available() else self.AE.to("cpu")
        self.Classifier.requires_grad_(True)
        self.Classifier.to("cuda") if torch.cuda.is_available() else self.Classifier.to("cpu")

    def enable_none_grad(self):
        self.AE.requires_grad_(False)
        self.AE.to("cuda") if torch.cuda.is_available() else self.AE.to("cpu")
        self.Classifier.requires_grad_(False)
        self.Classifier.to("cuda") if torch.cuda.is_available() else self.Classifier.to("cpu")
