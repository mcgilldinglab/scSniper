from .layers import *
class AE(nn.Module):
    def __init__(self, encoder_dim_dict, decoder_dim_list, num_batch, batchnorm=True,
                 activation=nn.LeakyReLU(), dropout=0.1):
        super().__init__()
        self.num_batch = num_batch

        self.first_layer = nn.ModuleList()

        self.emb_dim = 0
        # encoder for each modality
        self.encoder = nn.ModuleDict()

        for modality in encoder_dim_dict.keys():
            self.encoder[modality] = nn.ModuleDict()
            out_feature = 0
            for i in range(len(encoder_dim_dict[modality]) - 1):
                if i == 0:
                    self.encoder[modality][str(encoder_dim_dict[modality][i]) + "->" + str(
                        encoder_dim_dict[modality][i + 1])] = Linear_Layers(encoder_dim_dict[modality][i],
                                                                            encoder_dim_dict[modality][i + 1],
                                                                            num_batch, bias=True, batchnorm=batchnorm,
                                                                            activation=activation, dropout=dropout)
                    self.first_layer.append(self.encoder[modality][str(encoder_dim_dict[modality][i]) + "->" + str(
                        encoder_dim_dict[modality][i + 1])])
                else:
                    self.encoder[modality][str(encoder_dim_dict[modality][i]) + "->" + str(
                        encoder_dim_dict[modality][i + 1])] = Linear_Layers(encoder_dim_dict[modality][i],
                                                                            encoder_dim_dict[modality][i + 1],
                                                                            num_batch, batchnorm=batchnorm,
                                                                            activation=activation, dropout=dropout)

                out_feature = self.encoder[modality][
                    str(encoder_dim_dict[modality][i]) + "->" + str(encoder_dim_dict[modality][i + 1])].outfeature
            self.emb_dim += out_feature

        # decoder for each modality
        self.decoder = nn.ModuleDict()
        for modality in decoder_dim_list.keys():
            self.decoder[modality] = nn.ModuleDict()
            self.decoder[modality]["emb"] = nn.Linear(self.emb_dim, decoder_dim_list[modality][0])
            for i in range(len(decoder_dim_list[modality]) - 1):
                if i != len(decoder_dim_list[modality]) - 2:
                    self.decoder[modality][str(decoder_dim_list[modality][i]) + "->" + str(
                        decoder_dim_list[modality][i + 1])] = Linear_Layers(decoder_dim_list[modality][i],
                                                                            decoder_dim_list[modality][i + 1],
                                                                            num_batch, batchnorm=batchnorm,
                                                                            activation=activation)
                else:
                    self.decoder[modality][str(decoder_dim_list[modality][i]) + "->" + str(
                        decoder_dim_list[modality][i + 1])] = Linear_Layers(decoder_dim_list[modality][i],
                                                                            decoder_dim_list[modality][i + 1],
                                                                            num_batch, batchnorm=batchnorm,
                                                                            activation=activation)  # ,activation=nn.Identity(),dropout=0)

        # self.attention = nn.MultiheadAttention(self.emb_dim, self.emb_dim, batch_first=True)
        self.attention = MultiheadAttentionWithMimeticInit(self.emb_dim, self.emb_dim, batch_first=True)

    def forward(self, x, batch_onehot):
        # encoder
        emb = []
        for j, modality in enumerate(self.encoder.keys()):
            input = x[j]
            for key in self.encoder[modality].keys():
                 input = self.encoder[modality][key](input, batch_onehot)
            emb.append(input)

        emb = torch.cat(emb, dim=1)
        # (batch_size,emb_dim*2)
        # (batch_size,emb_dim*2,1)

        # attention
        emb = emb.unsqueeze(1)
        emb, _ = self.attention(emb, emb, emb)
        emb = emb.squeeze(1)

        # decoder
        x_re = []
        for j, modality in enumerate(self.decoder.keys()):
            output = emb
            for key in self.decoder[modality].keys():
                output = self.decoder[modality][key](output, batch_onehot) if isinstance(self.decoder[modality][key],
                                                                                         Linear_Layers) else \
                self.decoder[modality][key](output)
            x_re.append(output)
        return x_re, emb