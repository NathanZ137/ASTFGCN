import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GatedInception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(GatedInception, self).__init__()
        # Path1
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.bn1_1 = nn.BatchNorm2d(c1)
        # Path2
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.bn2_1 = nn.BatchNorm2d(c2[0])
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=(1, 3), padding=(0, 1))
        self.bn2_2 = nn.BatchNorm2d(c2[1])
        self.p2_3 = nn.Conv2d(c2[1], c2[2], kernel_size=(3, 1), padding=(1, 0))
        self.bn2_3 = nn.BatchNorm2d(c2[2])
        # Path3
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.bn3_1 = nn.BatchNorm2d(c3[0])
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=(1, 3), padding=(0, 1))
        self.bn3_2 = nn.BatchNorm2d(c3[1])
        self.p3_3 = nn.Conv2d(c3[1], c3[2], kernel_size=(3, 1), padding=(1, 0))
        self.bn3_3 = nn.BatchNorm2d(c3[2])
        self.p3_4 = nn.Conv2d(c3[2], c3[3], kernel_size=(1, 3), padding=(0, 1))
        self.bn3_4 = nn.BatchNorm2d(c3[3])
        self.p3_5 = nn.Conv2d(c3[3], c3[4], kernel_size=(3, 1), padding=(1, 0))
        self.bn3_5 = nn.BatchNorm2d(c3[4])
        # Path4
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
        self.bn4_2 = nn.BatchNorm2d(c4)

    def forward(self, x):
        # p1
        p1 = F.relu(self.bn1_1(self.p1_1(x)), inplace=True)
        # p2
        p2_1 = F.relu(self.bn2_1(self.p2_1(x)), inplace=True)
        p2_2 = F.relu(self.bn2_2(self.p2_2(p2_1)), inplace=True)
        p2_3 = F.relu(self.bn2_3(self.p2_3(p2_2)), inplace=True)
        # p3
        p3_1 = F.relu(self.bn3_1(self.p3_1(x)), inplace=True)
        p3_2 = F.relu(self.bn3_2(self.p3_2(p3_1)), inplace=True)
        p3_3 = F.relu(self.bn3_3(self.p3_3(p3_2)), inplace=True)
        p3_4 = F.relu(self.bn3_4(self.p3_4(p3_3)), inplace=True)
        p3_5 = F.relu(self.bn3_5(self.p3_5(p3_4)), inplace=True)
        # p4
        p4_1 = self.p4_1(x)
        p4_2 = F.relu(self.bn4_2(self.p4_2(p4_1)), inplace=True)
        # Concat
        output = torch.cat((p1, p2_3, p3_5, p4_2), dim=1)

        return output


class Bi_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True, bidirectional=True):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional
        )
    
    def forward(self, x):

        output, _ = self.bilstm(x)
        
        return output
    

class GatedCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GatedCNN, self).__init__()
        
        self.conv3 = nn.Conv1d(input_dim, output_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(input_dim, output_dim, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(input_dim, output_dim, kernel_size=7, padding=3)

        self.final_conv = nn.Conv1d(3 * output_dim, output_dim, kernel_size=1)

        self.residual_conv = nn.Conv1d(input_dim, output_dim, kernel_size=1)

        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x):
        origin_x = x

        x3 = self.conv3(x.permute(0, 2, 1)).permute(0, 2, 1)
        x5 = self.conv5(x.permute(0, 2, 1)).permute(0, 2, 1)
        x7 = self.conv7(x.permute(0, 2, 1)).permute(0, 2, 1)

        concat = torch.cat((x3, x5, x7), dim=-1)
        concat = self.final_conv(concat.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.ln(concat + self.residual_conv(origin_x.permute(0, 2, 1)).permute(0, 2, 1))
        
        return x
    

class GatedFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GatedFusion, self).__init__()
        
        self.gated = nn.Sequential(
            nn.Conv1d(2 * input_dim, output_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):

        concat = torch.cat((input1, input2), dim=-1)
        gated = self.gated(concat.permute(0, 2, 1)).permute(0, 2, 1)
        output = torch.add(torch.mul(gated, input1), torch.mul((1 - gated), input2))
        
        return output
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len, dropout):
        super(PositionalEncoding, self).__init__()
        self.num_of_vertices = max_len
        self.embedding = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        pos = torch.LongTensor(torch.arange(self.num_of_vertices)).to(x.device)
        embed = self.embedding(pos).unsqueeze(0)
        x = x + embed
        x = self.dropout(x)
        return x


class SelfAttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads, attn_drop, mask=None):
        super(SelfAttentionLayer, self).__init__()

        assert model_dim % num_heads == 0

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Conv1d(model_dim, model_dim, kernel_size=1, bias=False)
        self.FC_K = nn.Conv1d(model_dim, model_dim, kernel_size=1, bias=False)
        self.FC_V = nn.Conv1d(model_dim, model_dim, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(attn_drop)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        B, N, H = x.shape

        query = self.FC_Q(x.permute(0, 2, 1)).permute(0, 2, 1)
        key = self.FC_K(x.permute(0, 2, 1)).permute(0, 2, 1)
        value = self.FC_V(x.permute(0, 2, 1)).permute(0, 2, 1)

        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
        key = key.transpose(-1, -2)
        attn_score = torch.matmul(query, key) / self.head_dim**0.5

        if self.mask is not None:
            attn_score = attn_score.masked_fill_(self.mask, -torch.inf)

        attn_score = torch.softmax(attn_score, dim=-1)
        attn_score = self.dropout(attn_score)
        out = torch.matmul(attn_score, value)
        out = torch.cat(torch.split(out, B, dim=0), dim=-1)

        out = self.out_proj(out)
        out = self.dropout(out)

        return out
    

class FFN(nn.Module):
    def __init__(self, input_features, hidden_features, output_features, dropout):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_features, hidden_features)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, output_features)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x
    

class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self):
        super(Spatial_Attention_layer, self).__init__()

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, T, N, N)
        '''
        batch_size, num_of_nodes, in_channels = x.shape

        score = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(in_channels)  # (b, N, F_in)(b, F_in, N)=(b, N, N)

        score = F.softmax(score, dim=-1)  # the sum of each row is 1; (b, N, N)

        return score
    

class ChebConv(nn.Module):

    def __init__(self, in_c, out_c, K, bias=True):
        """
        ChebNet conv
        :param in_c: input channels
        :param out_c:  output channels
        :param K: the order of Chebyshev Polynomial
        :param bias:  if use bias
        :param normalize:  if use norm
        """
        super(ChebConv, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))  # [K+1, 1, in_c, out_c]
        nn.init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.K = K + 1
        
        self.SAt = Spatial_Attention_layer()

    def forward(self, inputs, adj_matrix):
        """

        :param inputs: the input data, [B, N, C]
        :param graph: the graph structure, [N, N]
        :return: convolution result, [B, N, D]
        """
        batch_size, num_of_nodes, in_channels = inputs.shape
        spatial_attention = self.SAt(inputs) / math.sqrt(in_channels)  # scaled self attention: (batch, N, N)

        L = ChebConv.get_laplacian(adj_matrix)  # [N, N]
        mul_L = self.cheb_polynomial(L).unsqueeze(1)  # [K, 1, N, N]
        result = torch.matmul(mul_L.mul(spatial_attention), inputs)  # [K, B, N, C]
        result = torch.matmul(result, self.weight)  # [K, B, N, D]
        result = torch.sum(result, dim=0) + self.bias  # [B, N, D]

        return result

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian

        :param laplacian: the multi order Chebyshev laplacian, [K, N, N]
        :return:
        """
        N = laplacian.size(0)  # [N, N]
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)  # [K, N, N]
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k - 1]) - \
                                               multi_order_laplacian[k - 2]

        return multi_order_laplacian

    @staticmethod
    def get_laplacian(adj_matrix):
        """
        compute the laplacian of the graph
        :param graph: the graph structure without self loop, [N, N]
        :param normalize: whether to used the normalized laplacian
        :return:
        """
        assert adj_matrix.shape[0] == adj_matrix.shape[1]
        D = torch.diag(torch.sum(adj_matrix, dim=-1) ** (-1 / 2))
        L = torch.eye(adj_matrix.size(0), device=adj_matrix.device, dtype=adj_matrix.dtype) - torch.mm(torch.mm(D, adj_matrix), D)
        return L

class ChebNet(nn.Module):

    def __init__(self, in_c, hid_c, out_c, K):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param out_c: int, number of output channels.
        :param K:
        """
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_c=in_c, out_c=hid_c, K=K)
        self.conv2 = ChebConv(in_c=hid_c, out_c=out_c, K=K)
        self.act = nn.ReLU()

    def forward(self, x, adj_mx):
        graph_data = adj_mx  # [N, N]

        output1 = self.act(self.conv1(x, graph_data))
        output2 = self.act(self.conv2(output1, graph_data))
        return output2


class submodel(nn.Module):
    def __init__(self, in_c, hid_c, out_c, num_nodes, K, num_layers, c1, c2, c3, c4, num_heads, dropout):
        super(submodel, self).__init__()
        self.chebnet = ChebNet(in_c, hid_c, out_c, K)
        self.inception = GatedInception(in_c, c1, c2, c3, c4)
        self.bi_lstm = Bi_LSTM(2 * in_c, hid_c, num_layers)
        self.gated = GatedCNN(2 * in_c, 2 * hid_c)
        self.fusion = GatedFusion(2 * hid_c, 2 * hid_c)

        self.positional_encoding1 = PositionalEncoding(2 * in_c, num_nodes, dropout)
        self.positional_encoding2 = PositionalEncoding(2 * hid_c, num_nodes, dropout)

        self.selfattn1 = SelfAttentionLayer(2 * in_c, num_heads, dropout)
        self.selfattn2 = SelfAttentionLayer(2 * hid_c, num_heads, dropout)

        self.conv1 = nn.Conv1d(2 * c2[-1], out_c, kernel_size=1)
        self.conv2 = nn.Conv1d(out_c, 2 * hid_c, kernel_size=1)

        self.layer_norm1 = nn.LayerNorm(2 * in_c)
        self.layer_norm2 = nn.LayerNorm(2 * hid_c)

        self.fc = FFN(2 * hid_c, hid_c, out_c, dropout)

    def forward(self, x, adj_mx):
        B, N = x.size(0), x.size(1)
        x = x.view(B, N, -1) # [B, N, H*D]

        chebnet_output = self.chebnet(x, adj_mx) # [B, N, H]
        chebnet_output = F.relu(chebnet_output + x)

        inception_output = self.inception(x.permute(0, 2, 1).unsqueeze(2))
        inception_output = self.conv1(inception_output.squeeze(2)).permute(0, 2, 1)

        concat_output = torch.cat((chebnet_output, inception_output), dim=-1)
        concat_pe = self.positional_encoding1(concat_output)
        concat_attn = self.selfattn1(concat_pe)
        concat_attn = self.layer_norm1(concat_attn + concat_pe)

        bi_lstm_output = self.bi_lstm(concat_attn)
        residual = self.conv2(x.permute(0, 2, 1)).permute(0, 2, 1)
        lstm_output = self.layer_norm2(bi_lstm_output + residual)

        gated_output = self.gated(concat_attn)
        time_output = self.fusion(lstm_output, gated_output)

        positional_encoding = self.positional_encoding2(time_output)
        attn_output = self.selfattn2(positional_encoding)
        attn_output = self.layer_norm2(attn_output + positional_encoding)

        output = self.fc(attn_output) + x

        return output


class make_model(nn.Module):
    def __init__(self, in_c, hid_c, out_c, skip_c, num_nodes, K, num_layers, c1, c2, c3, c4, num_blocks,
                 num_heads, dropout):
        super(make_model, self).__init__()
        self.submodels = nn.ModuleList([submodel(in_c, hid_c, out_c, num_nodes, K, num_layers, c1, c2, c3, c4, num_heads, dropout) for i in range(num_blocks)])
        self.skip_convs = nn.ModuleList([nn.Conv1d(in_channels=in_c, out_channels=skip_c, kernel_size=1) for _ in range(num_blocks)])
        
        self.ln = nn.LayerNorm(skip_c)
        self.end_conv_1 = nn.Conv1d(in_channels=skip_c, out_channels=512, kernel_size=1)
        self.end_conv_2 = nn.Conv1d(in_channels=512, out_channels=out_c, kernel_size=1)

    def forward(self, x, adj_mx):

        skip = 0
        for i, block in enumerate(self.submodels):
            x = block(x, adj_mx)
            skip += self.skip_convs[i](x.permute(0, 2, 1))

        output = self.end_conv_1(self.ln(skip.permute(0, 2, 1)).permute(0, 2, 1))
        output = self.end_conv_2(F.relu(output))    

        return output.permute(0, 2, 1)
    