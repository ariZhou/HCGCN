import torch as th
import torch.functional
from torch.nn import Module


class STSGCN(Module):
    def __init__(self, adj,
                 t, num_of_vertices, num_of_features, predict_length,
                 filter_list, module_type, activation,
                 use_mask=True, temporal=True, spatial=True, window_size=3, embedding_size=None, hidden_size=128,
                 rho=1):
        """
        Parameters
        ----------
        adj:tensor 3N*3N
        t: int T（历史时间步），由于延迟天数最长为3,不超过5,所以选择为5
        num_of_vertices:N
        num_of_features:C (原本的特征维度)
        predict_length:T'（未来的时间步）
        filter_list:
        module_type:str
        activation:str
        use_mask:bool
        temporal:bool
        spatial:bool
        window_size:int  最长延迟天数 3
        embedding_size:int C'(嵌入的维度)
        rho:float
        """
        super(STSGCN, self).__init__()
        self.adj = adj
        self.T = t
        self.N = num_of_vertices
        self.feature = num_of_features
        self.predict_T = predict_length
        self.filters_list = filter_list
        self.modules_type = module_type
        self.activation = None
        self.use_mask = use_mask
        self.temporal = temporal
        self.spatial = spatial
        self.mask = None
        self.hidden = hidden_size
        if embedding_size:
            self.embedding_transformer = th.nn.Linear(self.feature, embedding_size)
            self.feature = embedding_size
        self.window_size = window_size  # 暂时先别用参数，先直接用3
        assert activation in {"GLU", "relu","sigmoid","relu_","sigmoid_","GLU_","none","none_"}
        self.activation = activation
        if self.use_mask == True:
            mask = th.zeros(adj.shape)
            mask = mask.masked_fill(adj.cpu().bool(), 1)
            self.mask = th.nn.Parameter(mask)
        layers = []
        input_length = self.T

        assert self.modules_type in {'sharing', 'individual'}
        for idx, filters in enumerate(self.filters_list):
            if module_type == "individual":
                layers.append(
                    StsIndividualLayer(input_length, self.N, self.feature, filters, self.activation, self.temporal,
                                       self.spatial,window_size=self.window_size))
            elif module_type == "sharing":
                layers.append(
                    StsSharingLayer(input_length, self.N, self.feature, filters, self.activation, self.temporal,
                                    self.spatial,))
            input_length = input_length - self.window_size + 1
        self.layers = th.nn.ModuleList(layers)
        outs = []
        for t in range(predict_length):
            outs.append(outputLayer(self.feature * input_length, 1, self.hidden))
        self.outs = th.nn.ModuleList(outs)

    def forward(self, data):
        """

        Parameters
        ----------
        data:(B,T,N,C) 其中C表示特征数量

        Returns
        -------

        """
        B = data.shape[0]
        if self.embedding_transformer:
            data = self.embedding_transformer(data)
        if self.use_mask:
            adj = self.adj * self.mask
        else:
            adj = self.adj
        result = data
        for layer in self.layers:
            result = layer(result, adj)
        # result shape is (B, N, T=1, C')
        result = th.transpose(result, 1, 2)
        # result shape is (B, N, 1*C')
        #result = th.reshape(result, (-1, self.N, self*self.feature))
        result = th.reshape(result, (B, self.N, -1))
        need_concat = []

        for out in self.outs: #对应不同的时间步
            need_concat.append(th.unsqueeze(out(result), dim=-1))

        return th.unsqueeze(th.cat(need_concat, dim=1)[:, :, -1], dim=2)


class StsIndividualLayer(Module):
    def __init__(self, t, vertices_num, features_num, filters, activation, temporal=True, spatial=True, window_size=3):
        super(StsIndividualLayer, self).__init__()
        self.T = t
        self.N = vertices_num
        self.C = features_num
        self.filters = filters
        self.activation = activation
        self.temporal_emb = None
        self.spatial_emb = None
        self.window_size = window_size
        if temporal:
            # (1,T,1,C)
            emb = torch.empty(1, self.T, 1, features_num)
            emb = th.nn.init.xavier_normal_(emb, gain=1)
            self.temporal_emb = th.nn.Parameter(emb)
        if spatial:
            # shape is (1, 1, N, C)
            emb = torch.empty(1, 1, vertices_num, features_num)
            emb = th.nn.init.xavier_normal_(emb, gain=1)
            self.spatial_emb = th.nn.Parameter(emb)

        gcms = []
        for i in range(self.T - self.window_size + 1):
            gcms.append(StsGcm(filters, self.C, self.N, activation,window_size))
        self.gcms = th.nn.ModuleList(gcms)

    def forward(self, data, adj):
        """

        Parameters
        ----------
        data:(B, T, N, C)
        adj: 3N,3N
        Returns (B,T-2,N)
        -------

        """
        need_concat = []
        if self.temporal_emb is not None:
            data = data + self.temporal_emb
        if self.spatial_emb is not None:
            data = data + self.spatial_emb
        for i in range(self.T - self.window_size + 1):
            # shape is (B, 3, N, C)
            t = data[:, i:i + self.window_size]
            # shape is (B, 3N, C)
            t = th.reshape(t, (-1, self.window_size * self.N, self.C))
            # shape is (3N, B, C)
            t = th.transpose(t, 1, 0)
            # shape is (N, B, C')
            result = self.gcms[i](t, adj)
            # shape is (B, N, C')
            t = th.swapaxes(result.values, 0, 1)
            #t = th.swapaxes(result, 0, 1) #when  torch.mean() is used, use the code at this line
            # shape is (B, 1, N, C')
            need_concat.append(th.unsqueeze(t, dim=1))
        # shape is (B, T-2, N, C')
        concat = th.cat(need_concat, dim=1)
        return concat


class StsSharingLayer(Module):
    def __init__(self, t, vertices_num, features_num, filters, activation, temporal=True, spatial=True,window_size =3):
        super(StsSharingLayer, self).__init__()
        self.T = t
        self.N = vertices_num
        self.C = features_num
        self.filters = filters
        self.activation = activation
        self.temporal_emb = None
        self.spatial_emb = None
        if temporal:
            # (1,T,1,C)
            emb = torch.empty(1, self.T, 1, features_num)
            emb = th.nn.init.xavier_normal(emb, gain=1)
            self.temporal_emb = th.nn.Parameter(emb)
        if spatial:
            # shape is (1, 1, N, C)
            emb = torch.empty(1, 1, vertices_num, features_num)
            emb = th.nn.init.xavier_normal(emb, gain=1)
            self.spatial_emb = th.nn.Parameter(emb)
        self.gcm = StsGcm(filters, self.C, self.N, self.activation, window_size)

    def forward(self, data, adj):
        """

        Parameters
        ----------
        data:(B, T, N, C)

        Returns (B, T-2, N, C')
        -------

        """
        need_concat = []
        # shape is (B, T, N, C)

        if self.temporal_emb is not None:
            data = data + self.temporal_emb
        if self.spatial_emb is not None:
            data = data + self.spatial_emb

        for i in range(self.T - 2):
            # shape is (B, 3, N, C)
            t = data[:,i:i + 3]
            # shape is (B, 3N, C)
            t = th.reshape(t, (-1, 3 * self.N, self.C))

            # shape is (3N, B, C)
            t = th.swapaxes(t, 0, 1)
            need_concat.append(t)
        # shape is (3N, (T-2)*B, C)
        t = th.cat(need_concat, dim=1)
        # shape is (N, (T-2)*B, C')
        t = self.gcm(t,adj)

        # shape is (N, T - 2, B, C)
        t = th.reshape(t.values, (self.N, self.T - 2, -1, self.C))
        # shape is (B, T - 2, N, C)
        return th.swapaxes(t, 0, 2)


class StsGcm(Module):
    def __init__(self,filters, feature_num, vertices_num, activation, window_size):
        """
        Parameters
        ----------
        filters:list[int], list of C'
        feature_num: int,C
        vertices_num:int,N
        activation:str
        """
        super(StsGcm, self).__init__()
        self.N = vertices_num
        self.C = feature_num
        self.window_size = window_size
        layers = [Gcn(vertices_num=self.N, filter_features=feature_num, features_num=feature_num, activation=activation)
                  for i in range(len(filters))]
        self.gcns = th.nn.ModuleList(layers)

    def forward(self, data, adj):
        """
        Parameters
        ----------
        data   shape is (3N, B, C)

        Returns (N, B, C')
        -------

        """
        need_concat = []
        for id, gcn in enumerate(self.gcns):
            # (3N,B,C')
            result = gcn(data, adj)
            # (N,B,C')
            midstep = result[((self.window_size - 1) // 2) * self.N: (((self.window_size + 1) // 2)) * self.N]   #切割
            # (1,N,B,C')
            need_concat.append(th.unsqueeze(midstep, 0))
        # max_buf = th.min(th.cat(need_concat, dim=0), dim=0)
        max_buf = th.max(th.cat(need_concat, dim=0), dim=0)
        # max_buf = th.mean(th.cat(need_concat, dim=0), dim=0)
        return max_buf


class Gcn(Module):
    def __init__(self, filter_features, features_num, vertices_num, activation):
        super(Gcn, self).__init__()
        """
        Parameters
        ----------
        features_filter: int   一个filter的特征数 C'
        features_num:特征数,C
        vertices_num:顶点数
        """
        self.filter_features_num = filter_features
        self.features = features_num
        self.N = vertices_num
        self.activation = activation
        if activation == "GLU":
            self.linear = th.nn.Linear(self.features, 2 * self.filter_features_num)
        elif activation == "relu":
            self.linear = th.nn.Linear(self.features, self.filter_features_num)
        elif activation == "sigmoid":
            self.linear = th.nn.Linear(self.features, self.filter_features_num)
        elif activation == "none":
            self.linear = th.nn.Linear(self.features, self.filter_features_num)

    def forward(self, data, adj):
        """

        Parameters
        ----------
        data: (3N,B,C)
        adj: (3N,3N)
        activation: str
        Returns  (3N,B,C')
        -------
        """
        """
            添加的代码
        """
        # (B, 3N, C)
        data = th.transpose(data, 0, 1)
        # (B, 3N, 3N)
        adj = th.unsqueeze(adj, 0)
        batch =  data.shape[0]
        adj = adj.repeat(batch,1,1)
        """
            添加的代码
        """
        data = th.bmm(adj, data)
        data = th.transpose(data, 0, 1)
        if self.activation == "GLU":
            data = self.linear(data)
            lhs, rhs = th.chunk(data, chunks=2, dim=2)
            return lhs * torch.sigmoid(rhs)
        elif self.activation == "relu":
            data = self.linear(data)
            return torch.relu(data)
        elif self.activation == "sigmoid":
            data = self.linear(data)
            return torch.sigmoid(data)
        elif self.activation == "relu_":
            return torch.relu(data)
        elif self.activation == "GLU_":
            return data * torch.sigmoid(data)
        elif self.activation == "sigmoid_":
            return torch.sigmoid(data)
        elif self.activation == "none":
            return self.linear(data)
        elif self.activation == "none_":
            return data

class outputLayer(Module):
    def __init__(self, features_input, features_output, hidden_size):
        super(outputLayer, self).__init__()
        self.liner1 = th.nn.Linear(features_input, hidden_size)
        self.liner2 = th.nn.Linear(hidden_size, features_output)

    def forward(self, data):
        '''

        Parameters
        ----------
        data B*N*(1*C')

        Returns B*1*N
        -------

        '''
        result = self.liner1(data)
        result = th.functional.F.relu(result)
        result = self.liner2(result)
        return th.transpose(result, 1, 2)
