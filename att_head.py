
'''
输入是（B，N，D），B是batch size，N是节点数，D是每个节点的原始特征维数
输出是（B，N，F），F是每个节点的新特征维数
每个节点从维度D到维度F是按注意力为权重聚合了邻居节点的特征
'''

def att_head(seq, out_sz, bias_mat, activation, in_drop = 0.0, coef_drop = 0.0, residual = False):
    '''
    seq：输入（B，N，D），B是batch size，N是节点数，D是每个节点的原始特征维数
    out_sz：每个节点的输出特征维数，设为F
    bias_mat：（N，N）掩码矩阵
    activation：激活函数
    in_drop：输入的dropout率
    coef_drop：注意力矩阵的dropout率
    residual：是否使用残差网络
    '''
    
    with tf.name_scope('my_attn'):
        # drop out 防止过拟合;如果为0则不设置该层
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        
        '''
        为了获得足够的表达能力以将输入特征转化为高级特征，需要至少一种可学习的线性变换。为此，作为第一步，
        我们学习一个W矩阵用于投影特征
        实现公式seq_fts = Wh，即每个节点的维度变换
        '''
        
        # F2F'
        seq_fts = tf.keras.layers.Conv1D(seq, out_sz, 1, use_bias=False)

        '''
        实现公式 f_1 = a(Whi); f_2 = a(Whj)
        f_1+f_2的转置实现了logits = eij = a(Whi) + a(Whj)
        eij经过激活,softmax得到论文中的aij,即点i对点j的注意力
        bias_mat是为了让非互为邻居的注意力不要j进入softmax的计算
        只有互为邻居的注意力才能进入softmax,从而保证了注意力在局部
        '''
        
        # (B, N, F) => (B, N, 1)
        f_1 = tf.keras.layers.Conv1D(seq_fts, 1, 1)
        # (B, N, F) => (B, N, 1)
        f_2 = tf.keras.layers.Conv1D (seq_fts, 1, 1)
        
        # (B, N, 1) + (B, N, 1) = (B, N, N)
        # logits 即 eij
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        # (B, N, N) + (1, N, N) => (B, N, N) => softmax => (B, N, N)
        # 这里运用了 tensorflow 的广播机制
        # 得到的logits 并不是一个对角矩阵, 这是因为 f_1 和 f_2并非同一个参数 a
        # logits{i,j} 等于 a1(Whi) + a2(Whj)
        
        # 注意力系数矩阵coefs=(aij)_{N*N}
        # bias_mat 体现 mask 思想, 保留了图的结构信息, 

        
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)
        
        # 输入矩阵、注意力系数矩阵的dropout操作
        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)
            
        '''
        实现 hi = sum(aijWhj)
        即节点i根据注意力聚合邻居特征
        '''
        
        # (B, N, N) * (B, N, F) => (B, N, F)
        vals = tf.matmul(coefs, seq_fts)
        
        
        
        # 添加偏置项
        ret = tf.contrib.layers.bias_add(vals)
        
        '''
        添加残差连接后,激活
        如果输入(B, N, D)和聚合了节点特征的输出(B, N, F)的最后一个维度相同,则直接相加
        否则将(B, N, D)线性变换为(B, N, F) 再相加
        '''
        
        # residual connection
        if residual:
            # D != F
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq
        
        return activation(ret) # activation

