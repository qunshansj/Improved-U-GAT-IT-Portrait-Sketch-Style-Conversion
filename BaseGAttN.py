
class BaseGAttN:
    def loss(logits, labels, nb_classes, class_weights):
        sample_wts = tf.reduce_sum(tf.multiply(tf.one_hot(labels, nb_classes), class_weights), axis=-1)
        xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits), sample_wts)
        return tf.reduce_mean(xentropy, name='xentropy_mean')
    
    def training(loss, lr, l2_coef):
        # weight decay
        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not in ['bias', 'gamma', 'b', 'g', 'beta']] * l2_coef)
        
        # optimizer 
        opt = tf.train.AdamOptimizer(learning_rate = lr)
        
        # training op
        train_op = opt.minimize(loss + lossL2)
        
        return train_op
    
    
    def masked_softmax_cross_entropy(logits, labels, mask):
        '''
        Softmax cross-entropy loss with masking.
        logits: 模型的输出,维度(B, C); B是样本量, C是输出维度
        labels: 模型的标签,维度(B, C)
        mask: 掩码,维度(B, )
        '''
        
        # logits 先用softmax转化为概率分布,再和labelsj计算交叉熵
        # loss 维度是(B,)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
        
        # 将数据类型转化为 tf.float32
        mask = tf.cast(mask, dtype = tf.float32)
        
        # 将mask值归一化
        mask /= tf.reduce_mean(mask)
        
        # 屏蔽掉某些样本的损失
        loss *= mask
        
        # 返回均值损失
        return tf.reduce_mean(loss)
    
    
    def masked_sigmoid_cross_entropy(logits, labels, mask):
        '''
        Softmax cross-entropy loss with masking.
        logits:(B, C), 模型输出; B是样本量,C是输出维度
        labels:(B, C), 真实标签
        mask: 掩码,维度(B,)
        '''
        labels = tf.cast(mask, dtype = tf.float32)
        # loss 维度是(B,)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels)
        # (B,C) =>(B,)
        loss = tf.reduce_mean(loss, axis = 1)
        
        mask /= tf.reduce_mean(mask)
        loss *= mask
        
        return tf.reduce_mean(loss)
    
    def masked_accuracy(logits, labels, mask):
        '''
        Accuracy with masking
        logits:(B, C), 模型输出; B是样本量, C是输出维度
        labels:(B, C), 真实标签
        mask: 掩码,维度(B,)
        '''
        
        # 计算预测值和真实值的索引相同,则预测正确
        correct_prediction = tf.equal( tf.argmax(logits, 1), tf.argmax(labels, 1) )
        accuracy_all = tf.cast( correct_prediction, tf.float32 )
        mask = tf.cast( mask, dtype = tf.float32 )
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)
    
    
#%%
class GAT(BaseGAttN):
    
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop, bias_mat,
                  hid_mat, hid_units, n_heads, activation = tf.nn.elu, residual = False):
        '''
        inputs:(B,N,D), B是batch size, N是节点数, D是每个节点的原始特征维数
        nb_classes: 分类任务的类别数, 设为C
        nb_nodes: 节点个数,设为N
        training: 标志'训练阶段', '测试阶段'
        attn_drop: 注意力矩阵dropout率,防止过拟合
        ffd_drop: 输入的dropout率,防止过拟合
        bias_mat: 一个(N, N)矩阵,由邻接矩阵A变化而来,是注意力矩阵的掩码
        hid_units: 列表, 第i个元素是第i层的每个注意力头的隐藏单元数
        n_heads: 列表, 第i个元素是第i层的注意力头数
        activation: 激活函数
        resudial: 是否采用残差连接
        '''
        
        
        '''
        第一层,由H1个注意力头,每个头的输入都是(B, N, D), 每个头的注意力输出都是(B, N, F1)
        将所有注意力头的输出聚合, 聚合为(B, N, F1*H1)
        '''
        attns = []
        # n_heads[0] = 第一层注意力头数, 设为 H1
        for i in range(n_heads[0]):
            attns.append(
                    attn_head(inputs, bias_mat = bias_mat, 
                              out_sz = hid_units[0], activation = activatoin,
                              in_drop = ffd_drop, coef_drop = attn_drop, residual = False)
                    ) 
                    
        # [(B, N, F1), (B, N, F1)..] => (B, N, F1 * H1)
        
        h_1 = tf.concat(attns, axis = -1) # 连接上一层
        
        '''
        中间层,层数是 len(hid_units)-1;
        第i层有Hi个注意力头,输入是(B, N, F1*H1),每头注意力输出是(B, N, F1);
        每层均聚合所有头的注意力, 得到(B, N, Fi * Hi)
        '''
        # len(hid_units) = 中间层的个数
        for i in range(1, len(hid_units)):
            h_old = h_1 # 未使用
            attns = []
            # n_heads[i] = 中间第i层的注意力头数,设为Hi
            for _ in range(n_heads[i]):
                attns.append(
                        attn_head(h_1, bias_mat = bias_mat,
                                  out_sz = hid_units[i], activation = activation,
                                  in_drop = ffd_drop, coef_drop = attn_drop, residual = residual)
                        )
            
            # [(B, N, Fi), (B, N, Fi) ..] => (B, N, Fi*Hi)
            h_1 = tf.concat(attns, axis = -1) # 连接上一层
        
        '''
        最后一层,共有n_heads[-1]个注意力,一般为1
        输入: 最后一层的输出为(B, N, Fi*Hi)
        输出: (B, N, C), C是分类任务数
        输出:
        '''
        
        
        out = []
        for i in range(n_heads[-1]):
            out.append(
                    attn_head(h_1, bias_mat = bias_mat, 
                              out_sz = nb_classes, activation = lambda x : x,
                              in_  drop = ffd_drop, coef_drop = attn_drop, residual = False   )
                    )
        
        # 将多头注意力相加取平均
        logits = tf.add_n(out) / n_heads[-1]
        
        return logits
    


