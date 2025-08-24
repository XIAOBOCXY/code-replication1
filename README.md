#### 8.20 
服务器上传mae-main的4/5个数据集+exp_emb_cpde的1/2数据集

服务器上传mae-main提供的训练好的权重

#### 8.21 
本地和github连接

本地上传mae-main的4/5个数据集+exp_emb_cpde的1/2数据集

下载EFC数据集train部分

#### 8.22 
服务器上传EFC数据集train部分

修改exp_emb_code\dataset.py，将完整的三元组用于训练

#### 8.23
下载EFC数据集test部分

修改exp_emb_code\dataset.py，将完整的三元组用于训练

本地重新安装支持 CUDA 的 PyTorch

服务器上传EFC数据集test部分

#### 8.24
重新下载了mae_pretrain_vit_base.pth，原来的损坏了

修改了mae_train_expemb.yaml和exp_emb_code/train.py中服务器相关的路径

修改exp_emb_code/dataset.py,将损坏文件进行跳过

移动./url_error_test.csv、./url_error_train.csv到datasets01/FEC Google

添加exp_emb_code/log/vit_vase_16_exp_emb_test,测试tensorboard可正常打开，但训练由于数据集过少而accum_iter过大导致日志数量过少

第一次训练准确率30%-40%浮动，修改log和checkpoints路径,修改accum_iter、lr、weight_decay，进行第二次训练

#### 计划：
将服务器上生成的exp_emb_code/checkpoints（忽略上传github）和exp_emb_code/log（会上传github）下载到本地

服务器代码上传github