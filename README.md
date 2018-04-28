# tf_transfer_learning
这个主要是针对迁移学习做的， 对inception网络或者inception_resnet_v2网络进行微调， 然后用网络的权重值进行预训练， 注意在训练的时候， pet_net_v1中我们固定了inception或者inception_resnet_v2的权重值， 使其固化， 然后对后面我们添加的两层进行训练， 我在里面暂时还没有训练， 大家感兴趣的话， 可以加自己的数据集， 进行训练！
