#common hyperparameters
seed=42
device=None
batch_size=128
epochs=100
lr=0.1
# hyperparameters for SGD
SGD_lr=0.1
# hyperparameters for SGD momentum
SGD_momentum_lr=0.1
SGD_momentum_coeff=0.9
# hyperparameters for Adam
Adam_beta1=0.9
Adam_beta2=0.999
Adam_lr=0.001
# hyperparameters for AdamW
AdamW_beta1=0.9
AdamW_beta2=0.999
AdamW_lr=0.001
AdamW_weight_decay=0.01
# hyperparameters for DecGD
DecGD_lr=0.1
DecGD_c=1
DecGD_gamma=0.9
DecGD_ams=False
# hyperparameters for SMARAN
SMARAN_lr=0.1
SMARAN_gamma=0.9
SMARAN_weight_decay=0.01
# hyperparameters for Adam
RAdam_beta1=0.9
RAdam_beta2=0.999
RAdam_lr=0.001
#hyperparameters for prodigy
prodigy_lr=1
prodigy_weight_decay=0.01
# model to be trained ('ResNet50', 'DenseNet')
model_type='DenseNet'
# Dataset used ('CIFAR10','CIFAR100','tinyimagenet')
dataset_type='tinyimagenet'