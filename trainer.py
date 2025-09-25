import torch
import numpy
import copy
import utils
import torch.optim as optim
from optimizers import DecGD, SMARAN, Prodigy,PoNoS
import config
from sklearn.metrics import average_precision_score
import torch.nn.functional as F


class Trainer:
    def __init__(self, device):
        self.device=device
    
    def closure(self):
        self.optimizer.zero_grad()
        output = self.model(self.data)
        loss = self.criterion(output, self.target)
        # loss.backward()
        return loss

    def train_model(self, dataset, epochs):
        training_loss_list = []
        testing_loss_list = []
        testing_accuracy_list=[]
        mAP_list=[]
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for i, (self.data, self.target) in enumerate(dataset.train_loader):
                self.data, self.target = self.data.to(self.device), self.target.to(self.device)  # Move data to GPU
                loss = self.optimizer.step(self.closure)
                epoch_loss += loss.item()
               

            training_loss = epoch_loss / (i + 1)
            test_loss , test_accuracy, mAP= self.test_model_loss(dataset.test_loader)
            training_loss_list.append(training_loss)
            testing_loss_list.append(test_loss)
            mAP_list.append(mAP)
            testing_accuracy_list.append(test_accuracy)
            print(f'epoch :{epoch} training loss:{training_loss}, testing loss:{test_loss}, testing accuracy:{test_accuracy}, mAP:{mAP}')
        return training_loss_list, testing_loss_list, testing_accuracy_list, mAP_list

    def test_model(self, dataset):
        self.model.eval()
        correct = 0
        total = 0
    
        with torch.no_grad():
            for data, target in dataset:
                data, target = data.to(self.device), target.to(self.device)  # Move data to GPU
                output = self.model(data)
                predictions = output.argmax(dim=1)
                correct += (predictions == target).sum().item()
                total += target.size(0)
    
        accuracy = correct / total * 100
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

    # def test_model_loss(self, dataset):
    #     self.model.eval()
    #     total_loss = 0
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for i, (data, target) in enumerate(dataset):
    #             data, target = data.to(self.device), target.to(self.device)  # Move data to GPU
    #             output = self.model(data)
    #             loss = self.criterion(output, target)
    #             total_loss += loss.item()
    #             #test accuracy
    #             predictions = output.argmax(dim=1)
    #             correct += (predictions == target).sum().item()
    #             total += target.size(0)
    #     accuracy = correct / total * 100   
    #     return total_loss / (i + 1) , accuracy

    
    def test_model_loss(self, dataset):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
    
        all_targets = []
        all_probs = []
    
        with torch.no_grad():
            for i, (data, target) in enumerate(dataset):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
    
                # Loss
                loss = self.criterion(output, target)
                total_loss += loss.item()
    
                # Accuracy
                predictions = output.argmax(dim=1)
                correct += (predictions == target).sum().item()
                total += target.size(0)
    
                # Collect for mAP
                probs = F.softmax(output, dim=1)
                all_probs.append(probs.cpu())
                all_targets.append(target.cpu())
    
        # Stack all predictions and targets
        all_probs = torch.cat(all_probs, dim=0).numpy()  # shape: (num_samples, num_classes)
        all_targets = torch.cat(all_targets, dim=0)       # shape: (num_samples,)
        all_targets_onehot = F.one_hot(all_targets, num_classes=all_probs.shape[1]).numpy()
    
        # Compute AP per class
        aps = []
        for i in range(all_probs.shape[1]):
            ap = average_precision_score(all_targets_onehot[:, i], all_probs[:, i])
            aps.append(ap)
        mAP = sum(aps) / len(aps)
    
        accuracy = correct / total * 100
        avg_loss = total_loss / (i + 1)
    
        return avg_loss, accuracy, mAP


    def train(self, model, optimizer_names, dataset, criterion, epochs):
        self.criterion=criterion
        for opt_name in optimizer_names:
            opt_name_lower = opt_name.lower()
            if opt_name_lower == "sgd":
                self.model = copy.deepcopy(model).to(self.device)
                self.optimizer = optim.SGD(self.model.parameters(),
                                              lr=config.SGD_lr)
                train_list, test_list, accuracy_list,mAP_list= self.train_model(dataset, epochs=epochs)
                utils.save_losses_to_excel(train_list,test_list,accuracy_list,mAP_list, filename=f'results/{opt_name_lower}.xlsx')
                
                
            elif opt_name_lower == "sgdm":
                self.model = copy.deepcopy(model).to(self.device)
                self.optimizer = optim.SGD(self.model.parameters(),
                                              lr=config.SGD_momentum_lr,
                                              momentum=config.SGD_momentum_coeff)
                train_list, test_list,accuracy_list,mAP_list= self.train_model(dataset, epochs=epochs)
                utils.save_losses_to_excel(train_list,test_list,accuracy_list,mAP_list, filename=f'results/{opt_name_lower}.xlsx')
                
            
            elif opt_name_lower == "adam":
                self.model = copy.deepcopy(model).to(self.device)
                self.optimizer = optim.Adam(self.model.parameters(),
                                                lr=config.Adam_lr, betas=(config.Adam_beta1,config.Adam_beta2) )
                train_list, test_list,accuracy_list,mAP_list= self.train_model(dataset, epochs=epochs)
                utils.save_losses_to_excel(train_list,test_list,accuracy_list,mAP_list, filename=f'results/{opt_name_lower}.xlsx')


            elif opt_name_lower == "radam":
                self.model = copy.deepcopy(model).to(self.device)
                self.optimizer = optim.RAdam(self.model.parameters(),
                                                lr=config.RAdam_lr, betas=(config.RAdam_beta1,config.RAdam_beta2))
                train_list, test_list,accuracy_list,mAP_list= self.train_model(dataset, epochs=epochs)
                utils.save_losses_to_excel(train_list,test_list,accuracy_list,mAP_list, filename=f'results/{opt_name_lower}.xlsx')

                
                
            elif opt_name_lower == "adamw":
                self.model = copy.deepcopy(model).to(self.device)
                self.optimizer = optim.AdamW(self.model.parameters(),
                                                  lr=config.AdamW_lr, betas=(config.AdamW_beta1,config.AdamW_beta2),
                                                  weight_decay=config.AdamW_weight_decay)
                train_list, test_list,accuracy_list,mAP_list= self.train_model(dataset, epochs=epochs)
                utils.save_losses_to_excel(train_list,test_list,accuracy_list,mAP_list, filename=f'results/{opt_name_lower}.xlsx')

            elif opt_name_lower == "decgd":
                self.model = copy.deepcopy(model).to(self.device)
                self.optimizer = DecGD(self.model.parameters(), 
                                       lr=config.DecGD_lr,
                                       c=config.DecGD_c, 
                                       gamma=config.DecGD_gamma, 
                                       ams=config.DecGD_ams,
                                       device=self.device) 
                train_list, test_list,accuracy_list,mAP_list= self.train_model(dataset, epochs=epochs)
                utils.save_losses_to_excel(train_list,test_list,accuracy_list,mAP_list, filename=f'results/{opt_name_lower}.xlsx')

            elif opt_name_lower == "smaran":
                self.model = copy.deepcopy(model).to(self.device)
                self.optimizer = SMARAN(self.model.parameters(), 
                                       lr=config.SMARAN_lr,
                                       gamma=config.SMARAN_gamma, 
                                       weight_decay=config.SMARAN_weight_decay,
                                       device=self.device) 
                train_list, test_list,accuracy_list,mAP_list= self.train_model(dataset, epochs=epochs)
                utils.save_losses_to_excel(train_list,test_list,accuracy_list,mAP_list, filename=f'results/{opt_name_lower}.xlsx')

            elif opt_name_lower == "prodigy":
                self.model = copy.deepcopy(model).to(self.device)
                self.optimizer = Prodigy(self.model.parameters(), 
                                       lr=config.prodigy_lr,
                                       weight_decay=config.prodigy_weight_decay,
                                        use_bias_correction=True,d_coef=5,slice_p=11,safeguard_warmup=True) 
                train_list, test_list,accuracy_list,mAP_list= self.train_model(dataset, epochs=epochs)
                utils.save_losses_to_excel(train_list,test_list,accuracy_list,mAP_list, filename=f'results/{opt_name_lower}.xlsx')

            elif opt_name_lower == "ponos":
                self.model = copy.deepcopy(model).to(self.device)
                self.optimizer = PoNoS(self.model.parameters()) 
                train_list, test_list,accuracy_list,mAP_list= self.train_model(dataset, epochs=epochs)
                utils.save_losses_to_excel(train_list,test_list,accuracy_list,mAP_list, filename=f'results/{opt_name_lower}.xlsx')
    
            else:
                raise ValueError(f"Unsupported optimizer or missing import: {opt_name}")
    
        return True
