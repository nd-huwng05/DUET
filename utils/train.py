from sklearn import metrics
from torchvision import transforms
from dataset.dataset import DatasetTemplate
from torch.utils.data import DataLoader
from utils.loss import CKA
from models.handler import handler_model
from utils.logger import Logger
import torch
import time
import numpy as np
import os

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize(self.config.DATASET.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def train_ensemble(self):
        Logger.set_log_file(self.config.OUTPUT.LOGS_TRAIN)
        dataset_train = DatasetTemplate(self.config, mode='train', transform=self.transform)
        Logger.get_logger().info(f"Loading dataset_train {self.config.DATASET.NAME} includes {dataset_train.__len__()} images")
        train_loader = DataLoader(dataset_train, batch_size=self.config.TRAIN.BATCH_SIZE, shuffle=True,
                                  drop_last=True, num_workers=self.config.DATASET.NUM_WORKERS, pin_memory=True)
        dataset_val = DatasetTemplate(self.config, mode='test', transform=self.transform)
        Logger.get_logger().info(
            f"Loading dataset_test {self.config.DATASET.NAME} includes {dataset_val.__len__()} images")
        val_loader = DataLoader(dataset_val, batch_size=self.config.TRAIN.BATCH_SIZE, shuffle=False,
                                drop_last=True, num_workers=self.config.DATASET.NUM_WORKERS, pin_memory=True)

        seniors = []
        for i in range(self.config.TRAIN.INDEX):
            senior = handler_model(self.config)
            Logger.get_logger().info(f"Loading state dict senior {i}.....")
            senior.load_state_dict(torch.load(os.path.join(self.config.OUTPUT.CHECKPOINT, f"{i}.pth"),
                                              map_location=torch.device("cuda:{}".format(self.config.GPUS)),))
            seniors.append(senior)

        junior = handler_model(self.config)

        optimizer = torch.optim.AdamW(junior.parameters(), lr=self.config.TRAIN.LR, betas=(0.5,0.999),
                                      weight_decay=self.config.TRAIN.WEIGHT_DECAY)

        Logger.get_logger().info(f"Training.......")
        t0 = time.time()
        [model.eval() for model in seniors]

        for e in range(self.config.TRAIN.EPOCHS):
            loss1l, loss2l, rars = [],[], []
            junior.train()
            for (x, _, _) in train_loader:
                x = x.cuda()
                x.requires_grad = False
                x_hat, log_var, z = junior(x)
                log_var = torch.clamp(log_var, min=-10, max=10)
                rec_err = (x_hat -x)**2
                loss1 = torch.mean(rec_err/torch.exp(log_var))
                loss2 = torch.mean(log_var)

                features = [model(x)[2] for model in seniors]
                RAR = [CKA(z, feature.detach()) for feature in features]

                if RAR == []:
                    RAR = torch.tensor([0], dtype=float, device="cuda")
                else: RAR = torch.mean(torch.stack(RAR), dim=0)

                loss = loss1 + loss2 + self.config.TRAIN.LAMBDA * RAR
                loss1l.append(rec_err.mean().item())
                loss2l.append(loss2.item())
                rars.append(RAR.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss1l = np.mean(loss1l)
            loss2l = np.mean(loss2l) if len(loss2l) > 0 else 0
            rars = np.mean(rars)

            if e%25 == 0 or e == 0 or e == self.config.TRAIN.EPOCHS - 1:
                t = time.time() - t0
                auc, ap = self.test_ensemble(model=junior, test_loader=val_loader)
                Logger.get_logger().info("Epoch[{:3d}/{}]  Time:{:.2f}s  AUC:{:.3f}  AP:{:.3f} "
                "Rec_err:{:.4f}  RAR_Loss:{:.4f} logvars:{:.4f}".format(
                     e, self.config.TRAIN.EPOCHS, t, auc, ap, loss1l, rars, loss2l
                ))

        if not os.path.exists(self.config.OUTPUT.CHECKPOINT):
            os.makedirs(self.config.OUTPUT.CHECKPOINT)

        model_name = os.path.join(self.config.OUTPUT.CHECKPOINT, f"{self.config.TRAIN.INDEX}.pth")
        torch.save(junior.state_dict(), model_name)
        self.config.TRAIN.INDEX += 1


    def test_ensemble(self, model, test_loader):
        model.eval()
        with torch.no_grad():
            y_score, y_true = [], []
            for i, (x,label,image_id) in enumerate(test_loader):
                x = x.cuda()
                x_hat, log_var, z = model(x)
                log_var = torch.clamp(log_var, min=-10, max=10)
                rec_err = (x_hat -x)**2
                res = torch.exp(-log_var) * rec_err
                res = res.mean(dim=(1,2,3))
                y_true.append(label.cpu())
                y_score.append(res.cpu().view(-1))

            y_true = np.concatenate(y_true)
            y_score = np.concatenate(y_score)

            auc = metrics.roc_auc_score(y_true, y_score)
            ap = metrics.average_precision_score(y_true, y_score)

            return auc, ap

    def train(self):
        for _ in range(self.config.TRAIN.INDEX, self.config.TRAIN.NUM_MODEL):
            self.train_ensemble()
