from torch.utils.data import DataLoader
from dataset.dataset import DatasetTemplate
from utils.logger import Logger
from utils.utils import load_models
from torch.autograd import Variable
from torchvision import transforms
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import torch

class InferenceGrad:
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize(self.config.DATASET.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def inference(self):
        Logger.set_log_file(self.config.OUTPUT.LOGS_TEST)
        dataset_test = DatasetTemplate(self.config, mode="test", transform=self.transform)
        dataloader = DataLoader(dataset_test, batch_size=1, shuffle=False,
                                drop_last=False, num_workers=self.config.DATASET.NUM_WORKERS, pin_memory=True)
        models = load_models(self.config)
        y_true = []
        unc_dis_list = []

        for x, label, image_id in tqdm(dataloader):
            x = Variable(x, requires_grad=True)
            x = x.cuda()
            grad_recs, unc = [], []
            for model in models:
                x_hat, log_var, z = model(x)
                log_var = torch.clamp(log_var, min=-10, max=10)
                rec_err = (x - x_hat)**2
                loss = torch.mean(torch.exp(-log_var) * rec_err)
                gradient = torch.autograd.grad(torch.mean(loss), x)[0].squeeze(0)
                grad_rec = torch.abs(x_hat - x) * gradient

                grad_recs.append(grad_rec)
                unc.append(torch.exp(log_var).squeeze(0))

            grad_recs = torch.cat(grad_recs)
            unc = torch.cat(unc)

            var = torch.mean(unc, dim=0)
            unc_dis = torch.std(grad_recs / torch.sqrt(var), dim=0)

            unc_dis_list.append(unc_dis.detach().mean().cpu())
            y_true.append(label.cpu().item())

        unc_dis_l = np.array(unc_dis_list)
        y_true = np.array(y_true)

        unc_auc = metrics.roc_auc_score(y_true, unc_dis_l)
        unc_ap = metrics.average_precision_score(y_true, unc_dis_l)

        Logger.get_logger().info(f'AUC: {unc_auc}, AP: {unc_ap}')
        return unc_auc, unc_ap