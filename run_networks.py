from torch.optim import lr_scheduler
from model import *
from utils.evaluation import *
from utils.load_loss import getLoss
from utils.optimizer import CosineScheduler


class Train():
    def __init__(self, args, dataset, test=False):
        self.args = args
        self.dataset = dataset
        self.test_mode = test

        self.embed_mean = torch.zeros(int(self.args.linear_size)).numpy()
        self.mu = 0.9

        self.init_models()

        if not self.test_mode:

            print('Using steps for training.')
            self.training_data_num = len(self.dataset.train_loader)
            self.epoch_steps = int(self.training_data_num / self.args.batch_size)

            print('Initializing model optimizer.')
            self.init_optimizers()
            self.init_criterions()
        
    def init_models(self):
        x = self.dataset.label_encode.to(self.args.device)
        G = self.dataset.G.to(self.args.device)
        self.extractor = Extractor(self.args, x, G).to(self.args.device) # ours
        self.classifier = Classifier_TDE_GELU(self.args).to(self.args.device)  # ours

    def init_criterions(self):
        self.criteria = getLoss("FD", self.dataset.class_freq, self.dataset.train_num)

    def init_optimizers(self):
        self.model_optimizer_dict = {}
        extractor_parameters = [parameter for parameter in self.extractor.parameters() if parameter.requires_grad]
        classifier_parameters = [parameter for parameter in self.classifier.parameters() if parameter.requires_grad]
        parameters = extractor_parameters + classifier_parameters

        self.optimizer = torch.optim.Adam(parameters, lr=self.args.learning_rate)
        self.scheduler = CosineScheduler(10000, base_lr=self.args.learning_rate, warmup_steps=500)

        self.model_optimizer_dict['optimizer'] = self.optimizer

    def show_current_lr(self):
        max_lr = 0.0
        for key, val in self.model_optimizer_dict.items():
            lr_set = list(set([para['lr'] for para in val.param_groups]))
            if max(lr_set) > max_lr:
                max_lr = max(lr_set)
        return max_lr


    def batch_forward(self, seqvec, evolution, onehot, length, label, struct, feature_ext=False, phase='train'):

        if phase == 'train':
            self.extractor.train()
            self.classifier.train()
        else:
            self.extractor.eval()
            self.classifier.eval()

        # Calculate Features
        self.features = self.extractor(seqvec=seqvec,
                                       evolution=evolution,
                                       onehot=onehot,
                                       length=length,
                                       struct=struct)

        # update moving average
        if phase == 'train':
            self.embed_mean = self.mu * self.embed_mean + self.features.detach().mean(0).view(-1).cpu().numpy()

        # If not just extracting features, calculate logits
        if not feature_ext:
            # cont_eval = 'continue_eval' in self.training_opt and self.training_opt['continue_eval'] and phase != 'train'
            self.logits = self.classifier(self.features, self.embed_mean)

    def batch_backward(self, label, steps):

        loss = self.criteria(self.logits, label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            if self.scheduler.__module__ == lr_scheduler.__name__:
                self.scheduler.step()
            else:
                # Using custom defined scheduler
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.scheduler(steps)
        return loss.item(), steps

    def train(self):

        best_valid_result = 0.0
        steps = 1

        # Loop over epochs
        for epoch in range(self.args.epochs):
            torch.cuda.empty_cache()
            total_loss = 0.0

            # print learning rate
            current_lr = self.show_current_lr()
            current_lr = min(current_lr * 50, 1.0)
            self.mu = 1.0 - (1 - 0.9) * current_lr

            for step, (seqvec, evolution, onehot, length, label, struct) in enumerate(self.dataset.train_loader):

                seqvec = seqvec.to(self.args.device)
                evolution = evolution.to(self.args.device)
                onehot = onehot.to(self.args.device)
                length = length.to(self.args.device)
                label = label.to(self.args.device)
                struct = struct.to(self.args.device)

                # If on training phase, enable gradients
                with torch.set_grad_enabled(True):

                    # If training, forward with loss, and no top 5 accuracy calculation
                    self.batch_forward(seqvec, evolution, onehot, length, label, struct, phase='train')
                    batch_loss, steps = self.batch_backward(label, steps)
                    total_loss += batch_loss
                    steps += 1

            total_loss = total_loss / len(self.dataset.train_loader)
            
            print(f'Epochs: {epoch} | Train Loss: {total_loss: .4f}')

           
            if epoch = 199:
                torch.save({'extractor': self.extractor.state_dict(),
                            'classifier': self.classifier.state_dict(),
                            'embed_mean': self.embed_mean}, self.args.check_pt_model_path)

    def eval(self, phase='val'):
 
        torch.cuda.empty_cache()
        self.extractor.eval()
        self.classifier.eval()

        y_test = []
        y_pred = []
        feature = []

        # Iterate over dataset
        with torch.no_grad():
            for seqvec, evolution, onehot, length, label, struct in self.dataset.test_loader:

                seqvec = seqvec.to(self.args.device)
                evolution = evolution.to(self.args.device)
                onehot = onehot.to(self.args.device)
                length = length.to(self.args.device)
                label = label.to(self.args.device)
                struct = struct.to(self.args.device)

                self.batch_forward(seqvec, evolution, onehot, length, label, struct, phase=phase)
                feature.extend(self.features.cpu().tolist())
                y_pred.extend(self.logits.cpu().tolist())
                y_test.extend(label.cpu().tolist())

        y_pred, y_test, feature = np.array(y_pred), np.array(y_test), np.array(feature)
        result = evaluate(y_pred, y_test)

        # y_pred_head = y_pred[:, [1, 9]]
        # y_test_head = y_test[:, [1, 9]]
        # print("删除前y_pred_head的形状：", y_pred_head.shape)
        # print("删除前y_test_head的形状：", y_test_head.shape)
        # zero_rows = np.where(~y_test_head.any(axis=1))[0]
        # y_pred_head = np.delete(y_pred_head, zero_rows, axis=0)
        # y_test_head = np.delete(y_test_head, zero_rows, axis=0)
        # print("删除后y_pred_head的形状：", y_pred_head.shape)
        # print("删除后y_test_head的形状：", y_test_head.shape)
        # result = evaluate(y_pred_head, y_test_head)

        # y_pred_middle = y_pred[:, [2, 6, 8, 13, 20]]
        # y_test_middle = y_test[:, [2, 6, 8, 13, 20]]
        # print("删除前y_pred_middle的形状：", y_pred_middle.shape)
        # print("删除前y_test_middle的形状：", y_test_middle.shape)
        # zero_rows = np.where(~y_test_middle.any(axis=1))[0]
        # y_pred_middle = np.delete(y_pred_middle, zero_rows, axis=0)
        # y_test_middle = np.delete(y_test_middle, zero_rows, axis=0)
        # print("删除后y_pred_middle的形状：", y_pred_middle.shape)
        # print("删除后y_test_middle的形状：", y_test_middle.shape)
        # result = evaluate(y_pred_middle, y_test_middle)

        # y_pred_tail = y_pred[:, [0, 3, 4, 5, 7, 10, 11, 12, 14, 15, 16, 17, 18, 19]]
        # y_test_tail = y_test[:, [0, 3, 4, 5, 7, 10, 11, 12, 14, 15, 16, 17, 18, 19]]
        # print("删除前y_pred_tail的形状：", y_pred_tail.shape)
        # print("删除前y_test_tail的形状：", y_test_tail.shape)
        # zero_rows = np.where(~y_test_tail.any(axis=1))[0]
        # y_pred_tail = np.delete(y_pred_tail, zero_rows, axis=0)
        # y_test_tail = np.delete(y_test_tail, zero_rows, axis=0)
        # print("删除后y_pred_tail的形状：", y_pred_tail.shape)
        # print("删除后y_test_tail的形状：", y_test_tail.shape)
        # result = evaluate(y_pred_tail, y_test_tail)

        # y_pred_mixed = y_pred[:, [0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
        # y_test_mixed = y_test[:, [0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
        # print("删除前y_pred_mixed的形状：", y_pred_mixed.shape)
        # print("删除前y_test_tail的形状：", y_test_mixed.shape)
        # zero_rows = np.where(~y_test_mixed.any(axis=1))[0]
        # y_pred_mixed = np.delete(y_pred_mixed, zero_rows, axis=0)
        # y_test_mixed = np.delete(y_test_mixed, zero_rows, axis=0)
        # print("删除后y_pred_mixed的形状：", y_pred_mixed.shape)
        # print("删除后y_test_mixed的形状：", y_test_mixed.shape)
        # result = evaluate(y_pred_mixed, y_test_mixed)

        return result, y_pred, y_test, feature

    def load_model(self, model_dir=None):
        checkpoint = torch.load(model_dir, map_location=self.args.device)
        self.embed_mean = checkpoint['embed_mean']
        self.extractor.load_state_dict(checkpoint['extractor'])
        self.classifier.load_state_dict(checkpoint['classifier'])

    def predictor(self, seqvec, evolution, onehot, length, struct, phase='val'):

        torch.cuda.empty_cache()
        self.extractor.eval()
        self.classifier.eval()

        seqvec = seqvec.to(self.args.device)
        evolution = evolution.to(self.args.device)
        onehot = onehot.to(self.args.device)
        length = length.to(self.args.device)
        label = torch.zeros((2, 3))
        struct = struct.to(self.args.device)

        self.batch_forward(seqvec, evolution, onehot, length, label, struct, phase=phase)

        return self.logits




