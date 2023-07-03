import torch
import time
from datetime import datetime
from torch.utils.data import DataLoader, random_split


class Trainer():
    def __init__(self, model, train_data, test_data, loss_fn, optimizer, split_test:float=None, device=None, batch_size=32, epochs=10, shuffle=True, name=None, test_metrics=None):
        if model == None or train_data == None:
            raise Exception("Model and train_data must be specified")

        if test_data == None:
            if split_test == None:
                raise Exception("Must specify either test_data or split_train")
            elif split_test != None:
                #warnings.warn("Test data is not specified, will split train data into train and validation")
                train_len = int(len(train_data)*(1-split_test))
                test_len = len(train_data) - train_len
                train_data, test_data = random_split(train_data, (train_len, test_len))

        if optimizer == None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-8)
        else:    
            self.optimizer = optimizer

        if device == None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if name == None:
            self.name = datetime.now().strftime("%Y-%m-%d_%H")
        else:
            self.name = name
        
        if test_metrics == None:
            self.test_metrics = [loss_fn]
        else:
            self.test_metrics = test_metrics

        self.batch_size = batch_size
        self.num_epochs = epochs
        self.model = model.to(self.device)
        self.loss_fn = loss_fn

        self.train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=shuffle)
        self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=shuffle)

    def train_loop(self):
        size = len(self.train_dataloader.dataset)
        time_at_start = time.time()*1000
        self.model.train()
        for batch, (*input, y) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            pred = self.model(*[i.to(self.device) for i in input])
            loss = self.loss_fn(pred, y.to(self.device))
            loss.backward()
            self.optimizer.step()
            
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(input[0])
                print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')
                run_time = time.time()*1000 - time_at_start
                print(f'time running: {run_time}, time per elem: {run_time/(current+1)}')
            
            if batch % 1000 == 0:
                print('saving model...')
                torch.save(self.model, 'tmp_entire_model.pth')

    def test_loop(self):
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        test_loss = [0 for _ in self.test_metrics]
        self.model.eval()
        with torch.no_grad():
            for batch, (*input, y) in enumerate(self.test_dataloader):
                pred = self.model(*[i.to(self.device) for i in input])
                y = y.to(self.device)
                for i, metric in enumerate(self.test_metrics):
                    loss = metric(pred, y)
                    test_loss[i] += loss.item()
                if batch % 100 == 0:
                    for i, metric in enumerate(self.test_metrics):
                        loss = test_loss[i] / (batch+1)
                        print(f'Metric {i}: {loss:>7f}  [{batch:>5d}/{num_batches:>5d}]')
        for i, metric in enumerate(self.test_metrics):
            loss = test_loss[i] / num_batches
            print(f'Metric {i}: {loss:>7f}')
        return test_loss


    def train_test(self):
        try:
            for t in range(self.num_epochs):
                print(f"Epoch {t + 1}\n-------------------------------")
                self.train_loop()
                self.current_test_loss = self.test_loop()
            print("Done!")
            torch.save(self.model.state_dict(), f'model_weights_{self.name}.pth')
            torch.save(self.model, f'entire_model_{self.name}.pth')
        except KeyboardInterrupt:
            print('Abort...')
            safe = input('Safe model [y]es/[n]o: ')
            if safe == 'y' or safe == 'Y':
                torch.save(self.model.state_dict(), f'model_weights_{self.name}.pth')
                torch.save(self.model, f'entire_model_{self.name}.pth')
            else: 
                print('Not saving...')
        
