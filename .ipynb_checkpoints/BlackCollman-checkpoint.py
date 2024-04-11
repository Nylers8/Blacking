import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder

from pygame import sprite


class BlackCollman(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers, dropout_rate=0.5, learning_rate=0.0001, epochs = 10):
        super(BlackCollman, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)])
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(hidden_size, output_size)  
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.epochs = epochs

    def forward(self, x):
        x = self.relu(self.fc1(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
            x = self.dropout(x)
        x = self.fc_out(x)
        return x

    def load_data(self):
        with open("game_data.json", 'r') as f:
            self.game_data = json.load(f)
        
        self.X = []
        self.y = []
        
        for data_entry in self.game_data:
            # Извлечение признаков из данных и создание строки признаков
            X_row = [
                data_entry["ball_x"],
                data_entry["ball_y"],
                data_entry["ball_dx"],
                data_entry["ball_dy"],
                data_entry["player_y"],
                data_entry["screen_width"],
                data_entry["screen_height"]
            ]
            
            # Добавление строки признаков в матрицу признаков
            self.X.append(X_row)
            
            # Добавление метки (player_x) в список меток
            self.y.append(data_entry["action"])

    def train(self, batch_size=32):
        # Преобразование данных в тензоры
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Обучение модели
        for epoch in range(self.epochs):
            running_loss = 0.0
            for inputs, labels in dataloader:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {running_loss}")

    def predict_action(self, input_data):
        # Предсказание действия для входных данных
        with torch.no_grad():
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            output = self(input_tensor)
            _, predicted = torch.max(output, 1)
            return predicted.item()
    

    def gather_game_data(self, ball_x, ball_y, ball_dx, ball_dy, player_y, screen_width, screen_height):
        # Формируем game_state
        game_state = []
        game_state.extend([ball_x, ball_y, ball_dx, ball_dy, player_y, screen_width, screen_height])

        game_state_tensor = torch.tensor(game_state, dtype=torch.float32).unsqueeze(0)
                
        return game_state_tensor
            



    def save_train(self):
        torch.save(self.state_dict(), "trainModel")

    def load_train(self):
        self.load_state_dict(torch.load("trainModel"))

