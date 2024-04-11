import pygame
import os
import random
import json

import header
from Players import *
from Ball import *
from CalBall import *

from BlackCollman import BlackCollman

# Просчёт игры
def calBall(player):
    fakeBall = CalBall()
    fakeBall.rect.x = ball.rect.x
    fakeBall.rect.y = ball.rect.y
    fakeBall.dx = ball.dx
    fakeBall.dy = ball.dy
    while True:
        fakeBall.update()
        if player.rect.centerx + 5 > fakeBall.rect.centerx > player.rect.centerx - 5:
            break
    return fakeBall.rect.centery

# Авто-игра
def auto_game():
    global changeBict
    global bict
    # Смещение
    if changeBict:
        bict = random.randint(-50,50)
        changeBict = False
    # Если коснулась игрока, то меняется смещение
    if ball.rect.colliderect(player1.rect) or ball.rect.colliderect(player2.rect):
        changeBict = True
        ball.update(player1,player2)

    # Переключение левого игрока на АИ
    if not AiGame:
        if ball.dx < 0 and ball.rect.x > player1.rect.right:
            player1.predictY = calBall(player1) + bict
        if player1.predictY-10 < player1.rect.centery < player1.predictY+10:
            player2.action = "nothing"
        elif player1.rect.centery > player1.predictY:
            player1.action = "moveUp"
            player1.moveUp()
        elif player1.rect.centery < player1.predictY:
            player1.action = "moveDown"
            player1.moveDown()
            

    if ball.dx > 0 and ball.rect.right < player2.rect.x:
         player2.predictY = calBall(player2) + bict

            
    if player2.predictY-10 < player2.rect.centery < player2.predictY+10:
        player2.action = "nothing"
    elif player2.rect.centery > player2.predictY:
        player2.action = "moveUp"
        player2.moveUp()
    elif player2.rect.centery < player2.predictY:
        player2.action = "moveDown"
        player2.moveDown()


# Создание окна
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((header.WIDTH, header.HEIGHT))
pygame.display.set_caption("Blackanoid")
clock = pygame.time.Clock()


# Данные от игрока
header.FPS = int(input("Введите кол-во FPS(стандарт:60): "))
neuron = int(input("Введите кол-во нейронов: "))
slayer = int(input("Введите кол-во слоев: "))
speed_rate = float(input("Введите скорость обучения(стандарт:0.001): "))
epochs_rate = int(input("Введите кол-во эпох(стандарт:10):"))

auto_learn = bool(int(input("Включить авто-обучение<0/1>:")))
if auto_learn:
    _interval_learn = int(input("Введите время автообучения(в минутах): "))
    interval_learn = 60 * _interval_learn * header.FPS
    timer_learn = 0


# Создание объектов
player1 = Player(20, (25,255,20))
player2 = Player(header.WIDTH-player1.rect.width, (120,45,20))

ball = Ball()
all_sprites = pygame.sprite.Group()
all_sprites.add(player1, player2)

# Создание нейросети(Входные данные, нейроны, выходные, слои)
blackCollman = BlackCollman(7,neuron,3,slayer, learning_rate = speed_rate, epochs = epochs_rate)

# Текст
font = pygame.font.Font(None, 36)  # Вы можете указать путь к файлу шрифта и размер
text_surface = font.render(str(player1.score), True, (0,0,0))  # Текст, сглаживание, цвет

text_rect_player1 = text_surface.get_rect()
text_rect_player2 = text_surface.get_rect()
text_rect_player1.topleft = (50, 40) # Для Score
text_rect_player2.topleft = (header.WIDTH-86, 40) # Для Состояния

game_data = [] # Здесь хранятся данные

# Инициализация смещения
bict = random.randint(-20,20)
changeBict = False

running = True # Работает ли игра
AiGame = False # Играет ли нейросеть
auto_gaming = True

while running:
    # Держим цикл на правильной скорости
    clock.tick(header.FPS)

    # Обновления
    ball.update(player1,player2)

    if auto_gaming:
        auto_game()

    if auto_learn:
        if timer_learn >= interval_learn:
            with open('game_data.json', 'w') as json_file:
                json.dump(game_data, json_file)
            blackCollman.load_data()
            blackCollman.train()
            AiGame = True
            auto_learn = 0
        else:
            timer_learn += 1
            
        

    # Сохраненние данных
    action_to_label = {"moveUp": 0, "moveDown": 1, "nothing": 2}
    action_label = action_to_label[player1.action]
    game_data.append({
        "ball_x": ball.rect.x,
        "ball_y": ball.rect.y,
        "ball_dx": ball.dx,
        "ball_dy": ball.dy,
        "player_y": player1.rect.y,
        "screen_width": header.WIDTH,
        "screen_height": header.HEIGHT,
        "action": action_label  
    })

    # Играет ли ИИ
    if AiGame:
        gather_data = blackCollman.gather_game_data(ball.rect.x, ball.rect.y, ball.dx, ball.dy, player1.rect.y, header.WIDTH, header.HEIGHT)
        predict = blackCollman.predict_action(gather_data)
        if predict == 0 and player1.rect.top > 0:
            player1.moveUp()
            player1.action = "moveUp"
        if predict == 1.0 and player1.rect.bottom < header.HEIGHT:
            player1.moveDown()
            player1.action = "moveDown"
        else:
            player1.action = "nothing"

    
    # Разовые нажатия
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_s:
                with open('game_data.json', 'w') as json_file:
                    json.dump(game_data, json_file)
            if event.key == pygame.K_q:
                blackCollman.load_data()
                blackCollman.train()
            if event.key == pygame.K_z:
                blackCollman.save_train()
            if event.key == pygame.K_x:
                blackCollman.load_train()
            if event.key == pygame.K_f:
                AiGame = not AiGame
            if event.key == pygame.K_i:
                auto_gaming = not auto_gaming

    # Проверка нажатия клавиш    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] and player1.rect.top > 0:
        player1.moveUp()
    if keys[pygame.K_s] and player1.rect.bottom < header.HEIGHT:
        player1.moveDown()
    if keys[pygame.K_UP] and player2.rect.top > 0:
        player2.moveUp()
    if keys[pygame.K_DOWN] and player2.rect.bottom < header.HEIGHT:
        player2.moveDown()


    # Рендеринг
    screen.fill((0,0,0))
    all_sprites.draw(screen)
    screen.blit(ball.image,ball.rect)
    
    text_surface = font.render(str(player1.score), True, (255,255,255))  # Текст, сглаживание, цвет
    screen.blit(text_surface, text_rect_player1)
    text_surface = font.render(str(player2.score), True, (255,255,255))  # Текст, сглаживание, цвет
    screen.blit(text_surface, text_rect_player2)
    
    pygame.display.flip()

pygame.quit()
