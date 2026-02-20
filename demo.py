import pygame as pg
import numpy as np
import mnist_loader
import pickle
import cv2
import network1
pg.init()
pg.font.init()
font = pg.font.SysFont("Arial", 40)
    
screen = pg.display.set_mode((490, 600))

class square():
    def __init__(self, x, y):
        self.pos = (x, y)
        self.x = x
        self.y = y
        self.color = (255,255,255)
    def update(self):
        mouse = pg.mouse.get_pos()
        pressed = pg.mouse.get_pressed()

        mouse_left = (mouse[0] - 15, mouse[1])
        mouse_right = (mouse[0] + 15, mouse[1])
        mouse_up = (mouse[0], mouse[1] + 15)
        mouse_down = (mouse[0], mouse[1] - 15)

        grey_val = 140

        #
        if(self.x <= mouse[0] and self.x + 20 >= mouse[0] and self.y <= mouse[1] and self.y + 20 >= mouse[1] and pressed == (1, 0, 0)):
            self.color = (0, 0, 0)
            
        if(self.x <= mouse[0] and self.x + 15 >= mouse[0] and self.y <= mouse[1] and self.y + 15 >= mouse[1] and pressed == (0, 0, 1)):
            self.color = (255, 255, 255)
        
    def draw(self, screen):
        pg.draw.rect(screen, self.color, (self.x, self.y, 15, 15)) 
        

grid = []
for i in range(0, 28):
    row = []
    y = 20 + i * 16
    for j in range(0, 28):
        x = 20 + j * 16
        s = square(x = x, y = y)
        row.append(s)
    grid.append(row)

def reset_board():
    for r in grid:
        for x in r:
            x.color = (255, 255, 255)

class reset_button():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.color = (143, 22, 22)

    def update(self):
        mouse = pg.mouse.get_pos()
        pressed = pg.mouse.get_pressed()
        if(self.x <= mouse[0] and self.x + 140 >= mouse[0] and self.y <= mouse[1] and self.y + 70 >= mouse[1] and pressed == (1, 0, 0)):
            reset_board()

    def draw(self, screen):
        pg.draw.rect(screen, self.color, (self.x, self.y, 140, 70)) 

def vectorize_grid():
    input = []
    for row in grid:
        for sqaure in row:
            grayscale = 1.0 - (sqaure.color[0] / 255)
            input.append(grayscale)
    return np.reshape(input, (784, 1))

def enhanced_vectorize_grid():
    
    input = []
    for row in grid:
        for square in row:
            grayscale = 1.0 - (square.color[0] / 255.0)
            if 1:
                grayscale = 1.0 if grayscale > 0.3 else 0.0  # remove noise
            input.append(grayscale)
    return np.reshape(input, (784, 1))

with open("trained_network.pkl", "rb") as f:
    net = pickle.load(f)

rb = reset_button(20, 500)        


running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
    screen.fill((100, 100, 200))

    for row in grid:
        for sq in row:
            sq.update()
            sq.draw(screen)
    
    rb.draw(screen)
    rb.update()

    input = vectorize_grid()
    output = net.feedforward(input)
    prediction = np.argmax(output)

    text_surface = font.render(f"Prediction: {prediction}", True, (255, 255, 255))
    screen.blit(text_surface, (200, 510))

    pg.display.flip()




pg.quit()