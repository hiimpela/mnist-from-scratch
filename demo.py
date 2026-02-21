import pygame as pg
import numpy as np
import json
import network
from scipy import ndimage

pg.init()
screen = pg.display.set_mode((600, 800))
clock = pg.time.Clock()
font = pg.font.SysFont("Arial", 32)
drawing_surf = pg.Surface((448, 448))
drawing_surf.fill((0, 0, 0))

with open("trained_model.json", "r") as f:
    data = json.load(f)
net = network.network(data["sizes"])
net.weights = [np.array(w) for w in data["weights"]]
net.biases = [np.array(b) for b in data["biases"]]

def get_processed_input(surf):
    img_28 = pg.transform.smoothscale(surf, (28, 28))
    arr = pg.surfarray.array3d(img_28)
    gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114]).T / 255.0
    
    if np.max(gray) > 0:
        gray = gray / np.max(gray)
        
    if np.sum(gray) > 0.1:
        cy, cx = ndimage.center_of_mass(gray)
        shift_y, shift_x = 13.5 - cy, 13.5 - cx
        gray = ndimage.shift(gray, (shift_y, shift_x), mode='constant', cval=0)
    
    return gray.clip(0, 1)

def draw_brush(surf, pos):
    for r in range(25, 0, -5):
        alpha = int(255 * (1 - (r / 30)))
        pg.draw.circle(surf, (alpha, alpha, alpha), pos, r)

drawing = False
prediction = 0
preview_img = np.zeros((28, 28))
running = True

while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.MOUSEBUTTONDOWN:
            if pg.Rect(76, 50, 448, 448).collidepoint(event.pos):
                drawing = True
            if pg.Rect(230, 720, 140, 45).collidepoint(event.pos):
                drawing_surf.fill((0, 0, 0))
                preview_img = np.zeros((28, 28))
        elif event.type == pg.MOUSEBUTTONUP:
            drawing = False
            processed = get_processed_input(drawing_surf)
            preview_img = processed
            prediction = np.argmax(net.feedforward(processed.reshape(784, 1)))

    if drawing:
        m_pos = pg.mouse.get_pos()
        if pg.Rect(76, 50, 448, 448).collidepoint(m_pos):
            draw_brush(drawing_surf, (m_pos[0] - 76, m_pos[1] - 50))

    screen.fill((25, 25, 30))
    pg.draw.rect(screen, (255, 255, 255), (74, 48, 452, 452), 2)
    screen.blit(drawing_surf, (76, 50))
    
    preview_pixels = np.repeat((preview_img.T * 255)[:, :, np.newaxis], 3, axis=2)
    preview_surf = pg.surfarray.make_surface(preview_pixels.astype(np.uint8))
    preview_surf = pg.transform.scale(preview_surf, (112, 112))
    screen.blit(preview_surf, (76, 520))
    
    screen.blit(font.render(f"Prediction: {prediction}", True, (0, 255, 150)), (230, 550))
    pg.draw.rect(screen, (150, 30, 30), (230, 720, 140, 45))
    screen.blit(font.render("Clear", True, (255, 255, 255)), (265, 722))

    pg.display.flip()
    clock.tick(60)

pg.quit()