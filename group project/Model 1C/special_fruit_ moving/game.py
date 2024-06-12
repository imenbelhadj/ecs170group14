import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# RGB colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
PURPLE = (128, 0, 128)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
LIGHT_BLUE = (26, 207, 255)
LIGHT_GREEN = (102, 211, 146)

BLOCK_SIZE = 20
SPEED = 40

class MovingObstacle:
    def __init__(self, x, y, w, h):
        self.position = Point(x,y)
        self.w = w
        self.h = h
        self.direction = random.choice([Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT])
    
    def possible_move(self):
        if self.direction == Direction.UP:  #checks if the obstacle direction is upward
            self.position = Point(self.position.x, self.position.y - BLOCK_SIZE)  #if upward, moving it up by one by subtracting BLOCK_SIZE from its current y position
        elif self.direction == Direction.DOWN:  #checks if the obstacle direction is downward
            self.position = Point(self.position.x, self.position.y + BLOCK_SIZE)   #if so, moving it up by down by adding  BLOCK_SIZE from its current y position
        elif self.direction == Direction.LEFT:  #checks if the obstacle direction is left faced
            self.position = Point(self.position.x - BLOCK_SIZE, self.position.y)   #if so, moving it left by substracting  BLOCK_SIZE from its current x position
        elif self.direction == Direction.RIGHT: #checks if the obstacle direction is right faced
            self.position = Point(self.position.x + BLOCK_SIZE, self.position.y)    #if so, moving it right by adding  BLOCK_SIZE from its current x position

     # Check for boundary collisions and change direction
        if self.position.x < 0 or self.position.x >= self.w or self.position.y < 0 or self.position.y >= self.h:
            self.change_direction()

    def change_direction(self):
        self.direction = random.choice([Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT])


class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # Init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self.special = None
        self.obstacle = None  # list to hold regular obstacles
        self.special_elements_collected = 0  # Track special elements
        self._place_food()
        self._place_obstacle()
        self._place_special()
        self.special_active = 0
        self.frame_iteration = 0
        self.moving_obstacle = MovingObstacle(random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE,
                                              random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE,
                                              self.w, self.h)

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food in self.snake:
                break

    def _place_obstacle(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.obstacle = Point(x, y)
            if self.obstacle not in self.snake and self.obstacle != self.food:
                break

    def _place_special(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.special = Point(x, y)
            if self.special not in self.snake and self.special != self.food and self.special != self.obstacle:
                break

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move
        self._move(action)  # Update the head
        self.snake.insert(0, self.head)

        self.moving_obstacle.possible_move()

        reward = 0
        game_over = False

        calculate_distance_to_food = self._manhattan_distance(self.head, self.food)
        calculate_distance_to_special = self._manhattan_distance(self.head, self.special)


        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -1
            return reward, game_over, self.score, self.special_elements_collected

        # Place new food or just move
        if self.head == self.food:
            self.score += 5
            reward = 10
            self._place_food()

        # Collect special element
        elif self.head == self.special:
            self.score += 10
            reward = 20
            self.special_active = 3  # Activate special effect for the next 3 games
            self.special_elements_collected += 1 
            self._place_special()
        

        elif self.head == self.moving_obstacle.position:
            if self.special_active > 0:
                reward = 0
            else:
                self.score -= 10 #higher penalty for crashing moving obstacle
                reward = -10
                self.snake.pop(0)

        # Check for obstacle collision without ending the game
        elif self.head == self.obstacle:
            if self.special_active > 0:
                reward = 0  # No penalty
            else:
                self.score -= 1
                reward = -1
                self.snake.pop(0)
        else:
            self.snake.pop()

        if game_over and self.special_active > 0:
            self.special_active -= 1  # Decrease the active game counter
        
        calculate_distance_to_food_2 = self._manhattan_distance(self.head, self.food)
        calculate_distance_to_special_2= self._manhattan_distance(self.head, self.special)

        if calculate_distance_to_food_2 < calculate_distance_to_food:
            reward += 0.2
        if calculate_distance_to_special_2 < calculate_distance_to_special:
            reward += 0.2

        reward += 0.1  # Small reward for each step taken

        # Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score, self.special_elements_collected

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True

        #hits obstacle:
        if pt == self.obstacle:
            return True
        # Hits moving obstacle
        if pt == self.moving_obstacle.position:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, PURPLE, pygame.Rect(self.obstacle.x, self.obstacle.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, LIGHT_BLUE, pygame.Rect(self.special.x, self.special.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, LIGHT_GREEN, pygame.Rect(self.moving_obstacle.position.x, self.moving_obstacle.position.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

        # Display special elements collected
        special_text = font.render("Specials: " + str(self.special_elements_collected), True, WHITE)
        self.display.blit(special_text, [0, 30])

        pygame.display.flip()


    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)


    def _manhattan_distance(self, point1, point2):
        return abs(point1.x - point2.x) + abs(point1.y - point2.y)