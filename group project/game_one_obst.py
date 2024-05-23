import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
GREEN = (0, 255, 0) #this will be the color of the obstacle

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.obstacle = None
        self.food = None
        self._place_food()
        self.place_obstacle()
        self.frame_iteration = 0


    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake or self.food == self.obstacle:
            self._place_food()

    def place_obstacle(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        self.obstacle = Point(x,y)
        if self.obstacle in self.snake or self.obstacle == self.food:
            self.place_obstacle()

    def play_step(self, action):
        #action as an argument
        #where action is a direction the snake should move
        #incrementing frame_iteration which keeps track 
        #of how many frames have passed since game started
        self.frame_iteration += 1

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        #collects events from the game
            #such as mouse clicks, key presses
        #checks for events that would cause a quit
        #quit is when game window closes -- game quits

        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        #addresses snake's movement
        #uses move function with the action parameters
        #updates snake's head based on chosen action
        #then, inserts this updated head at head of snake's body
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        #if collision occurs (is collision)
        #or snake becomes too long
        #game ends, and game_over variable will be True
        #assigns negative reward, returns reward, game_over state, and score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        #handling whether snake has eaten food or not
        #if the head of the snake is in the same nplace as the food
        #then the snake will eat the food
        #score increments, and positive reward is given
        #then new food is placed
        #if snake doesn't eat the food, snake's tail falls off

        #5. place obstacle or just move
        if self.head == self.obstacle:
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # 6. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        #makes sure game runs at a consistent speed
        #refresh display and control frame rate

        # 7. return game over and score
        return reward, game_over, self.score
        #returning the reward, the game_over state, and the score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        #takes argument pt
        #pt is where we check for collision
        #if not given, use the head of the snake

        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        #checking if pt is hitting game boundary
        #if outside the bounds of the game, will return True

        # hits itself
        if pt in self.snake[1:]:
            return True
        #checks if pt is hitting the snake's own body
        #self.snake is the list of snake body segements (w/o the head of the snake)
        #if we find pt location in list of snake body
        #then we're colliding with snake

        #hits obstacle
        if pt in self.obstacle:
            return True
        #checks if the snake has hit the obstacle

        return False
        #if none of these conditions met
        #then we're not colliding and we can return False

    def _update_ui(self):
        #meant to update user interface
        #basically just how it looks on our end
        self.display.fill(BLACK)
        #filing the game with black color

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        #draws each segment of the snake
        #drawing main body of snake using BLUE1
        #using rectangles of the size BLOCK_SIZE
        #then using smaller rectangles in dif BLUE2 to show borders

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        #drawing the food pellets on the display
        #just a red rectangle at coordinates of self.food
        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.obstacle.x, self.obstacle.y, BLOCK_SIZE, BLOCK_SIZE))
        #drawing the obstacles on the display
        #just a green rectangle at coordinates of self.obstacle
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        #rendering and displaying score in top left corner
        #displaying in WHITE font

        pygame.display.flip()
        #updating display surface to show the changes from above
        #so as a whole - draws snake, food, and score

    def _move(self, action):
        # [straight, right, left]
        #handles movement of snake using a given action

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        #clock_wise is a list
            #represents clockwise order of directions
            #right, down, left, up
        #to determine new direction using current direction
        #idx will store index of current direction

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d
        #interpretting the action from the user
        #uses one-hot coding
        #[1,0,0] for no change in direction
        #[0,1,0] for turning right
        #[0,0,1] for turning left
        #based on action -- determines new direction
        #so if we turn right
            #new_dir is the next direction in clockwise order
        #if left
            #new direc is the previous direction in clockwise order

        self.direction = new_dir
        #updates snake's direction to new direction

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
        #updating position of snake head using direction
        #incrementing or decrementing using BLOCK_SIZE based on direction

        self.head = Point(x, y)
        #update position of snake's head with the new coordinates
        #as a whole -- handles movement of the snake
            #updating direction and position using given action