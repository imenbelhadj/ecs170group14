import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Initialize Pygame
pygame.init()
# Load font for displaying score
font = pygame.font.Font('arial.ttf', 25)

# Define Direction enumeration for snake movement
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Define a Point named tuple for representing coordinates
Point = namedtuple('Point', 'x, y')

# Define RGB colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
GREEN = (0, 255, 0)

# Set the size of each block in the game grid
BLOCK_SIZE = 20
LARGE_BLOCK_SIZE = 30
# Set the speed of the game (frames per second)
SPEED = 40

# Snake game class
class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # Initialize display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.high_score = 0 
        # Reset the game state
        self.reset()

    # Reset the game state
    def reset(self):
        # Set initial direction of the snake
        self.direction = Direction.RIGHT

        # Initialize snake with three segments
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        # Reset score and place food
        self.score = 0
        self.food = None
        #### 
        #self.larger_food = None 
        self.larger_foods = [] 
        self._place_food()
        self.frame_iteration = 0

    # Place food at a random location on the grid
    
    #### regular food: control
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        # If food is placed on the snake, place it again
        if self.food in self.snake:
            self._place_food()
            
    ###larger food
        #largex = random.randint(0, (self.w-LARGE_BLOCK_SIZE)//LARGE_BLOCK_SIZE) * LARGE_BLOCK_SIZE
       # largey = random.randint(0, (self.h-LARGE_BLOCK_SIZE)//LARGE_BLOCK_SIZE) * LARGE_BLOCK_SIZE
       # self.larger_food = Point(largex,largey)
       # if self.larger_food in self.snake or self.larger_food == self.food:
        #   self._place_food()
        self.larger_foods = []  # Clear the list before adding new larger foods
        for _ in range(3):  # Number of larger fruits
            largex = random.randint(0, (self.w - LARGE_BLOCK_SIZE) // LARGE_BLOCK_SIZE) * LARGE_BLOCK_SIZE
            largey = random.randint(0, (self.h - LARGE_BLOCK_SIZE) // LARGE_BLOCK_SIZE) * LARGE_BLOCK_SIZE
            larger_food = Point(largex, largey)
            if larger_food in self.snake or larger_food == self.food or larger_food in self.larger_foods:
                self._place_food()
            else:
                self.larger_foods.append(larger_food)

        
            
    # def _is_colliding_with_large_food(self):
    # # Check if the snake's head collides with any part of the larger food
    #     head_rect = pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE)
    #     large_food_rect = pygame.Rect(self.larger_food.x, self.larger_food.y, LARGE_BLOCK_SIZE, LARGE_BLOCK_SIZE)
    #     return head_rect.colliderect(large_food_rect)
    
    def _is_colliding_with_large_food(self):
        # Check if the snake's head collides with any part of any larger food
        head_rect = pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE)
        for larger_food in self.larger_foods:
            large_food_rect = pygame.Rect(larger_food.x, larger_food.y, LARGE_BLOCK_SIZE, LARGE_BLOCK_SIZE)
            if head_rect.colliderect(large_food_rect):
                return larger_food
        return None

            
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
            print(f"Ate regular food: New score: {self.score}")  # Debug statement
            self._place_food()
        else:
            larger_food = self._is_colliding_with_large_food()  # Change made here
            if larger_food:
                self.score += 3
                reward = 30
                print(f"Ate larger food: New score: {self.score}")  # Debug statement
                self.larger_foods.remove(larger_food)  # Change made here
                self._place_food()
            else:
                self.snake.pop()
        #handling whether snake has eaten food or not
        #if the head of the snake is in the same nplace as the food
        #then the snake will eat the food
        #score increments, and positive reward is given
        #then new food is placed
        #if snake doesn't eat the food, snake's tail falls off

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        #makes sure game runs at a consistent speed
        #refresh display and control frame rate

        # 6. return game over and score
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
    
         for larger_food in self.larger_foods:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(larger_food.x, larger_food.y, LARGE_BLOCK_SIZE, LARGE_BLOCK_SIZE))  # Draw the larger fruit

    #drawing the food pellets on the display
    #just a red rectangle at coordinates of self.food

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