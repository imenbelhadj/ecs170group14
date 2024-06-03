#import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
#from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        # Q: memory management- initialize a memory buffer with a maximum size to store experiences (state, action, reward, next state, done)
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        #self.model = Linear_QNet(11, 256, 3)
        #self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.q_table = {} # Q: initialize q-table

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        #return np.array(state, dtype=int)
        return state
    
	# Q: storing experiences- appends the current experience to the memory buffer, later used for training
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

	# Q: training with experiences- samples a batch of experiences from memory and uses them to train model
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
            
        for state, action, reward, next_state, done in mini_sample:
            self.update_q_table(state, action, reward, next_state, done)
        #states, actions, rewards, next_states, dones = zip(*mini_sample)
        #self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

	# Q: immediate training- trains the model with most recent experience, useful for learning from real-time
    def train_short_memory(self, state, action, reward, next_state, done):
        #self.trainer.train_step(state, action, reward, next_state, done)
        self.update_q_table(state, action, reward, next_state, done)
        
	# Q: epsilon-greedy policy- agent sometimes takes random actions (exploration) and sometimes best-known actions (exploitation)
    # Q: helps agent explore state space while exploiting known good policies
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            #final_move[move] = 1
        else:
            state_str = str(state)
            if state_str not in self.q_table:
                self.q_table[state_str] = [0, 0, 0]
            move = np.argmax(self.q_table[state_str])
            #state0 = torch.tensor(state, dtype=torch.float)
            #prediction = self.model(state0)
            #move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1

        return final_move
    
    def update_q_table(self, state, action, reward, next_state, done):
        state_str = str(state)
        next_state_str = str(next_state)
        
        if state_str not in self.q_table:
            self.q_table[state_str] = [0, 0, 0]

        if next_state_str not in self.q_table:
            self.q_table[next_state_str] = [0, 0, 0]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_str])

        self.q_table[state_str][action.index(1)] += LR * (target - self.q_table[state_str][action.index(1)])

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                #agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()