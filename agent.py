import torch # pytorch library
import random
import numpy as np     # numpy library, for math operations
from collections import deque # effeicient memory management 
from game import SnakeGameAI, Direction, Point # call other files in folder
from model import Linear_QNet, QTrainer # deep Q learning 
from helper import plot  # custom plotting function
import matplotlib.pyplot as plt

MAX_MEMORY = 100_000
BATCH_SIZE = 1000  #batch size for training neural network
LR = 0.001   # learning rates (for neural network training)

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) # takes parameters and executes training through deep Q   

    def get_state(self, game):
        head = game.snake[0]  #initialize position of snake's head 
        point_l = Point(head.x - 20, head.y)#point to the left of snake head
        point_r = Point(head.x + 20, head.y) #point to the right of snake head
        point_u = Point(head.x, head.y - 20) # "" above
        point_d = Point(head.x, head.y + 20) # "" below

       #movement directions of snake: 

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        #Constructing the state vector (using vector to represent the snake's current state)

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
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x,# food right
            game.food.y < game.head.x, # food up
            game.food.y > game.head.x # food down
        ]

        return np.array(state, dtype=int)  # return the state as an array generated by numpy

    def remember(self, state, action, reward, next_state, done):    #Remembering the game experience in the agent's memory
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    #Training the neural network using experiences stored in the memory
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # randomly selected set of experiences to train the neural net 
        else:
            mini_sample = self.memory  #once batch size is reached, sample from the entire memory

        states, actions, rewards, next_states, dones = zip(*mini_sample) # "unzipping" batch size -> seperating factors of each experience into their respective list
        self.trainer.train_step(states, actions, rewards, next_states, dones)  #training samples taken
         # for state, action, reward, nexrt_state, done in mini_sample:
        # self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):  #training neural net on single 
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):  #determine what action to take based on current state
               # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games # adjust exploration factor(here it is epsilon, a parameter for RL) (determines balance between exploration and exploitation) 
        final_move = [0,0,0] # this list of 3 move choices by th agent will be edited until the final move 
        if random.randint(0, 200) < self.epsilon:  #choosing an action at random between 0-200
            move = random.randint(0, 2) # move either left, right, or (up/down)
            final_move[move] = 1 # chosen action set to 1 
               # if the epsilon value is >= the random integer:
        else:
            state0 = torch.tensor(state, dtype=torch.float)  # convert the state to a pytorch tesor of type float
            prediction = self.model(state0)  # get prediction of model for each action
            move = torch.argmax(prediction).item() # choose action with the highest predictive value 
            final_move[move] = 1 # chosen action set to 1 
        return final_move

def play_itself(agent, n_games=10):
    scores = []
    for _ in range(n_games):
        game = SnakeGameAI()
        score = 0
        while True:
            state = agent.get_state(game)
            final_move = agent.get_action(state)
            reward, done, score = game.play_step(final_move)
            if done:
                scores.append(score)
                break
    return scores

def train(n_games=200):  # change the number of iterations 
        # Create empty lists to keep track of scores

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
        #create an instance of the agent and the game env

    agent = Agent()
    game = SnakeGameAI()

    # begin training loop:

    while agent.n_games < n_games:  
                # get current state of the game 
        state_old = agent.get_state(game)
                # get action based on current state 
        final_move = agent.get_action(state_old)
                # perform move and get new state
        reward, done, score = game.play_step(final_move)    #apply new action to the game & recive reward, done, flag, & score
        state_new = agent.get_state(game)   #get new state of the game after action is performed
        # train short memory based in current experience
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
                # remember by storing the experience in the agent's memory for later training 
        agent.remember(state_old, final_move, reward, state_new, done)
        # if game over, train long memory & plot the result:
        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score # update the record if there is a new high score
                agent.model.save()   # save the model 

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # Update list of scores for plotting
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores) # plot the scores 

    scores_after_training = play_itself(agent, n_games=10) 
    print("Scores after training:", scores_after_training)
    plt.figure()
    plt.plot(scores_after_training)
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.title('Scores after Training')
    plt.show()

if __name__ == '__main__':
    train()  # Highlighted change: run the train function with the default parameter
