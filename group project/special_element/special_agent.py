import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000  #total number of experiences the agent can remember
BATCH_SIZE = 1000   #how many of those experiences are used in each iteration
LR = 0.001  #it is common to set with low value

#there would be max_memory/batch_size = 100 iteartions in total, i mean after 100 iterations agent will be processed all experiences in its memory once.
class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # exploration-exploitation trade-off in RL, a value 0 represnts that doesn't explore (acts greedily) and always chooses the action with the highest predicted value.
        self.gamma = 0.9 # discount rate, future rewards were discounted by 10% per time
        self.memory = deque(maxlen=MAX_MEMORY) # popleft(), The deque will store the agent's experiences, with new experiences replacing the oldest ones when the deque reaches its maximum capacity.
        self.model = Linear_QNet(15, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    """""
    #Important Notes:
    1-  If the game environment changes significantly, more fruits, more obstacles, we should play round with gamma,
    A higher gamma value would prioritize long-term rewards more, encourage agent to choose bigger apples
    We may start with high epsilon value to encourage exploration, allowing the agent to discover new elements and learn about their effects. 
    As the agent gains more experience and knowledge about the environment, we can gradually decrease epsilon to favor exploitation and focus on maximizing rewards.
    
   2-  Reward Value: If we modify the reward to provide larger rewards for eating bigger fruits or achieving certain goals (e.g., collecting special elements), 
    we may need to adjust the rewards accordingly. such as increasing the  reward magnitude to reinforce desirable behaviors and discourage undesirable ones.
    Punishment: Similarly, if the punishment for crashing into obstacles is changed (e.g., larger punishment for crashing into bigger obstacles), 
    we may need to adjust the punishment magnitude accordingly. This would discourage the agent from colliding with obstacles and prioritize safer navigation strategies.
   
   3- Gamma and Epsilon: Introducing special elements that give the snake special powers would require careful consideration of both the discount factor (gamma) and exploration rate (epsilon). 
   The agent needs to learn how to interact with these special elements effectively, which may involve exploring different strategies and exploiting successful ones.
   
   4- Speed and Block Size: As the snake's score increases, and the game becomes more challenging, you may consider adjusting the game's speed and block size dynamically. 
   A higher score could trigger an increase in game speed or block size, providing a gradual increase in difficulty. This would require monitoring the agent's performance and adjusting the game parameters accordingly.
   
    """""


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
            game.food.y > game.head.y,  # food down

            # Obstacle location
            game.obstacle.x < game.head.x,  # obstacle left
            game.obstacle.x > game.head.x,  # obstacle right
            game.obstacle.y < game.head.y,  # obstacle up
            game.obstacle.y > game.head.y   # obstacle down
        ]

        return np.array(state, dtype=int)



    """""
    1- In order to change obstacle size and position of obstacles, we may implement the following:
    add additional features in the state representation to indicate the presence and characteristics of obstacles. 
    For example:Encode the positions of stationary obstacles as binary features indicating whether each grid cell contains an obstacle.
    For moving obstacles, encode their positions and velocities to track their movement patterns.
    Include features to distinguish between different types of obstacles based on their size or behavior (e.g., big/small, stationary/moving).
   

   2- Within the get_state function, calculate the rewards based on the current state and the agent's interactions with obstacles.
    Define different reward signals for collisions with stationary obstacles and moving obstacles, adjusting the penalty values accordingly to reflect the difficulty and consequences of colliding with each type.
    Consider providing rewards for successfully navigating around obstacles, collecting special elements that mitigate obstacles' effects, or achieving specific goals related to obstacle avoidance.
   
    """""

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    """
    You may need to adjust the way experiences are stored in memory (self.memory) to include information about the new game elements (e.g., obstacles, special elements).
    Ensure that the reward values are correctly recorded and stored along with other experience tuples.

    """
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    """"
    As this function samples experiences from memory for training, make sure it can handle the increased complexity of the game environment and reward structure.
    If the state representation or reward values have changed significantly, you may need to update the training process accordingly to ensure effective learning.
    """

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    """""
    Smilarly, ensure that the training step performed with short-term memory accounts for any changes in the game environment or reward structure.
    Check if the state-action-reward-next_state-done tuples provided to this function accurately reflect the current game state and reward values.
    """""

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    """""
    This function determines the agent's action based on the current game state.
    If the state representation has changed to include new game elements, ensure that the agent's action selection process considers these elements appropriately.
    Adjust the exploration-exploitation strategy (epsilon) based on the complexity of the game environment and the agent's learning progress.
    """""


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
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

    """""
Update the main training loop to handle the new game elements, reward structure, and agent behavior.
Ensure that the agent interacts correctly with the game environment, collects experiences, and trains its neural network based on the collected experiences.
If the game's dynamics change gradually (e.g., increasing speed and block size as the score increases), incorporate mechanisms to adjust the environment dynamically during training.
    """""

if __name__ == '__main__':
    train()

