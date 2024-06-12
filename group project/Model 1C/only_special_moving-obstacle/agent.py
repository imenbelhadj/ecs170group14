import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import matplotlib.pyplot as plt

MAX_MEMORY = 100_000  #total number of experiences the agent can remember
BATCH_SIZE = 1000   #how many of those experiences are used in each iteration
LR = 0.001  #it is common to set with low value


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0.05 # exploration-exploitation trade-off in RL, a value 0 represnts that doesn't explore (acts greedily) and always chooses the action with the highest predicted value.
        self.gamma = 0.9 # discount rate, future rewards were discounted by 10% per time
        self.memory = deque(maxlen=MAX_MEMORY) # popleft(), The deque will store the agent's experiences, with new experiences replacing the oldest ones when the deque reaches its maximum capacity.
        self.model = Linear_QNet(15, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


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

            game.special.x < game.head.x,  # food left
            game.special.x > game.head.x,  # food right
            game.special.y < game.head.y,  # food up
            game.special.y > game.head.y,  # food down

             # special location 

            # game.special.x < game.head.x,  # special left
            # game.special.x > game.head.x,  # special right
            # game.special.y < game.head.y,  # special up
            # game.special.y > game.head.y,  # special down

            # Obstacle location
            game.obstacle.x < game.head.x,  # obstacle left
            game.obstacle.x > game.head.x,  # obstacle right
            game.obstacle.y < game.head.y,  # obstacle up
            game.obstacle.y > game.head.y   # obstacle down
        ]

        return np.array(state, dtype=int)


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

def play_itself(agent, n_games=10):  #it will play 10 games by itself after training
    scores = []
    for i in range(n_games):
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


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while agent.n_games < 10:  #train for 200 games
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

# Play 10 games after training and plot the results
    scores_after_training = play_itself(agent, n_games=10)
    print("Scores after training:", scores_after_training)
    plt.figure()
    plt.plot(scores_after_training)
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.title('Scores after Training')
    plt.show()


if __name__ == '__main__':
    train()