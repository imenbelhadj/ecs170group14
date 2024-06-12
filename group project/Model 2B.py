#import libraries
import numpy as np
import matplotlib.pyplot as plt

# I worked on Google Collab environment initially, so there is no class names, etc

#Initialization part
row_size = 11
column_size = 11

Q_values = np.zeros((row_size, column_size, 4))

actions = ['up', 'right', 'down', 'left']

#Create a numpy array to hold the rewards for each state.  (2D)
rewards = np.full((row_size, column_size), -100.) # a function that returns a new array of a given shape and type, filled with a -100
rewards[0, 5] = 100. #set the reward for the food area, its going to be 100 when arrived that location

#define locations  where agent can travel (out of range indicates obstacles, where agent is not allowed to be)
squares = {} #store locations in a dictionary 
squares[0] = [i for i in range(11)] #basically agent can travel through columns 1,2,3....10
squares[1] = [i for i in range(1, 10)]  #basically agent can travel through columns 1,2,3....9
squares[2] = [1, 7, 9]  #basically agent can travel through these columns: 1,7,9
squares[3] = [i for i in range(1, 8)]
squares[4] = [3, 7]
squares[5] = [i for i in range(11)]
squares[6] = [5]
squares[7] = [i for i in range(1, 10)]
squares[8] = [3, 7]
squares[9] = [i for i in range(11)]
squares[10] = [i for i in range(11)]

#set the rewards for all  locations 
for row in range(1, 10):
  for column in squares[row]:
    rewards[row, column] = -1. #small punishment just wandering around without going after food

#print rewards matrix
for r in rewards:  #for row in rewards
  print(r)

#function that decides if the specified location is a terminal state
def is_terminal_state(current_row, current_column):
  #if the reward for this location is -1, then it is not a terminal state, its allowed
  if rewards[current_row, current_column] == -1.:
    return False
  else:
    return True

def start_location():
  current_row = np.random.randint(row_size)
  current_column = np.random.randint(column_size)

   # Loop until a non-terminal state is found

  while is_terminal_state(current_row, current_column):
    current_row = np.random.randint(row_size)
    current_column = np.random.randint(column_size)
  return current_row, current_column


  #an epsilon greedy algorithm that will choose the next action
def next_action(current_row, current_column, epsilon):
  if np.random.random() < epsilon:
    return np.argmax(Q_values[current_row, current_column])
  else: #choose a random action
    return np.random.randint(4)

#a function that gets the next location based on the action chosen by
def next_location(current_row, current_column, action_index):
  new_row = current_row
  new_column = current_column

  if actions[action_index] == 'up' and current_row > 0:
    new_row -= 1
  elif actions[action_index] == 'right' and current_column < column_size - 1:
    new_column += 1
  elif actions[action_index] == 'down' and current_row < row_size - 1:
    new_row += 1
  elif actions[action_index] == 'left' and current_column > 0:
    new_column -= 1
  return new_row, new_column


def plot_steps(steps): #to keep track how many steps does it need to find the fruit

    plt.plot(steps, label='Steps to Find Fruit')
    plt.xlabel('Game')
    plt.ylabel('Steps')
    plt.title('Steps to Find Fruit Over 10 Games')
    plt.legend()
    plt.show()


def plot_performance(episodes, rewards):  #to plot reward results
    plt.plot(episodes, rewards, label='Cumulative Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title('Performance of Training')
    plt.legend()
    plt.show()


def train_q_learning(epsilon=1.0, discount_factor=0.9, learning_rate=0.9, episodes=1000):
    episode_rewards = []
    epsilon_decay = epsilon / episodes #it will decrease later on
    
    for episode in range(episodes):
        row, column = start_location()
        cumulative_reward = 0 #initialize
        
        while not is_terminal_state(row, column):
        #calling next_action function, it will decide whether to explore(random) or exploit(best action)
            action_index = next_action(row, column, epsilon) 
            old_row, old_column = row, column  #current position agent is leaving from
            row, column = next_location(row, column, action_index)  #updates the new position agent is moved to
            reward = rewards[row, column] #retrieve the rewards based on the position

            cumulative_reward += reward
            old_q_value = Q_values[old_row, old_column, action_index] #to get the current Q value(action, state) pair
            
          # use temporal difference formula, which is the difference between the predicted future reward and the current Q-value.
            temporal_difference = reward + (discount_factor * np.max(Q_values[row, column])) - old_q_value
            new_q_value = old_q_value + (learning_rate * temporal_difference) #update the Q-value
            Q_values[old_row, old_column, action_index] = new_q_value #updated Q-value back into the Q-values matrix
        
        episode_rewards.append(cumulative_reward)  #keeps track the total reward

        epsilon = max(epsilon - epsilon_decay, 0.1) # we can start decaying epsilon after it searches the environment sometime (exploration-exploiation)
    

    plot_performance(range(episodes), episode_rewards)
    print('Training completed!')

def play_game(n_games = 10):  #define a function where agent will play 10 games by itself after training
      row, column = start_location()
      steps = 0

      while not is_terminal_state(row, column) and steps < 100:
          action_index = np.argmax(Q_values[row, column])
          row, column = next_location(row, column, action_index)
          steps += 1
          print(f"Step {steps}: Moved to ({row}, {column})")
          if rewards[row, column] == 100.:
              print("Fruit found!")
              break
      return steps

train_q_learning()


steps_to_find_fruit = []
for game in range(10):
    steps = play_game()
    steps_to_find_fruit.append(steps)
    print(f"The game {game + 1}: {steps} takes steps to find the fruit")

# Plot the steps taken to find the fruit for 10 games
plot_steps(steps_to_find_fruit)

