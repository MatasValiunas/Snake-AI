import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class SnakeEnv(gym.Env):
    _scale = 50                  # The size of each grid cell in pixels
    _snake_speed = 50            # The speed of the snake (FPS)

    def __init__(self, render_mode=False, size=18):
        self.render_mode = render_mode               # The render mode
        self.size = size                             # The size of the square grid
        self.window_size = size * self._scale        # The size of the PyGame window

        # The observation space is a box of 12 elements: [obst_left, obst_straight, obst_right, food_left, food_straight, food_right, food_back]
        self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=float)  

        # The action space is a discrete space of 3 elements: left, straight, right
        self.action_space = spaces.Discrete(3)
        self._action_to_direction = {
            0: np.array([[0, 1], [-1, 0]]),   # Left
            1: np.array([[1, 0], [0, 1]]),    # Straight
            2: np.array([[0, -1], [1, 0]])    # Right
        }

        if self.render_mode:
            pygame.init()
            pygame.display.set_caption('Snake')
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        self._walls = [pygame.Rect(0, 0, size, 1),                     # Top wall
                       pygame.Rect(0, size - 1, size, 1),              # Bottom wall
                       pygame.Rect(0, 0, 1, size),                     # Left wall
                       pygame.Rect(size - 1, 0, 1, size),              # Right wall
                       pygame.Rect(0, 0, 0, 0),                        # Place-holder for Random wall #1
                       pygame.Rect(0, 0, 0, 0)]                        # Place-holder for Random wall #2


    # Function to check if xy collides with walls (or snake)
    def _collision(self, xy, ignore_head=True, ignore_snake=False):
        for wall in self._walls:
            if wall.colliderect(pygame.Rect(xy, (1, 1))):
                return True
        
        if not ignore_snake:
            if ignore_head:
                for block in self._snake_body[1:]:
                    if np.array_equal(xy, block):
                        return True
            else:
                for block in self._snake_body:
                    if np.array_equal(xy, block):
                        return True
        return False

    # Function to generate a new position which does not collide with walls (or snake)
    def _new_position(self, ignore_snake=False):
        while True:
            xy = np.random.randint(1, self.size, size=(2,))

            if not self._collision(xy, ignore_head=False, ignore_snake=ignore_snake):
                return xy

    # Function to generate a new snake which does not collide with walls
    def _new_snake(self, lenght=4):
        self._snake_location = self._new_position(ignore_snake=True)    # Generates coordinates for snake's head
        self._snake_body = np.array([self._snake_location.copy()])      # Assigns snake's head to snake's body

        directions = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])       # Possible directions (left, up, right, down)

        for i in range(0, lenght-1):
            # Adds a new block to snake's body in direction if it does not collide with walls or itself
            for direction in directions:
                if not self._collision(self._snake_body[i] + direction, ignore_head=False):
                    self._snake_body = np.vstack([self._snake_body, self._snake_body[i] + direction])
                    break
            else:
                break

    # Function that determines if there's a snake body to the given direction             
    def _snake_to_side(self, direction):
        object = self._snake_location.copy()
        while True:
            object += direction

            for wall in self._walls:
                if wall.colliderect(pygame.Rect(object, (1, 1))):
                    return False

            for block in self._snake_body[2:]:
                if np.array_equal(object, block):
                    return True


    # Function to get the observations
    def _observation(self):
        snake_left, snake_right = False, False

        if self._direction[0] != 0:     # X-axis movement
            if self._direction[0] == 1:     # Right
                obst_left =  self._collision([self._snake_location[0], self._snake_location[1] - 1])
                obst_straight = self._collision([self._snake_location[0] + 1, self._snake_location[1]])
                obst_right = self._collision([self._snake_location[0], self._snake_location[1] + 1])
                obst_straight_left = self._collision([self._snake_location[0] + 1, self._snake_location[1] - 1])
                obst_straight_right = self._collision([self._snake_location[0] + 1, self._snake_location[1] + 1])

                food_left = self._target_location[1] < self._snake_location[1]
                food_right = self._target_location[1] > self._snake_location[1]
                food_straight = self._target_location[0] > self._snake_location[0]
                food_back = self._target_location[0] < self._snake_location[0]

                if obst_straight:
                    snake_left = self._snake_to_side([0, -1])
                    snake_right = self._snake_to_side([0, 1])
            
            else:                           # Left
                obst_left = self._collision([self._snake_location[0], self._snake_location[1] + 1])
                obst_straight = self._collision([self._snake_location[0] - 1, self._snake_location[1]])
                obst_right = self._collision([self._snake_location[0], self._snake_location[1] - 1])
                obst_straight_left = self._collision([self._snake_location[0] - 1, self._snake_location[1] + 1])
                obst_straight_right = self._collision([self._snake_location[0] - 1, self._snake_location[1] - 1])

                food_left = self._target_location[1] > self._snake_location[1]
                food_right = self._target_location[1] < self._snake_location[1]
                food_straight = self._target_location[0] < self._snake_location[0]
                food_back = self._target_location[0] > self._snake_location[0]

                if obst_straight:
                    snake_left = self._snake_to_side([0, 1])
                    snake_right = self._snake_to_side([0, -1])

        else:
            if self._direction[1] == 1:     # Down
                obst_left =  self._collision([self._snake_location[0] + 1, self._snake_location[1]])
                obst_straight = self._collision([self._snake_location[0], self._snake_location[1] + 1])
                obst_right = self._collision([self._snake_location[0] - 1, self._snake_location[1]])
                obst_straight_left = self._collision([self._snake_location[0] + 1, self._snake_location[1] + 1])
                obst_straight_right = self._collision([self._snake_location[0] - 1, self._snake_location[1] + 1])

                food_left = self._target_location[0] > self._snake_location[0]
                food_right = self._target_location[0] < self._snake_location[0]
                food_straight = self._target_location[1] > self._snake_location[1]
                food_back = self._target_location[1] < self._snake_location[1]

                if obst_straight:
                    snake_left = self._snake_to_side([1, 0])
                    snake_right = self._snake_to_side([-1, 0])
            
            else:                           # Up
                obst_left =  self._collision([self._snake_location[0] - 1, self._snake_location[1]])
                obst_straight = self._collision([self._snake_location[0], self._snake_location[1] - 1])
                obst_right = self._collision([self._snake_location[0] + 1, self._snake_location[1]])
                obst_straight_left = self._collision([self._snake_location[0] - 1, self._snake_location[1] - 1])
                obst_straight_right = self._collision([self._snake_location[0] + 1, self._snake_location[1] - 1])

                food_left = self._target_location[0] < self._snake_location[0]
                food_right = self._target_location[0] > self._snake_location[0]
                food_straight = self._target_location[1] < self._snake_location[1]
                food_back = self._target_location[1] > self._snake_location[1]

                if obst_straight:
                    snake_left = self._snake_to_side([-1, 0])
                    snake_right = self._snake_to_side([1, 0])

        lenght = len(self._snake_body) / (self.size * (self.size - 4))
        
        return [lenght, obst_left, obst_straight_left, obst_straight, obst_straight_right, obst_right, food_left, food_straight, food_right, food_back, snake_left, snake_right]


    # Function to reset (or start) the environment
    def reset(self):
        self._score = 0
        self._direction = np.array([1, 0])

        # Generates 2 random walls
        self._walls[-2] = pygame.Rect(np.random.randint(2, self.size-3, size=(2,)), (2, 2))  
        self._walls[-1] = pygame.Rect(np.random.randint(2, self.size-3, size=(2,)), (2, 2))   

        self._new_snake()

        #self._history = deque(maxlen=10)             # History of the snake's head
        #self._history.append(self._snake_location)   # Adding the snake's head coordinates to the history

        self._target_location = self._new_position()

        return self._observation(), {}

    # Function that is executed every step
    def step(self, action):
        self._direction = self._direction.dot(self._action_to_direction[action])   # Applying direction given by the model
        self._snake_location += self._direction                                    # Moving the snake

        # Elongating the snake
        self._snake_body = np.insert(self._snake_body, 0, self._snake_location, axis=0)
        if np.array_equal(self._snake_location, self._target_location):         # If snake eats fruit
            self._target_location = self._new_position()
            self._score += 10
            reward = 10
        else:
            self._snake_body = np.delete(self._snake_body, -1, axis=0)
            reward = 0

        # Checking if the snake has hit a wall or itself
        terminated = self._collision(self._snake_location)

        #if terminated: print("Score: ", self._score)           # Printing the score if the snake dies (useful for Testing.py)

        if self.render_mode:
            self.render()

        return self._observation(), reward, terminated, False, {} #self._stuck(), {}
    

    def render(self):
        self.window.fill(pygame.Color(0, 0, 0))

        # Drawing walls
        for wall in self._walls:
            pygame.draw.rect(self.window, (255, 100, 100), pygame.Rect(wall.x * self._scale, wall.y * self._scale, wall.width * self._scale, wall.height * self._scale))

        # Drawing snake
        green = 255
        for pos in self._snake_body[1:]:
            pygame.draw.rect(self.window, pygame.Color(0, green, 0), pygame.Rect(pos * self._scale, (self._scale, self._scale)))
            green = max(80, green - 15)
        pygame.draw.rect(self.window, pygame.Color(255, 0, 255), pygame.Rect(self._snake_location * self._scale, (self._scale, self._scale)))

        # Drawing fruit
        pygame.draw.rect(self.window, pygame.Color(255, 255, 255), pygame.Rect(self._target_location * self._scale, (self._scale, self._scale)))

        # creating font object score_font
        score_font = pygame.font.SysFont('verdana', int(self._scale * 0.9))
        # create the display surface object score_surface
        score_surface = score_font.render('Score: ' + str(self._score), True, pygame.Color(0, 0, 0))
        # displaying text
        self.window.blit(score_surface, (0, 0, 0, 0))

        # Updating the window
        pygame.display.update()
        pygame.time.Clock().tick(self._snake_speed)
     

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()