# reinforcement_face
import cv2
import numpy as np
import tensorflow as tf

from keras_facenet import FaceNet

from tensorflow.keras import layers, models

import random
from collections import deque

###########################
### PHẦN 1: MÔ PHỎNG MÔI TRƯỜNG ###
###########################


class FaceVerificationEnv:
    def __init__(self):

        # Load FaceNet model
        self.facenet = FaceNet()
        self.target_size = (160, 160)
        
        # Tạo dataset giả lập (ảnh tối + ảnh gốc)
        self.normal_images, self.dark_images = self._create_simulated_data()
        
        # Thiết lập action space
        self.action_space = [
            'no_action', 
            'adjust_brightness', 
            'histogram_equalization'
        ]
        self.n_actions = len(self.action_space)
        
    def _create_simulated_data(self):
       
        normal = [np.random.rand(160,160,3) for _ in range(100)]

        dark = [self._simulate_low_light(img) for img in normal]

        return normal, dark
    
    def _simulate_low_light(self, img):
       
        dark_img = img * 0.2 + np.random.normal(0, 0.05, img.shape)
        return np.clip(dark_img, 0, 1)
    
    def _preprocess(self, img, action):
        
        if action == 'adjust_brightness':
            img = cv2.convertScaleAbs(img, alpha=1.5, beta=40)
        elif action == 'histogram_equalization':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            img[:,:,0] = cv2.equalizeHist(img[:,:,0])
            img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
        return img
    
    def reset(self):
        # Chọn ngẫu nhiên 1 cặp ảnh
        idx = random.randint(0, len(self.normal_images)-1)
        self.current_dark = self.dark_images[idx]
        self.reference = self.normal_images[idx]
        return self._get_embedding(self.current_dark)
    
    def _get_embedding(self, img):
      
      
        img = (img * 255).astype('uint8')
        embeds = self.facenet.embeddings([img])
        return embeds[0]
    
    def step(self, action_idx):
        # Áp dụng action
        action = self.action_space[action_idx]
        processed_img = self._preprocess(self.current_dark.copy(), action)
        
        # Tính toán phần thưởng
        emb = self._get_embedding(processed_img)
        ref_emb = self._get_embedding(self.reference)
        similarity = np.dot(emb, ref_emb)
        
        reward = 10.0 if similarity > 0.7 else -5.0
        done = True
        return emb, reward, done

###########################
### PHẦN 2: DQN AGENT ###
###########################

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # 512 (kích thước embedding)
        self.action_size = action_size
        
        self.model = self._build_model()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def _build_model(self):
        model = models.Sequential([
            layers.Dense(64, input_dim=self.state_size, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))
        return model
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])
    
    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([x[0] for x in minibatch])
        targets = []
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state[np.newaxis], verbose=0)[0])
            target_f = self.model.predict(state[np.newaxis], verbose=0)
            target_f[0][action] = target
            targets.append(target_f[0])
            
        self.model.fit(states, np.array(targets), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

###########################
### PHẦN 3: HUẤN LUYỆN ###
###########################

env = FaceVerificationEnv()
agent = DQNAgent(state_size=512, action_size=env.n_actions)

n_episodes = 100
batch_size = 32

for e in range(n_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.memory.append((state, action, reward, next_state, done))
        total_reward += reward
        state = next_state
        
    agent.train(batch_size)
    print(f"Episode: {e+1}/{n_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

# Lưu model
agent.model.save('face_verification_rl.h5')
