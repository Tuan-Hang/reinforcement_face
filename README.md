# reinforcement_face
import numpy as np
import cv2
import os
import tensorflow as tf
from keras_vggface.vggface import VGGFace

from keras_vggface.utils import preprocess_input

from tensorflow.keras import layers, Model
from collections import deque
import random

# load Face embedding model
facenet = VGGFace(model='resnet50',
                  include_top=False,
                  input_shape=(160,160,3))

facenet.trainable = False

def get_embedding(image):
  image = preprocess_input(image,version=2)
  return facenet.predict(image[np.newaxis,...])[0]




###########################
### PHẦN 1: MÔ PHỎNG MÔI TRƯỜNG ###
###########################

# define Enviroment

class FaceVerificationEnv:
  def __init__(self, data_dir, num_actions=3) -> None:
    self.data_dir = data_dir
    self.num_actions = num_actions
    self.people = {}

    for person_dir in os.listdir(data_dir):
      person_path = os.path.join(data_dir, person_dir)
      if os.path.isdir(person_path):
        images = [os.path.join(person_path,f) for f in os.listdir(person_path) if f.endswith(('.jpg', '.png'))]
        if len(images)>1:
          self.people[person_dir]= images
    
    # the end of for
    self.person_ids = list(self.people.keys())
    if not self.person_ids:
      raise ValueError("Dataset is invalid")

  def reset(self):
    
    if np.random.rand()<0.5:


      person_id = np.random.choice(self.person_ids)
      image1, image2 = np.random.choice(self.people[person_id], 2, replace=False)
      self.gt_label =0
    else:
      person1, person2 = np.random.choice(self.person_ids, 2, replace=False)

      image1 = np.random.choice(self.people[person1])
      image2 = np.random.choice(self.people[person2])
      self.gt_label =1
    
    img1 = self.preprocess_image(image1, augment=False)
    img2 = self.preprocess_image(image2, augment=False)
    return img1, img2
  
  def preprocess_image(self, image_path, augment=True):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if augment:
      img = cv2.convertScaleAbs(img, alpha=0.3, beta=0)
      img = cv2.GaussianBlur(img, (5,5),0)
    return img

  def step(self, action):
    img1, img2 = self.reset()
    processed_img = self.apply_action(img2, action)

    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)

    similarity = np.dot(emb1,emb2)/ (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    reward = 1.0 if (similarity>0.6 and self.gt_label ==0) or (similarity <0.4 and self.gt_label=1) else -0.5

    return processed_img, reward, similarity

  def apply_action(self, img, action):

    if action==0:
      return cv2.equalizeHist(img)
    elif action==1:
      return cv2.detailEnhance(img, sigma_s=10, sigma_r = 0.15)
    elif action==2:
      return cv2.cvColor(img, cv2.COLOR_RGB2LAB)[...,0]

###########################
### PHẦN 2: DQN AGENT ###
###########################

class DQNAgent:

  def __init__(self, state_size=512, action_size=3) -> None:
    self.state_size = state_size
    self.action_size = action_size
    self.memory = deque(maxlen=2000)
    self.gama = 0.95
    self.epsilon_min =0.01
    self.epsilon_decay = 0.995
    self.model = self.build_model()
  

  def build_model(self):
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(self.state_size)),
        layers.Dense(64, activation='relu'),
        layers.Dense(self.action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer='adam')
    return model

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
  
  def act(self, state):

    if np.random.rand() <= self.epsilon:
      return np.random.choice(self.action_size)
    
    act_values = self.model.predict(state[np.newaxis, ...])

    return np.argmax(act_values[0])

  def replay(self, batch_size =32):

    minibatch = random.sample(self.memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
      target = reward
      if not done:
        target = reward + self.gama * np.amax(self.model.predict(next_state[np.newaxis,...])[0])
      target_f = self.model.predict(state[np.newaxis,...])
      target_f[0][action] = target

      self.model.fit(state[np.newaxis,...], target_f, epochs=1, verbose=0)

      if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay

###########################
### PHẦN 3: HUẤN LUYỆN ###
###########################

env= FaceVerificationEnv(data_dir='c:/dataset/images')
agent = DQNAgent(state_size=512, action_size=3)

num_episodes = 1000

batch_size = 32

for e in range(num_episodes):
  state = env.reset()
  state = get_embedding(state[0])
  total_reward =0
  done = False

  while not done:

    action = agent.act(state)

    next_state, reward, similarity = env.step(action)

    next_state_emb = get_embedding(next_state)

    agent.remember(state, action,reward, next_state_emb, done)

    total_reward += reward

    state = next_state_emb

    if len(agent.memory) > batch_size:
      agent.replay(batch_size)


      
  print(f"epoch: {e+1}, total reward: {total_reward} epsilon: {agent.epsilon:.2f}")
  

# Lưu model
agent.model.save('face_verification_rl.h5')
