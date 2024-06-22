# 參考了老師提供的資源以及GPT輔助完成
import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human") 
observation, info = env.reset(seed=42)
score = 0

for _ in range(1000):
   env.render()
   if observation[2] and observation[3] >  0:
     action = 1
   else:
     action = 0

   observation, reward, terminated, truncated, info = env.step(action)
   score += reward

   if terminated or truncated: 
      observation, info = env.reset()
      print('done, score=',score)
      score = 0

env.close()