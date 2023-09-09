import envpool
import numpy as np
import gym
from getkey import getkey,keys
import os
import pygame

from video_recorder import VideoRecorder


BLACK = (0,0,0) #background 

DARK_GREEN = (0,100,0)#Snake head color
GREEN = (0, 255, 0) #Snake body color

RED = (255, 0, 0) #Fruit color
BROWN = (156,102,31) #Grid color


display_size = 400
scale = 10
body_width =40
fruit_position = None
head_position = None
bodies_position = []
frame =None

def render(fruit_position,head_position,body_positions):
    #running = True
    pygame.init()
    pygame.display.init()
    #window = pygame.display.set_mode((400,400))


    #while running == True:
    surf = pygame.Surface((display_size,display_size))
    surf.fill(BLACK)
    pygame.transform.scale(surf, (scale,scale))
    
    pygame.draw.rect(surf,BROWN,pygame.Rect(0,0,display_size,display_size),int(body_width))

    if fruit_position[0] !=0 and fruit_position[1] != 0:
        pygame.draw.rect(surf,RED,pygame.Rect(fruit_position[0]*scale-body_width/2,fruit_position[1]*scale-body_width/2,body_width,body_width))
    
    if head_position[0] !=0 and head_position[1] != 0:
        pygame.draw.rect(surf,DARK_GREEN,pygame.Rect(head_position[0]*scale-body_width/2,head_position[1]*scale-body_width/2,body_width,body_width))  

    for i in range(0,len(body_positions)):
        if body_positions[i][0] != 0 and body_positions[i][1] != 0 :
            pygame.draw.rect(surf,GREEN,pygame.Rect(body_positions[i][0]*scale-body_width/2,body_positions[i][1]*scale-body_width/2,body_width,body_width))  

 
    frame= np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2))

    return frame

    '''window.blit(surf, surf.get_rect())
    pygame.event.pump()
    pygame.display.update()

        
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            running = False'''    

def capped_cubic_video_schedule(episode_id: int) -> bool:
    #print("Enter capped_cubic_video_schedule")
    '''if self.update  < self.num_updates: 
        if episode_id < 1000:
            return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
        else:
            return episode_id % 1000 == 0
    else:'''
    return True 


envs = envpool.make_gym('SnakeDiscrete-v2',num_envs=1)

episodeid = 1
video_path = os.path.abspath(os.getcwd())+"/video-env-test/"+"Episode_"+str(episodeid)+".mp4"
vi_rec = VideoRecorder(video_path)

envs.reset()


lis = [np.array([0])]*10
i = 0
#action=-1
while(i<10):    
    '''key = getkey()
    if key == keys.LEFT:
        print('LEFT')
        action = 0 
    elif key == keys.RIGHT:
        print('RIGHT')
        action = 1
    elif key == keys.UP:
        print('UP')
        action = 2
    elif key == keys.DOWN:
        print('DOWN')
        action = 3'''

    obs,rewards,dones,infos= envs.step(lis[i])
    #print("obs=",obs)
    print("reward=",rewards) 
    '''print("is_game_over=",dones)
    print("infos=",infos)'''

    head_position = obs["head_position"][0]
    fruit_position = obs["fruit_position"][0]
    body_positions = obs["body_positions"][0]

    #print(type(fruit_position))
    print(head_position)
    #print(body_positions)

    frame=render(fruit_position,head_position,body_positions)

    if(capped_cubic_video_schedule(1)):
        vi_rec.capture_frame(frame)

    i=i+1
vi_rec.close()