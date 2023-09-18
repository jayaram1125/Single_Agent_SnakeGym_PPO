Snake Gym implemented using Box2d and Envpool libraries

PPO(Proximal policy optimization) algorithm used to train the snake agent is adapted from OpenAI's Baselines and
https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

Training duration : 7hrs 17min  using 1 TeslaV100 GPU and 6 CPU machine 
<p>
    <img width="300" height="300" src="https://github.com/jayaram1125/Single_Agent_SnakeGym_PPO/assets/16265393/721cdc1c-8137-408e-aa88-b5e6a85ed599">
    <img width="400" height="400" src="https://github.com/jayaram1125/Single_Agent_SnakeGym_PPO/assets/16265393/19c9d899-1b85-4a53-bf96-5621391a3f5b" hspace="50" >
</p>


Note:Except the files mpi_pytorch.py, mpi_tools.py,ppo.py, video_recorder.py, snake_env_test.py , all other files have to be placed in the path envpool/envpool/box2d folder to build and install the environment <br/>
Commands used for envpool lib <br/>
1.To build : make bazel-build   in the path ~/envpool   <br/>
2.To install : pip install /home/jayaram/SnakeGame/envpool/dist/envpool-0.6.7-cp39-cp39-linux_x86_64.whl 
