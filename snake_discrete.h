/*
 * Copyright 2022 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENVPOOL_BOX2D_SNAKE_DISCRETE_H_
#define ENVPOOL_BOX2D_SNAKE_DISCRETE_H_

#include "envpool/box2d/snake.h"
#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include<string>
#include <typeinfo>

using namespace std;

namespace box2d {

class SnakeDiscreteEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("rank"_.Bind(0));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs:fruit_position"_.Bind(Spec<int>({2})),
    "obs:head_position"_.Bind(Spec<int>({2})),
    "obs:body_positions"_.Bind(Spec<int>({63,2})),
    "obs:masked_direction"_.Bind(Spec<int>({1})));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<int>({-1}, std::tuple<int,int>{0, 3})));
  }
};

using SnakeDiscreteEnvSpec = EnvSpec<SnakeDiscreteEnvFns>;

class SnakeDiscreteEnv : public Env<SnakeDiscreteEnvSpec>,
                               public SnakeGameEnv {
 public:
  SnakeDiscreteEnv(const Spec& spec, int env_id)
      : Env<SnakeDiscreteEnvSpec>(spec, env_id),
        SnakeGameEnv(spec.config["rank"_],env_id){}

  bool IsDone() override { return m_isGameOver; }

  void Reset() override {
    SnakeGameEnvReset();
    WriteState();
  }

  void Step(const Action& action) override {
    int act = action["action"_];
    SnakeGameEnvStep(act);
    WriteState();
  }

 private:
  void WriteState() {
    State state = Allocate();
    float temp_reward = m_reward;
    state["reward"_] = temp_reward;
 
    //cout<<"Write state enter"<<endl;

    b2Vec2 fruitPos;
    if(m_fruit)
    {
      fruitPos = m_fruit->GetPosition();
      //cout<<fruitPos.x<<"*****fruitPos****"<<fruitPos.y<<endl;
    }
    
    b2Vec2 headPos;
    if(m_head)
    {
      headPos= m_head->GetPosition();
      //cout<<headPos.x<<"*headPos********"<<headPos.y<<endl;
    }
    
    //cout<<"m_body size is:"<<m_body.size()<<endl;

    state["obs:fruit_position"_](0) = int(fruitPos.x);
    state["obs:fruit_position"_](1) = int(fruitPos.y);

    state["obs:head_position"_](0) = int(headPos.x);
    state["obs:head_position"_](1) = int(headPos.y);

    for(int i = 0;i<m_body.size();i++)
    {
        b2Vec2 bodyPos;
        if(m_head)
        {
          bodyPos = m_body[i]->GetPosition();
        }
        //cout<<bodyPos.x<<"*bodyPos********"<<bodyPos.y<<endl;
        state["obs:body_positions"_](i)(0) = int(bodyPos.x);
        state["obs:body_positions"_](i)(1) = int(bodyPos.y);
    }

    for(int i = m_body.size();i<63;i++)
    {
        state["obs:body_positions"_](i)(0) = 0;
        state["obs:body_positions"_](i)(1) = 0;
    } 


    state["obs:masked_direction"_] = m_maskedDirection; 
    //cout<<"Write state exit"<<endl;

  }
};

using SnakeDiscreteEnvPool = AsyncEnvPool<SnakeDiscreteEnv>;

}  // namespace box2d

#endif  // ENVPOOL_BOX2_SNAKE_DISCRETE_H_
