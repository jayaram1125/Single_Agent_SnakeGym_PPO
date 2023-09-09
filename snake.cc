#include "envpool/box2d/snake.h"
#include <algorithm>
#include "envpool/box2d/utils.h"
#include <iterator>
#include <algorithm>
#include <cmath>
#include <queue>
#include <tuple>
#include <numeric>
#include <iostream>
#include <map>

using namespace std;

namespace box2d 
{

	SnakeGameEnv::SnakeGameEnv(int rank,int env_id)
	:m_world(new b2World(b2Vec2(0.0, 0.0))),
	m_maze(NULL),
	m_mazeCollisionBound_1(0),
	m_mazeCollisionBound_2(0),
	m_mazeIndices(),
	m_isFruitEaten(false),
	m_numMoves(0),
	m_playAreaSet(),
	m_areaPosMap(),
	m_cols(0),
	m_headPosSet(),
	m_bodyPosSet(),
	m_totalSize(GAME_ENV_SIZE/BODY_WIDTH),
	m_reward(0.0),
	m_done(false),
	m_obsPreFlatten(),
	m_fruit(NULL),
	m_head(NULL),
	m_body(),
	m_isGameOver(false),
	m_maskedDirection(-1),
	m_dcurrentFruit(0),
	m_dprevFruit(0),
	m_randomSeed(0),
	m_rank(rank)
	{
		
		createMaze();
		m_randomSeed =(rank+1)*(env_id+1)*5000;
	}


	SnakeGameEnv::~SnakeGameEnv()
	{
		//cout<<"destuctor called"<<endl;
		destroySnake();
		destroyFruit();
		m_world->DestroyBody(m_maze);
		m_maze = NULL;
	}

	void SnakeGameEnv::createMaze()
	{

		vector<b2Vec2>vertices_vec = {b2Vec2(0.0,0.0), b2Vec2(GAME_ENV_SIZE/SCALE,0.0), b2Vec2(GAME_ENV_SIZE/SCALE,GAME_ENV_SIZE/SCALE),b2Vec2(0.0,GAME_ENV_SIZE/SCALE),
					b2Vec2(0.0,0.0), b2Vec2(BODY_WIDTH/SCALE,BODY_WIDTH/SCALE),b2Vec2(GAME_ENV_SIZE/SCALE-BODY_WIDTH/SCALE,BODY_WIDTH/SCALE),b2Vec2(GAME_ENV_SIZE/SCALE-BODY_WIDTH/SCALE,GAME_ENV_SIZE/SCALE-BODY_WIDTH/SCALE),b2Vec2(BODY_WIDTH/SCALE,GAME_ENV_SIZE/SCALE-BODY_WIDTH/SCALE),b2Vec2(BODY_WIDTH/SCALE,BODY_WIDTH/SCALE)};

		b2Vec2*vertices = vertices_vec.data();

		b2ChainShape b2ChainShapeObj;

		b2ChainShapeObj.CreateLoop(vertices,10);

		m_mazeCollisionBound_1 = BODY_WIDTH/SCALE/2;
		m_mazeCollisionBound_2 = GAME_ENV_SIZE/SCALE-BODY_WIDTH/SCALE/2;


		b2BodyDef bd;
		bd.type = b2_staticBody;
		bd.position = b2Vec2(0.0f, 0.0f);
		bd.angle = 0.0;

	  	b2FixtureDef fd;
  		fd.shape = &b2ChainShapeObj;

		m_maze = m_world->CreateBody(&bd);
		m_maze->CreateFixture(&fd);


		int start = 0;
		int end = GAME_ENV_SIZE/BODY_WIDTH;

	

		//top side of maze
		for(int j=start;j<end ;j++)
			m_mazeIndices.push_back(make_pair(start,j));


		//left side of the maze
		int begin = start+1;
		for(int i=begin;i<end; i++)
			m_mazeIndices.push_back(make_pair(i,start));


		//bottom side of maze	
		for(int j=begin ;j<end ;j++)
			m_mazeIndices.push_back(make_pair(end-1,j));


		//right side of the maze
		for(int i=begin;i<end-1;i++)
			m_mazeIndices.push_back(make_pair(i,end-1)); 


		int playAreaBound_1 = BODY_WIDTH/SCALE + 2;
		int playAreaBound_2 = GAME_ENV_SIZE/SCALE-BODY_WIDTH/SCALE-2;

		m_cols = playAreaBound_2/4;



		for(int i=playAreaBound_1;i<playAreaBound_2+4;i=i+4)
		{
			for(int j = playAreaBound_1;j<playAreaBound_2+4;j=j+4)
			{
				//cout<<i<<","<<j;
				//i*self.cols+j helps to create unique key in the play area set
				m_playAreaSet.insert(i/4+(j/4)*m_cols);
				m_areaPosMap[i/4+(j/4)*m_cols] =make_pair(i,j);
			}
		}

	}

	optional<pair<int,int>> SnakeGameEnv::samplePositionFromPlayArea(vector<unordered_set<int>>occupiedAreaSetsList)
	{

		unordered_set<int> remainingAreaSet;
		std::copy_if(m_playAreaSet.begin(), m_playAreaSet.end(), inserter(remainingAreaSet, remainingAreaSet.begin()),
			[&occupiedAreaSetsList] (int val) { return occupiedAreaSetsList[0].count(val)==0;});

		
		for(int i =1;i<occupiedAreaSetsList.size();i++)
		{
			unordered_set<int> tempRemainingAreaSet;
			std::copy_if(remainingAreaSet.begin(), remainingAreaSet.end(), inserter(tempRemainingAreaSet, tempRemainingAreaSet.begin()),
				[&occupiedAreaSetsList,i] (int val) { return occupiedAreaSetsList[i].count(val)==0;});
			remainingAreaSet =tempRemainingAreaSet; 
		}

		/*cout<<"remaining_area_set="<<endl;
		for(auto a:remainingAreaSet)
			cout<<a<<"\t";
		cout<<endl;
		*/	
		optional<pair<int,int>>sampledPos;

		if(!remainingAreaSet.empty())
		{
			vector<int>sampleVec; 
			std::sample(remainingAreaSet.begin(), remainingAreaSet.end(), std::back_inserter(sampleVec),1,std::mt19937{std::random_device{}()});
			sampledPos = m_areaPosMap[sampleVec[0]];
			//cout<<"SAMPLED POSITION ="<<sampledPos.value().first<<":"<<sampledPos.value().second<<endl;
		}

		return sampledPos;
	}	 

	double SnakeGameEnv::sampleAngle()
    { 
		vector<double>angles = {0.0,M_PI/2,3*M_PI/2,M_PI};
		vector<double>sampledAngle; 
		std::sample(angles.begin(), angles.end(), std::back_inserter(sampledAngle),1,std::mt19937{std::random_device{}()});
		return sampledAngle[0];
	}
		

	void SnakeGameEnv::createFruit()
	{
		auto sampledPos = samplePositionFromPlayArea({unordered_set<int>()});
		b2PolygonShape b2PolygonShapeObj;
		b2PolygonShapeObj.SetAsBox(2,2);

		b2BodyDef bd;
		bd.type = b2_staticBody;
		bd.position = b2Vec2(sampledPos.value().first, sampledPos.value().second);
		bd.angle = 0.0;

		b2FixtureDef fd;
		fd.shape = &b2PolygonShapeObj;

		m_fruit = m_world->CreateBody(&bd);
		if(m_fruit)
		{
			m_fruit->CreateFixture(&fd);

			//cout<<"inside create fruit"<<endl;
			//auto fruitPos =m_fruit->GetPosition();
			//cout<<fruitPos.x<<":"<<fruitPos.y<<endl;
		}
		else
		{
			cout<<"Error:Fruit is null.Could not be created"<<endl;
		}
	}	


	void SnakeGameEnv::destroyFruit()
	{
		//cout<<"enter destroy fruit called"<<endl;
		m_world->DestroyBody(m_fruit);
		m_fruit = NULL;
	}

	void SnakeGameEnv::moveFruitToAnotherLocation()
	{
		//cout<<"Enter moveFruitToAnotherLocation"<<endl;
		auto sampledPos = samplePositionFromPlayArea({m_headPosSet,m_bodyPosSet});   
		if(sampledPos)
		{
			m_fruit->SetTransform(b2Vec2(sampledPos.value().first,sampledPos.value().second),0.0);
			//auto pos = m_fruit->GetPosition();
			//cout<<"fruit x="<<pos.x<<"fruit y="<<pos.y<<endl;
		}
		else
		{
			//cout<<"Else block in moveFruitToAnotherLocation"<<endl;
			//destroy fruit object when play area is filled with snake
			destroyFruit();
		}
		//cout<<"Exit moveFruitToAnotherLocation"<<endl;
	}

	void SnakeGameEnv::createSnake()
	{
		auto pos = m_fruit->GetPosition();
		unordered_set<int>fruitAreaSet;
		fruitAreaSet.insert(int(pos.x)/4+(int(pos.y)/4)*m_cols);

		auto sampledPos = samplePositionFromPlayArea({fruitAreaSet});	
		auto sampledAngle = sampleAngle();

		b2PolygonShape b2PolygonShapeObj;
		b2PolygonShapeObj.SetAsBox(2,2);

		b2BodyDef bd;
		bd.type = b2_staticBody;
		bd.position = b2Vec2(sampledPos.value().first, sampledPos.value().second);
		bd.angle = sampledAngle;

		b2FixtureDef fd;
		fd.shape = &b2PolygonShapeObj;

		m_head = m_world->CreateBody(&bd);
		if(m_head)
		{
			m_head->CreateFixture(&fd);
			auto headPos = m_head->GetPosition();
			m_headPosSet.insert(int(headPos.x)/4+(int(headPos.y)/4)*m_cols);
		}
		else
		{
			cout<<"Error:head could not be created"<<endl;
		}	
	}


	void SnakeGameEnv::moveSnake(const int nextDirection)
	{

		b2Vec2 headPosition;
		float headAngle =0.0;


		if(m_head)
		{
			headPosition = m_head->GetPosition();
			headAngle = m_head->GetAngle();
		

			auto prevHeadPosition = headPosition;
			auto prevHeadAngle = headAngle;

			if(nextDirection == UP && round(headAngle*RAD2DEG) != 270)
			{ 
				headPosition.y = headPosition.y - BODY_WIDTH/SCALE; 
				headAngle = M_PI/2;
			}
			else if(nextDirection == DOWN && round(headAngle*RAD2DEG) != 90)
			{ 
				headPosition.y = headPosition.y + BODY_WIDTH/SCALE;
				headAngle = 3*M_PI/2;
			}		
			else if(nextDirection == RIGHT && round(headAngle*RAD2DEG) != 180)
			{ 				 
				headPosition.x = headPosition.x + BODY_WIDTH/SCALE;
				headAngle = 0;
			}
			else if(nextDirection == LEFT && round(headAngle*RAD2DEG) != 0)
			{
				headPosition.x = headPosition.x-BODY_WIDTH/SCALE;
				headAngle = M_PI;
			}

			m_head->SetTransform(headPosition,headAngle);
			m_headPosSet.clear();
			m_headPosSet.insert(int(headPosition.x)/4+(int(headPosition.y)/4)*m_cols);

			vector<pair<b2Vec2,float>> prevbodyPosVec; 
		
			//Update the body positions only if the head is moved .Snake cannot move in opposite direction	
			if(prevHeadPosition != headPosition && m_body.size()>0)
			{	
				//cout<<"Enter that IF block"<<endl;	 	
				m_bodyPosSet.clear();

				for(int i = 0; i<m_body.size()-1;i++)
				{
					prevbodyPosVec.push_back(make_pair(m_body[i]->GetPosition(),m_body[i]->GetAngle()));
				}

				m_body[0]->SetTransform(prevHeadPosition,prevHeadAngle);
				auto p_pos = m_body[0]->GetPosition();
				m_bodyPosSet.insert(int(p_pos.x)/4+(int(p_pos.y)/4)*m_cols);

				for(int i = 1; i<m_body.size();i++)
				{
					m_body[i]->SetTransform(prevbodyPosVec[i-1].first,prevbodyPosVec[i-1].second);
					auto p_pos = m_body[i]->GetPosition();
					m_bodyPosSet.insert(int(p_pos.x)/4+(int(p_pos.y)/4)*m_cols);
				}
			}
	  		m_numMoves++;	
	  	}
	  	else
	  	{
	  		cout<<"Error:head is NULL.Could be destroyed earlier due  to an error"<<endl;
	  	}	

	}
		
	void SnakeGameEnv::checkContact()
	{


        int headPosX = 0;
        int headPosY = 0;

        int fruitPosX =0;
        int fruitPosY =0;
 

		if(m_head) 
		{
			auto headPos = m_head->GetPosition();
			headPosX = int(headPos.x);
			headPosY = int(headPos.y);
		}
		else
		{
			cout<<"head is NULL.Could be destroyed earlier due  to an error"<<endl;
		}

		if(m_fruit)
		{
			auto fruitPos = m_fruit->GetPosition();
			fruitPosX = int(fruitPos.x);
			fruitPosY = int(fruitPos.y);
		}
		else
		{
			cout<<"Fruit is NULL.Could be destroyed earlier due  to an error"<<endl;
		}

		bool snakeCollidedWithMaze = (headPosX == m_mazeCollisionBound_1 || headPosY == m_mazeCollisionBound_1 || headPosX  == m_mazeCollisionBound_2 or headPosY == m_mazeCollisionBound_2);

		

		//Checking contact with Maze or snake itself
		if(snakeCollidedWithMaze || m_bodyPosSet.count(headPosX/4+(headPosY/4)*m_cols))
		{	
			
			/*if(snakeCollidedWithMaze)
				cout<<"Snake collided with maze"<<endl;
			else
				cout<<"snake collided with itself"<<endl;*/

			destroySnake();
			m_isGameOver = true;
		}
		else if(fruitPosX == headPosX && fruitPosY == headPosY)
		{
			//cout<<"check4"<<endl;
			//Checking contact with fruit
			increaseSnakeLength();
			m_isFruitEaten = true;
			moveFruitToAnotherLocation();
		}

	}


	void SnakeGameEnv::increaseSnakeLength()
	{
		int newBodyUnitPosition_x = 0 ;
		int newBodyUnitPosition_y = 0 ;

		int lastUnitAngle = int(m_body.size()>0 ?round((m_body[m_body.size()-1]->GetAngle())*RAD2DEG):round((m_head->GetAngle())*RAD2DEG)); 
		
		b2Vec2 lastBodyPos;

		if(m_body.size()>0)
	 		lastBodyPos = m_body[m_body.size()-1]->GetPosition();
		auto headPos = m_head->GetPosition(); 

		int lastUnitPosition_x = int(m_body.size()>0?lastBodyPos.x:headPos.x);
		int lastUnitPosition_y = int(m_body.size()>0?lastBodyPos.y:headPos.y);

		int distanceDelta = BODY_WIDTH/SCALE;

		if(lastUnitAngle == 0)
		{
			newBodyUnitPosition_x = lastUnitPosition_x-distanceDelta;
			newBodyUnitPosition_y = lastUnitPosition_y;
		}	
		else if(lastUnitAngle == 90)
		{
			newBodyUnitPosition_x = lastUnitPosition_x;
			newBodyUnitPosition_y = lastUnitPosition_y+distanceDelta;
		}
		else if(lastUnitAngle == 180)
		{
			newBodyUnitPosition_x = lastUnitPosition_x+distanceDelta;
			newBodyUnitPosition_y = lastUnitPosition_y;
		}
		else if(lastUnitAngle == 270)
		{
			newBodyUnitPosition_x = lastUnitPosition_x;
			newBodyUnitPosition_y = lastUnitPosition_y-distanceDelta;
		}

		b2PolygonShape b2PolygonShapeObj;
		b2PolygonShapeObj.SetAsBox(2,2);

		b2BodyDef bd;
		bd.type = b2_staticBody;
		bd.position = b2Vec2(newBodyUnitPosition_x, newBodyUnitPosition_y);
		bd.angle = lastUnitAngle;

		b2FixtureDef fd;
		fd.shape = &b2PolygonShapeObj;

		auto bodyPtr = m_world->CreateBody(&bd);
		if(bodyPtr)
		{
			bodyPtr->CreateFixture(&fd);

			m_body.push_back(bodyPtr);
			m_bodyPosSet.insert(newBodyUnitPosition_x/4+(newBodyUnitPosition_y/4)*m_cols);
		}
		else
		{
			cout<<"bodyPtr is NULL.Body could not be created"<<endl;
		}
	
	}


	int SnakeGameEnv::findDirection(b2Body*unit)
	{

		int direction = -1;

		if(unit)
		{

			auto angle = unit->GetAngle();

			if(round(angle*RAD2DEG) == 270)
				direction = DOWN;
			else if(round(angle*RAD2DEG) == 90)
				direction = UP;
			else if(round(angle*RAD2DEG) == 180) 
				direction = LEFT;
			else if(round(angle*RAD2DEG) == 0)
				direction = RIGHT;
		}

		return direction;

	}


	int SnakeGameEnv::findMaskedDirection()
	{

		int maskedDirection = -1;

		if(m_head)
		{
			auto angle = m_head->GetAngle();

			if(round(angle*RAD2DEG) == 270)
				maskedDirection = UP;
			else if(round(angle*RAD2DEG) == 90)
				maskedDirection = DOWN;
			else if(round(angle*RAD2DEG) == 180)
				maskedDirection = RIGHT;
			else if(round(angle*RAD2DEG) == 0)
				maskedDirection = LEFT;
		}
		return maskedDirection;
	}


	void SnakeGameEnv::destroySnake()
	{

		if(m_head)
		{
			m_world->DestroyBody(m_head);
			m_head = NULL;
			m_headPosSet.clear();
		}
		for(int i=0 ;i<m_body.size();i++)
		{
			//cout<<"enter destroy body:"<<endl;
			m_world->DestroyBody(m_body[i]);
		}
		m_body.clear();
		m_bodyPosSet.clear();
		//cout<<"Snake Destroyed:"<<endl;	

	}

	vector<vector<int>> SnakeGameEnv::createObservations()
	{

		vector<vector<int>>obs(m_totalSize, vector<int>(m_totalSize, COLOR_CODE_BLACK));
		/*
		cout<<"Maze indices in create Observations function"
		for(int i=0 ;i<m_mazeIndices.size();i++)
		{
			cout<<m_mazeIndices[i].first<<m_mazeIndices[i].second<<endl;
		}
		*/
		for(int i=0 ;i<m_mazeIndices.size();i++)
			obs[m_mazeIndices[i].first][m_mazeIndices[i].second] = COLOR_CODE_BROWN;

		if(m_fruit != NULL)
		{
			auto fruitPos = m_fruit->GetPosition();
			obs[int(fruitPos.y)/4][int(fruitPos.x)/4] = COLOR_CODE_RED;
		}

		for(int i=0;i<m_body.size();i++)
		{
			if(m_body[i] != NULL)
			{
				auto bodyPos = m_body[i]->GetPosition();
				obs[int(bodyPos.y)/4][int(bodyPos.x)/4] = COLOR_CODE_GREEN;
			}
		}

		if(m_head != NULL)
		{
			auto headPos = m_head->GetPosition();
			obs[int(headPos.y)/4][int(headPos.x)/4] = COLOR_CODE_DARK_GREEN;
		}

		return obs;
	}


	int SnakeGameEnv::calculateShortestPathDistance(vector<vector<int>>obs, b2Vec2 destination,int numRows,int numCols,int passThroughUnitColor)
	{

		int originX = 0;
		int originY = 0;

		if(m_body.size()==0)
		{
			b2Vec2 headPos ;

			if(m_head)
			{
			 	headPos = m_head->GetPosition();
			}
			originX = int(headPos.y/4);
			originY = int(headPos.x/4); 
		}
		else
		{ 
			b2Vec2 tailPos ;

		 	tailPos = m_body[m_body.size()-1]->GetPosition();
			
			originX = int(tailPos.y/4);
			originY = int(tailPos.x/4); 

		}	
	
		int destinationX = int(destination.y)/4;
		int destinationY = int(destination.x)/4;

		if(destinationX==0 && destinationY==0)  //destination = fruit case : Fruit is destroyed when snake fills up play area
			return 0;                          



		tuple<int,int,int>source = make_tuple(originX, originY,0);    

		vector<vector<bool>>visited(numRows,vector<bool>(numCols,false));

		vector<vector<int>>delta ={{-1,0},{1,0},{0,-1},{0,1}};
               
		queue<tuple<int, int, int>> q;
		q.push(source);
		visited[get<0>(source)][get<1>(source)] = true;


		while(!q.empty())
		{
			tuple<int, int, int>s = q.front();
			q.pop();
 
        	//Fruit found,return its distance from snake head
			//cout<<obs[get<0>(s)][get<1>(s)];
			if (get<0>(s)==destinationX && get<1>(s) == destinationY)
				return get<2>(s);

			for(int i=0;i<delta.size();i++)
			{
				int x = get<0>(s)+delta[i][0];
				int y = get<1>(s)+delta[i][1];

				bool canPass = ((obs[x][y] == passThroughUnitColor) || (x == destinationX && y==destinationY));

				if(x>=0 && y>=0 && x<(numRows-1) && y<(numCols-1) && canPass && visited[x][y] == false)
				{
					q.push(make_tuple(x,y,get<2>(s)+1));
					visited[x][y] = true;
				}
			}
		}

		return -1;

	}

	void SnakeGameEnv::SnakeGameEnvReset()
	{
		//cout<<"SnakeGameEnvReset function enter called"<<endl;
		if(m_fruit)
		{
			destroyFruit();
			m_fruit = NULL;
		}

		if(m_head)	
			destroySnake();

		m_isGameOver = false;

		m_reward = 0.0;

		createFruit();

		createSnake();

		m_isFruitEaten = false;

		m_obsPreFlatten = createObservations();


		b2Vec2 fruitPos;
		if(m_fruit)
		{
			fruitPos = m_fruit->GetPosition();
		}
		m_dprevFruit=0;
		m_dcurrentFruit=calculateShortestPathDistance(m_obsPreFlatten,fruitPos,m_obsPreFlatten.size(),m_obsPreFlatten[0].size(),COLOR_CODE_BLACK);

		//m_maxLenLimitBeforeTurning=0;
		//SetCoilParamsToDefault();
		//m_visitedSet.clear();

		m_maskedDirection = findMaskedDirection();
		//cout<<"SnakeGameEnvReset function exit"<<endl;
	}
	

	void SnakeGameEnv::SnakeGameEnvStep(int action)
	{
		m_obsPreFlatten = createObservations();

		//SnakeGameEnvBeforeStep(m_obsPreFlatten);

		moveSnake(action);
		
		checkContact();

		m_obsPreFlatten = createObservations();

	
		if(m_head && m_fruit)
		{
			auto fruitPos = m_fruit->GetPosition();
			if(m_dcurrentFruit != -1)
			{
				m_dprevFruit = m_dcurrentFruit;
			}
			m_dcurrentFruit = calculateShortestPathDistance(m_obsPreFlatten,fruitPos,m_obsPreFlatten.size(),m_obsPreFlatten[0].size(),COLOR_CODE_BLACK);				
		}	
	

		m_reward = 0.0;

		if(m_isGameOver)
		{
			m_reward = DEATH_REWARD;
		}
		else if(m_isFruitEaten)
		{
			m_reward  = FRUIT_REWARD;
			m_isFruitEaten = false;
			m_numMoves = 0;
			//m_visitedSet.clear();
			//SetCoilParamsToDefault();
			if(m_body.size()>=63)
			{
				m_reward += 64;
				m_isGameOver =true;
			}

		}
		else if(m_numMoves>=100)
		{
			m_reward = DEATH_REWARD;
			m_isGameOver = true;
			m_numMoves = 0;
			//cout<<"reward=-1, check2"<<endl;
		}
		else
		{
			if(m_dcurrentFruit == -1 || m_dcurrentFruit>=m_dprevFruit)
			{
				m_reward -= MOVE_REWARD;	
			}
			else
			{
				m_reward += MOVE_REWARD;	
			}
		}
		m_maskedDirection = findMaskedDirection();
	}

}

