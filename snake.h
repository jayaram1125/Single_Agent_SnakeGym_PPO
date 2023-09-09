#ifndef SNAKE_ENV_H_
#define SNAKE_ENV_H_


#include <box2d/box2d.h>

#include <array>
#include <memory>
#include <random>
#include <vector>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <array>
#include <cmath>
#include <random>

using namespace std;

namespace box2d {


	class SnakeGameEnv
	{
		const double DEG2RAD = M_PI/180.0;
		const double RAD2DEG = 180.0/M_PI;

		const int LEFT = 0;
		const int RIGHT =1;
		const int UP = 2;
		const int DOWN = 3;

		const int COLOR_CODE_BLACK = 4;
		const int COLOR_CODE_BROWN = 5;
		const int COLOR_CODE_RED = 6;
		const int COLOR_CODE_GREEN = 7;
		const int COLOR_CODE_DARK_GREEN = 8;


		const double FRUIT_REWARD = 1;
		const double DEATH_REWARD = 1;
		const double MOVE_REWARD = 0.1; 

		const int GAME_ENV_SIZE = 400;

		const int BODY_WIDTH = 40;
		const int SCALE = 10;

		std::unique_ptr<b2World>m_world;

		b2Body*m_maze;

		int m_mazeCollisionBound_1;
		int m_mazeCollisionBound_2;

		vector<pair<int,int>>m_mazeIndices;

		bool m_isFruitEaten;

		int m_numMoves;

		unordered_set<int>m_playAreaSet;
		unordered_map<int,pair<int,int>>m_areaPosMap;

		int m_cols;

		unordered_set<int>m_headPosSet;
		unordered_set<int>m_bodyPosSet;

		int m_totalSize;

		protected:
			double m_reward;
			bool m_done;
			vector<vector<int>> m_obsPreFlatten;
			b2Body*m_fruit;
			b2Body*m_head;
			vector<b2Body*>m_body;
			bool m_isGameOver;
			int m_maskedDirection;
		private:
			void createMaze();
			optional<pair<int,int>> samplePositionFromPlayArea(vector<unordered_set<int>>list_occupied_area_sets);
			double sampleAngle();

			void createFruit();
			void destroyFruit();
			void moveFruitToAnotherLocation();
			
			void createSnake();
			void moveSnake(const int next_direction);
			void checkContact();
			void increaseSnakeLength();
			int findDirection(b2Body*unit);
			int findMaskedDirection();
			
			void destroySnake();

			vector<vector<int>> createObservations();
			int calculateShortestPathDistance(vector<vector<int>>obs,b2Vec2 destination,int numRows,int numCols,int passThroughUnitColor);

			int m_dcurrentFruit;
			int m_dprevFruit;
			unsigned int m_randomSeed;
			int m_rank;
		public:
			SnakeGameEnv(int rank,int env_id);
			~SnakeGameEnv();
			void SnakeGameEnvReset();
			void SnakeGameEnvStep(int action);
	};

}

#endif  // ENVPOOL_SNAKE_ENV_H_