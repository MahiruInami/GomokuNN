using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GomokuNN.Sources.Estimators;

namespace GomokuNN.Sources
{
    internal class GameController
    {
        public GameBoard gameBoard;

        public GameSettings gameSettings { get; private set; }

        public IGameEstimator? firstAgent = null;
        public IGameEstimator? secondAgent = null;

        public bool isGameStarted = false;
        public bool isGameInProgress = false;
        public bool isGameEnded = false;

        public GameController(GameSettings newGameSettings)
        {
            gameSettings = newGameSettings;
            gameBoard = new GameBoard(new ArrayGameBoardState(Constants.DEFAULT_BOARD_SIZE), Constants.CROSS_COLOR);
        }

        public void CreateEstimators()
        {
            firstAgent = CreateEstimator(gameSettings.firstAgent, Constants.CROSS_COLOR, Constants.CROSS_COLOR);
            secondAgent = CreateEstimator(gameSettings.secondAgent, Constants.CROSS_COLOR, Constants.ZERO_COLOR);
        }

        private IGameEstimator? CreateEstimator(GameAgentSettings agentSettings, int moveColor, int estimatorColor)
        {
            if (agentSettings.type == EstimatorType.NONE)
            {
                return null;
            }

            if (agentSettings.type == EstimatorType.CNN)
            {
                var agent = new CNNEstimator(agentSettings.isTraining);
                agent.LoadModel(agentSettings.modelPath);
                agent.InitFromState(gameBoard.GetBoardState(), moveColor, estimatorColor);

                return agent;
            }

            if (agentSettings.type == EstimatorType.MCTS)
            {
                var agent = new MCTSEstimator();
                agent.InitFromState(gameBoard.GetBoardState(), moveColor, estimatorColor);

                return agent;
            }

            if (agentSettings.type == EstimatorType.IDDFS)
            {
                var agent = new IDDFSEstimator();
                agent.InitFromState(gameBoard.GetBoardState(), moveColor, estimatorColor);

                return agent;
            }

            if (agentSettings.type == EstimatorType.RANDOM)
            {
                var agent = new RandomEstimator();
                agent.InitFromState(gameBoard.GetBoardState(), moveColor, estimatorColor);

                return agent;
            }

            return null;
        }

        public void StartGame()
        {
            isGameInProgress = true;
            isGameEnded = false;
            isGameStarted = true;

            firstAgent?.StartEstimation();
            secondAgent?.StartEstimation();
        }

        public bool IsGameEnded()
        {
            return gameBoard.GetGameState() != Constants.GameResult.IN_PROGRESS;
        }

        public bool MakeMove(int x, int y)
        {
            var color = gameBoard.GetCurrentTurnColor();
            if (gameBoard.MakeMove(x, y))
            {
                firstAgent?.SelectNextNode(x, y, color);
                secondAgent?.SelectNextNode(x, y, color);

                if (gameBoard.GetGameState() != Constants.GameResult.IN_PROGRESS)
                {
                    isGameInProgress = false;
                    isGameEnded = true;

                    firstAgent?.StopEstimation();
                    secondAgent?.StopEstimation();
                }

                return true;
            }

            return false;
        }
    }
}
