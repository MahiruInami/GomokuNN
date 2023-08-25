using Raylib_cs;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal class AIGym
    {
        List<GymParticipant> _baseLines = new List<GymParticipant>();
        HashSet<long> _knownPositions = new HashSet<long>();

        public AIGym() { }

        public void AddBaseLineParticipant(GymParticipant participant)
        {
            _baseLines.Add(participant);
        }

        public void RemoveBaseLineParticipant(int id)
        {
            _baseLines.RemoveAll(participant => participant.id == id);
        }

        public int TrainAgent(int startingGeneration)
        {
            GymParticipant currentAgent = new GymParticipant()
            {
                id = 1,
                agent = new GameAgentSettings(EstimatorType.CNN, CNNHelper.GetCNNPathByGeneration(startingGeneration), 1, 2.0f)
            };

            // start selfplay session
            var samples = new List<TrainingSample>();
            const int TRAINING_GAMES_COUNT = 1000;
            for (int gameIndex = 0; gameIndex < TRAINING_GAMES_COUNT && samples.Count < 500000; gameIndex++)
            {
                var gameSettings = new GameSettings();

                gameSettings.firstAgent = new GameAgentSettings(currentAgent.agent.type, currentAgent.agent.modelPath, currentAgent.agent.playoutsCount, currentAgent.agent.explorationRate, true);
                gameSettings.secondAgent = new GameAgentSettings(currentAgent.agent.type, currentAgent.agent.modelPath, currentAgent.agent.playoutsCount, currentAgent.agent.explorationRate, true);

                if (gameIndex % 100 == 0)
                {
                    Console.WriteLine("Starting trainging game " + gameIndex);
                    Console.WriteLine("Samples count: " + samples.Count);
                    Console.WriteLine("Known positions count: " + _knownPositions.Count);
                }
                
                var gameController = new GameController(gameSettings);
                gameController.CreateEstimators();
                gameController.StartGame();

                while (!gameController.IsGameEnded())
                {
                    if (gameController.firstAgent != null && !gameController.firstAgent.HasContiniousEstimationSupport())
                    {
                        gameController.firstAgent.EstimateOnce();
                    }
                    if (gameController.secondAgent != null && !gameController.secondAgent.HasContiniousEstimationSupport())
                    {
                        gameController.secondAgent.EstimateOnce();
                    }

                    if (gameController.firstAgent?.GetCurrentPlayoutsCount() > gameController.gameSettings.firstAgent.playoutsCount && gameController.gameBoard.GetCurrentTurnColor() == Constants.CROSS_COLOR)
                    {
                        var bestMove = gameController.firstAgent.GetBestMove();
                        if (gameController.MakeMove(bestMove.X, bestMove.Y))
                        {
                        }
                    }

                    if (gameController.secondAgent?.GetCurrentPlayoutsCount() > gameController.gameSettings.secondAgent.playoutsCount && gameController.gameBoard.GetCurrentTurnColor() == Constants.ZERO_COLOR)
                    {
                        var bestMove = gameController.secondAgent.GetBestMove();
                        if (gameController.MakeMove(bestMove.X, bestMove.Y))
                        {
                        }
                    }
                }

                if (gameController.gameBoard.GetGameState() == Constants.GameResult.WIN && gameController.gameBoard.GetCurrentTurnColor() == Constants.CROSS_COLOR)
                {
                    var gameSamples = gameController.firstAgent.GetTrainingSamples(Constants.CROSS_COLOR, ref _knownPositions);
                    samples.AddRange(gameSamples);
                }

                if (gameController.gameBoard.GetGameState() == Constants.GameResult.WIN && gameController.gameBoard.GetCurrentTurnColor() == Constants.ZERO_COLOR)
                {
                    var gameSamples = gameController.secondAgent.GetTrainingSamples(Constants.ZERO_COLOR, ref _knownPositions);
                    samples.AddRange(gameSamples);
                }
            }

            var generation = CNNHelper.GetCNNGeneration(currentAgent.agent.modelPath);
            NetworkTrainer.Train(CNNHelper.GetCNNGeneration(currentAgent.agent.modelPath), generation + 1, ref samples, 0.2f, 32, 1);

            _baseLines.Add(currentAgent);

            var newGenerationAgent = new GymParticipant()
            {
                id = 10000,
                agent = new GameAgentSettings(EstimatorType.CNN, CNNHelper.GetCNNPathByGeneration(generation + 1), 2, 2.0f)
            };

            int[] wonGamesCount = new int[_baseLines.Count];
            const int GYM_GAMES_COUNT = 4;
            for (int agentIndex = 0; agentIndex < _baseLines.Count; agentIndex++)
            {
                wonGamesCount[agentIndex] = 0;
                for (int gameIndex = 0; gameIndex < GYM_GAMES_COUNT; gameIndex++)
                {
                    bool newNetFirst = gameIndex % 2 == 0;
                    var gameSettings = new GameSettings();
                    gameSettings.firstAgent = newNetFirst ? newGenerationAgent.agent : _baseLines[agentIndex].agent;
                    gameSettings.secondAgent = newNetFirst ? _baseLines[agentIndex].agent : newGenerationAgent.agent;

                    gameSettings.firstAgent.isTraining = false;
                    gameSettings.secondAgent.isTraining = false;

                    gameSettings.firstAgent.playoutsCount = 100;
                    gameSettings.secondAgent.playoutsCount = 100;

                    Console.WriteLine("Starting game estimation game " + gameIndex);
                    var gameController = new GameController(gameSettings);
                    gameController.CreateEstimators();
                    gameController.StartGame();

                    while (!gameController.IsGameEnded())
                    {
                        if (gameController.firstAgent != null && !gameController.firstAgent.HasContiniousEstimationSupport())
                        {
                            gameController.firstAgent.EstimateOnce();
                        }
                        if (gameController.secondAgent != null && !gameController.secondAgent.HasContiniousEstimationSupport())
                        {
                            gameController.secondAgent.EstimateOnce();
                        }

                        if (gameController.firstAgent?.GetCurrentPlayoutsCount() > gameController.gameSettings.firstAgent.playoutsCount && gameController.gameBoard.GetCurrentTurnColor() == Constants.CROSS_COLOR)
                        {
                            var bestMove = gameController.firstAgent.GetBestMove();
                            if (gameController.MakeMove(bestMove.X, bestMove.Y))
                            {
                            }
                        }

                        if (gameController.secondAgent?.GetCurrentPlayoutsCount() > gameController.gameSettings.secondAgent.playoutsCount && gameController.gameBoard.GetCurrentTurnColor() == Constants.ZERO_COLOR)
                        {
                            var bestMove = gameController.secondAgent.GetBestMove();
                            if (gameController.MakeMove(bestMove.X, bestMove.Y))
                            {
                            }
                        }
                    }

                    if (gameController.gameBoard.GetCurrentTurnColor() == Constants.CROSS_COLOR && newNetFirst)
                    {
                        wonGamesCount[agentIndex]++;
                    }

                    if (gameController.gameBoard.GetCurrentTurnColor() == Constants.ZERO_COLOR && !newNetFirst)
                    {
                        wonGamesCount[agentIndex]++;
                    }
                }
            }

            for (int i = 0; i < wonGamesCount.Length; i++)
            {
                float newGenerationWinrate = (float)wonGamesCount[i] / (float)GYM_GAMES_COUNT;
                Console.WriteLine("Winrate against " + CNNHelper.GetCNNGeneration(_baseLines[i].agent.modelPath) + ": " + newGenerationWinrate);
            }
            _baseLines.RemoveAt(_baseLines.Count - 1);
            _knownPositions.Clear();

            return generation + 1;
        }
    }
}
