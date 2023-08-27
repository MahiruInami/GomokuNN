using GomokuNN.Sources.Estimators;
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
                id = 1000,
                agent = new GameAgentSettings(EstimatorType.CNN, CNNHelper.GetCNNPathByGeneration(startingGeneration), 20, 2.0f)
            };

            // start selfplay session
            var rnd = new Random();
            var watch = System.Diagnostics.Stopwatch.StartNew();

            var samples = new List<TrainingSample>();
            const int TRAINING_GAMES_COUNT = 2000;
            for (int gameIndex = 0; samples.Count < 40000; gameIndex++)
            {
                var gameSettings = new GameSettings();

                //gameSettings.firstAgent = new GameAgentSettings(currentAgent.agent.type, currentAgent.agent.modelPath, currentAgent.agent.playoutsCount, currentAgent.agent.explorationRate, true);
                //gameSettings.secondAgent = new GameAgentSettings(currentAgent.agent.type, currentAgent.agent.modelPath, currentAgent.agent.playoutsCount, currentAgent.agent.explorationRate, true);

                if (gameIndex % 10 == 0)
                {
                    Console.WriteLine("Starting trainging game " + gameIndex);
                    Console.WriteLine("Samples count: " + samples.Count);
                    Console.WriteLine("Known positions count: " + _knownPositions.Count);
                }
                var gameController = new GameController(gameSettings);
                gameController.StartGame();

                var agent = new CNNEstimator(true);
                agent.LoadModel(currentAgent.agent.modelPath);
                agent.InitFromState(gameController.gameBoard.GetBoardState(), Constants.CROSS_COLOR, Constants.CROSS_COLOR);

                int playoutsCount = 2;// rnd.Next(currentAgent.agent.playoutsCount);
                while (!gameController.IsGameEnded())
                {
                    //if (gameController.firstAgent != null && !gameController.firstAgent.HasContiniousEstimationSupport())
                    //{
                    //    gameController.firstAgent.EstimateOnce();
                    //}
                    //if (gameController.secondAgent != null && !gameController.secondAgent.HasContiniousEstimationSupport())
                    //{
                    //    gameController.secondAgent.EstimateOnce();
                    //}

                    //if (gameController.firstAgent?.GetCurrentPlayoutsCount() > gameController.gameSettings.firstAgent.playoutsCount && gameController.gameBoard.GetCurrentTurnColor() == Constants.CROSS_COLOR)
                    //{
                    //    var bestMove = gameController.firstAgent.GetBestMove();
                    //    if (gameController.MakeMove(bestMove.X, bestMove.Y))
                    //    {
                    //    }
                    //}

                    //if (gameController.secondAgent?.GetCurrentPlayoutsCount() > gameController.gameSettings.secondAgent.playoutsCount && gameController.gameBoard.GetCurrentTurnColor() == Constants.ZERO_COLOR)
                    //{
                    //    var bestMove = gameController.secondAgent.GetBestMove();
                    //    if (gameController.MakeMove(bestMove.X, bestMove.Y))
                    //    {
                    //    }
                    //}

                    agent.EstimateOnce();
                    if (agent.GetCurrentPlayoutsCount() > playoutsCount && gameController.gameBoard.GetCurrentTurnColor() == Constants.CROSS_COLOR)
                    {
                        var color = gameController.gameBoard.GetCurrentTurnColor();
                        var bestMove = agent.GetBestMove();
                        if (gameController.MakeMove(bestMove.X, bestMove.Y))
                        {
                            agent.SelectNextNode(bestMove.X, bestMove.Y, color, false);
                        }
                    }

                    if (agent.GetCurrentPlayoutsCount() > playoutsCount && gameController.gameBoard.GetCurrentTurnColor() == Constants.ZERO_COLOR)
                    {
                        var color = gameController.gameBoard.GetCurrentTurnColor();
                        var bestMove = agent.GetBestMove();
                        if (gameController.MakeMove(bestMove.X, bestMove.Y))
                        {
                            agent.SelectNextNode(bestMove.X, bestMove.Y, color, false);
                        }
                    }
                }

                if (gameController.gameBoard.GetGameState() == Constants.GameResult.WIN && gameController.gameBoard.GetCurrentTurnColor() == Constants.CROSS_COLOR)
                {
                    var gameSamples = agent.GetTrainingSamples(Constants.CROSS_COLOR, ref _knownPositions);
                    samples.AddRange(gameSamples);
                }

                if (gameController.gameBoard.GetGameState() == Constants.GameResult.WIN && gameController.gameBoard.GetCurrentTurnColor() == Constants.ZERO_COLOR)
                {
                    var gameSamples = agent.GetTrainingSamples(Constants.ZERO_COLOR, ref _knownPositions);
                    samples.AddRange(gameSamples);
                }
            }

            watch.Stop();
            Console.WriteLine("Self-play time: " + watch.ElapsedMilliseconds / 1000.0);

            NetworkTrainer.Train(startingGeneration, startingGeneration + 1, ref samples, 0.2f, 4096, 1);

            _baseLines.Add(currentAgent);

            var newGenerationAgent = new GymParticipant()
            {
                id = 10000,
                agent = new GameAgentSettings(EstimatorType.CNN, CNNHelper.GetCNNPathByGeneration(startingGeneration + 1), 100, 2.0f)
            };

            int[] wonGamesCount = new int[_baseLines.Count];
            int[] losesGamesCount = new int[_baseLines.Count];
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

                    if (gameController.gameBoard.GetCurrentTurnColor() == Constants.CROSS_COLOR && !newNetFirst)
                    {
                        losesGamesCount[agentIndex]++;
                    }

                    if (gameController.gameBoard.GetCurrentTurnColor() == Constants.ZERO_COLOR && !newNetFirst)
                    {
                        wonGamesCount[agentIndex]++;
                    }

                    if (gameController.gameBoard.GetCurrentTurnColor() == Constants.ZERO_COLOR && newNetFirst)
                    {
                        losesGamesCount[agentIndex]++;
                    }
                }
            }

            for (int i = 0; i < wonGamesCount.Length; i++)
            {
                int draws = GYM_GAMES_COUNT - wonGamesCount[i] - losesGamesCount[i];
                Console.WriteLine("Winrate against " + CNNHelper.GetCNNGeneration(_baseLines[i].agent.modelPath) + ": " + wonGamesCount[i] + " " + losesGamesCount[i] + " " + draws);
            }
            _baseLines.RemoveAt(_baseLines.Count - 1);
            _knownPositions.Clear();

            return startingGeneration + 1;
        }
    }
}
