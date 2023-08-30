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
        List<TrainingSample> _trainingSamples = new List<TrainingSample>();
        HashSet<long> _knownPositions = new HashSet<long>();

        private readonly object lockGuard = new object();

        public AIGym() { }

        public void AddBaseLineParticipant(GymParticipant participant)
        {
            _baseLines.Add(participant);
        }

        public void RemoveBaseLineParticipant(int id)
        {
            _baseLines.RemoveAll(participant => participant.id == id);
        }

        private List<TrainingSample> SelfPlay(GymParticipant agentSettings)
        {
            var gameSettings = new GameSettings();
            var gameController = new GameController(gameSettings);
            gameController.StartGame();

            var gameAgent = new CNNEstimator(true);
            gameAgent.LoadModel(agentSettings.agent.modelPath);
            gameAgent.InitFromState(gameController.gameBoard.GetBoardState(), Constants.CROSS_COLOR, Constants.CROSS_COLOR);

            while (!gameController.IsGameEnded())
            {
                while (gameAgent.GetCurrentPlayoutsCount() < agentSettings.agent.playoutsCount)
                {
                    gameAgent.EstimateOnce();
                }

                if (gameAgent.GetCurrentPlayoutsCount() >= agentSettings.agent.playoutsCount && gameController.gameBoard.GetCurrentTurnColor() == Constants.CROSS_COLOR)
                {
                    var color = gameController.gameBoard.GetCurrentTurnColor();
                    var bestMove = gameAgent.GetBestMove();
                    if (gameController.MakeMove(bestMove.X, bestMove.Y))
                    {
                        gameAgent.SelectNextNode(bestMove.X, bestMove.Y, color, false);
                    }
                }

                if (gameAgent.GetCurrentPlayoutsCount() >= agentSettings.agent.playoutsCount && gameController.gameBoard.GetCurrentTurnColor() == Constants.ZERO_COLOR)
                {
                    var color = gameController.gameBoard.GetCurrentTurnColor();
                    var bestMove = gameAgent.GetBestMove();
                    if (gameController.MakeMove(bestMove.X, bestMove.Y))
                    {
                        gameAgent.SelectNextNode(bestMove.X, bestMove.Y, color, false);
                    }
                }
            }

            if (gameController.gameBoard.GetGameState() == Constants.GameResult.WIN)
            {
                List<TrainingSample> result;
                lock (lockGuard)
                {
                    result = gameAgent.GetTrainingSamples(gameController.gameBoard.GetCurrentTurnColor(), ref _knownPositions);
                }

                return result;
            }

            if (gameController.gameBoard.GetGameState() == Constants.GameResult.TIE)
            {
                List<TrainingSample> result;
                lock (lockGuard)
                {
                    result = gameAgent.GetTrainingSamples(Constants.EMPTY_COLOR, ref _knownPositions);
                }

                return result;
            }

            return new List<TrainingSample>();
        }

        private int EstimationPlay(GymParticipant agent1, GymParticipant agent2)
        {
            var gameSettings = new GameSettings();
            gameSettings.firstAgent = agent1.agent;
            gameSettings.secondAgent = agent2.agent;

            gameSettings.firstAgent.isTraining = false;
            gameSettings.secondAgent.isTraining = false;

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

            if (gameController.gameBoard.GetGameState() == Constants.GameResult.WIN)
            {
                return gameController.gameBoard.GetCurrentTurnColor();
            }

            return 0;
        }

        public int TrainAgent(int startingGeneration)
        {
            GymParticipant currentAgent = new GymParticipant()
            {
                id = 1000,
                agent = new GameAgentSettings(EstimatorType.CNN, CNNHelper.GetCNNPathByGeneration(startingGeneration), 400, 1.1f)
            };

            // start selfplay session
            var watch = System.Diagnostics.Stopwatch.StartNew();

            
            const int TRAINING_GAMES_COUNT = 2000;
            const int BATCH_SIZE = 1024;
            const int EPOCHS = 1;
            const int OVERFIT_EXIT_INTERVAL = 100;
            const int LOG_INTERVAL = 10;
            const int THREADS_COUNT = 24;
            int currentOverfitIntervalIndex = 0;
            int currentLogIntervalIndex = 0;

            int runSamplesCount = 0;
            int newSamplesDiscovered = 0;
            for (int gameIndex = 0;;)
            {
                if (runSamplesCount > BATCH_SIZE * 10)
                {
                    Console.WriteLine("Target games count reached");
                    Console.WriteLine("Samples count: " + runSamplesCount);
                    Console.WriteLine("Known positions count: " + _knownPositions.Count);
                    break;
                }

                if (runSamplesCount > BATCH_SIZE && gameIndex > TRAINING_GAMES_COUNT)
                {
                    break;
                }

                if (currentOverfitIntervalIndex >= OVERFIT_EXIT_INTERVAL && runSamplesCount > 0)
                {
                    if (newSamplesDiscovered < 50)
                    {
                        //_knownPositions.Clear();
                        break;
                    }

                    newSamplesDiscovered = 0;
                    currentOverfitIntervalIndex = 0;
                }

                if (currentLogIntervalIndex >= LOG_INTERVAL)
                {
                    Console.WriteLine("Starting trainging game " + gameIndex);
                    Console.WriteLine("Samples count: " + runSamplesCount);
                    Console.WriteLine("Known positions count: " + _knownPositions.Count);

                    currentLogIntervalIndex = 0;
                }

                List<Task<List<TrainingSample>>> selfPlayTasks = new List<Task<List<TrainingSample>>>();
                for (int i = 0; i < THREADS_COUNT; i++)
                {
                    selfPlayTasks.Add(Task.Factory.StartNew(() =>
                    {
                        return SelfPlay(currentAgent);
                    }));
                }

                Task.WaitAll(selfPlayTasks.ToArray());

                for (int i = 0; i < THREADS_COUNT; i++)
                {
                    _trainingSamples.AddRange(selfPlayTasks[i].Result);
                    runSamplesCount += selfPlayTasks[i].Result.Count;
                    newSamplesDiscovered += selfPlayTasks[i].Result.Count;
                }

                gameIndex += THREADS_COUNT;
                currentOverfitIntervalIndex += THREADS_COUNT;
                currentLogIntervalIndex += THREADS_COUNT;
            }

            watch.Stop();
            Console.WriteLine("Self-play time: " + watch.ElapsedMilliseconds / 1000.0);

            NetworkTrainer.Train(startingGeneration, startingGeneration + 1, ref _trainingSamples, 0.2f, BATCH_SIZE, EPOCHS);

            //_baseLines.Add(currentAgent);

            //var newGenerationAgent = new GymParticipant()
            //{
            //    id = 10000,
            //    agent = new GameAgentSettings(EstimatorType.CNN, CNNHelper.GetCNNPathByGeneration(startingGeneration + 1), 2500, 1.0f)
            //};

            //Console.WriteLine("Starting estimation games for new network " + newGenerationAgent.agent.modelPath);
            //int[] wonGamesCount = new int[_baseLines.Count];
            //int[] losesGamesCount = new int[_baseLines.Count];
            //const int GYM_GAMES_COUNT = 4;
            //List<Task<int>> estimationPlayTasks = new List<Task<int>>();
            //for (int agentIndex = 0; agentIndex < _baseLines.Count; agentIndex++)
            //{
            //    wonGamesCount[agentIndex] = 0;
            //    for (int gameIndex = 0; gameIndex < GYM_GAMES_COUNT; gameIndex++)
            //    {
            //        bool newNetFirst = gameIndex % 2 == 0;

            //        var agent1 = newNetFirst ? newGenerationAgent : _baseLines[agentIndex];
            //        var agent2 = newNetFirst ? _baseLines[agentIndex] : newGenerationAgent;

            //        Console.WriteLine("Starting game estimation game " + gameIndex + " vs " + _baseLines[agentIndex].agent.modelPath);
            //        estimationPlayTasks.Add(Task.Factory.StartNew(() =>
            //        {
            //            return EstimationPlay(agent1, agent2);
            //        }));
            //    }
            //}

            //Task.WaitAll(estimationPlayTasks.ToArray());
            //for (int agentIndex = 0; agentIndex < _baseLines.Count; agentIndex++)
            //{
            //    for (int gameIndex = 0; gameIndex < GYM_GAMES_COUNT; gameIndex++)
            //    {
            //        var gameResult = estimationPlayTasks[agentIndex * GYM_GAMES_COUNT + gameIndex].Result;
            //        bool newNetFirst = gameIndex % 2 == 0;
            //        if (gameResult == Constants.CROSS_COLOR && newNetFirst)
            //        {
            //            wonGamesCount[agentIndex]++;
            //        }

            //        if (gameResult == Constants.CROSS_COLOR && !newNetFirst)
            //        {
            //            losesGamesCount[agentIndex]++;
            //        }

            //        if (gameResult == Constants.ZERO_COLOR && !newNetFirst)
            //        {
            //            wonGamesCount[agentIndex]++;
            //        }

            //        if (gameResult == Constants.ZERO_COLOR && newNetFirst)
            //        {
            //            losesGamesCount[agentIndex]++;
            //        }
            //    }
            //}

            //for (int i = 0; i < wonGamesCount.Length; i++)
            //{
            //    int draws = GYM_GAMES_COUNT - wonGamesCount[i] - losesGamesCount[i];
            //    Console.WriteLine("Winrate against " + CNNHelper.GetCNNGeneration(_baseLines[i].agent.modelPath) + ": " + wonGamesCount[i] + " " + losesGamesCount[i] + " " + draws);
            //}
            //_baseLines.RemoveAt(_baseLines.Count - 1);

            //_knownPositions.Clear();

            return startingGeneration + 1;
        }
    }
}
