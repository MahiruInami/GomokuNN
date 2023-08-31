using GomokuNN.Sources.Estimators;
using Raylib_cs;
using Spectre.Console;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal class AIGym
    {
        private readonly bool PLAY_VS_BASELINES_ENABLED = false;

        List<GymParticipant> _baseLines = new List<GymParticipant>();
        List<TrainingSample> _trainingSamples = new List<TrainingSample>();
        HashSet<long> _knownPositions = new HashSet<long>();
        Random _rnd = new Random();

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

            int playouts = agentSettings.agent.playoutsCount;
            while (!gameController.IsGameEnded())
            {
                while (gameAgent.GetCurrentPlayoutsCount() < playouts)
                {
                    gameAgent.EstimateOnce();
                }

                if (gameAgent.GetCurrentPlayoutsCount() >= playouts && gameController.gameBoard.GetCurrentTurnColor() == Constants.CROSS_COLOR)
                {
                    var color = gameController.gameBoard.GetCurrentTurnColor();
                    var bestMove = gameAgent.GetBestMove();
                    if (gameController.MakeMove(bestMove.X, bestMove.Y))
                    {
                        gameAgent.SelectNextNode(bestMove.X, bestMove.Y, color, false);
                    }

                    playouts = _rnd.Next(10, 100);
                }

                if (gameAgent.GetCurrentPlayoutsCount() >= playouts && gameController.gameBoard.GetCurrentTurnColor() == Constants.ZERO_COLOR)
                {
                    var color = gameController.gameBoard.GetCurrentTurnColor();
                    var bestMove = gameAgent.GetBestMove();
                    if (gameController.MakeMove(bestMove.X, bestMove.Y))
                    {
                        gameAgent.SelectNextNode(bestMove.X, bestMove.Y, color, false);
                    }

                    playouts = _rnd.Next(10, 100);
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
            AnsiConsole.MarkupLine("[chartreuse1]Training session for {0}[/]", startingGeneration);

            GymParticipant currentAgent = new GymParticipant()
            {
                id = 1000,
                agent = new GameAgentSettings(EstimatorType.CNN, CNNHelper.GetCNNPathByGeneration(startingGeneration), 100, 1.1f)
            };

            // start selfplay session
            var watch = System.Diagnostics.Stopwatch.StartNew();

            var progress = AnsiConsole.Progress();

            const int MAX_TRAINING_SAMPLES_COUNT = 2000000;

            const int TRAINING_GAMES_COUNT = 200;
            const int MAX_SAMPLES_PER_ITERATION = 200000;
            const int BATCH_SIZE = 1024;
            const int EPOCHS = 1;
            const int OVERFIT_EXIT_INTERVAL = 100;
            const int LOG_INTERVAL = 512;
            const int THREADS_COUNT = 48;
            const int TRAINING_SAMPLE_SIZE = BATCH_SIZE * 20;
            int currentOverfitIntervalIndex = 0;
            int currentLogIntervalIndex = 0;

            int gameIndex = 0;
            int runSamplesCount = 0;
            int newSamplesDiscovered = 0;
            bool selfPlayFinished = false;

            var progressIndex = 0;
            var progressTask = AnsiConsole.Progress().Columns(new ProgressColumn[]
            {
                new TaskDescriptionColumn(),    // Task description
                new ProgressBarColumn(),        // Progress bar
                new PercentageColumn(),         // Percentage
                new ElapsedTimeColumn()
            }).StartAsync(async ctx =>
            {
                // Define tasks
                var playedGamesProgressTask = ctx.AddTask("[yellow]Self-play[/]");
                playedGamesProgressTask.MaxValue(TRAINING_GAMES_COUNT);

                while (!ctx.IsFinished)
                {
                    // Simulate some work
                    await Task.Delay(250);

                    // Increment
                    if (progressIndex != gameIndex)
                    {
                        var diff = gameIndex - progressIndex;
                        playedGamesProgressTask.Increment((float)diff);
                        progressIndex = gameIndex;
                    }

                    if (selfPlayFinished)
                    {
                        playedGamesProgressTask.StopTask();
                    }
                }
            });

            for (gameIndex = 0;;)
            {
                
                if (runSamplesCount > MAX_SAMPLES_PER_ITERATION)
                {
                    AnsiConsole.MarkupLine("[green]Target games count reached[/]");
                    //AnsiConsole.MarkupLine("[green]New samples count: {0}[/]", runSamplesCount);
                    //AnsiConsole.MarkupLine("[green]Known positions count: {0}[/]",  _knownPositions.Count);
                    break;
                }

                if (runSamplesCount > BATCH_SIZE && gameIndex > TRAINING_GAMES_COUNT)
                {
                    break;
                }

                if (currentOverfitIntervalIndex >= OVERFIT_EXIT_INTERVAL && gameIndex > 0)
                {
                    if (newSamplesDiscovered < 50)
                    {
                        //_knownPositions.Clear();
                        break;
                    }

                    newSamplesDiscovered = 0;
                    currentOverfitIntervalIndex = 0;
                }

                //if (currentLogIntervalIndex >= LOG_INTERVAL)
                //{
                //    Console.WriteLine("Starting trainging game " + gameIndex);
                //    Console.WriteLine("Samples count: " + runSamplesCount);
                //    Console.WriteLine("Known positions count: " + _knownPositions.Count);

                //    currentLogIntervalIndex = 0;
                //}

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

            selfPlayFinished = true;
            watch.Stop();
            progressTask.Wait(1000);

            AnsiConsole.MarkupLine("[red]Self-play time: {0}[/]", watch.ElapsedMilliseconds / 1000.0);

            if (_trainingSamples.Count > MAX_TRAINING_SAMPLES_COUNT)
            {
                AnsiConsole.MarkupLine("[steelblue] Max samples amount reached: {0}. Reducing samples count...[/]", _trainingSamples.Count);

                int gamesCountToRemove = _trainingSamples.Count - MAX_TRAINING_SAMPLES_COUNT;
                for (int i = 0; i < gamesCountToRemove; i++)
                {
                    _knownPositions.Remove(_trainingSamples[i].positionHash);
                }
                _trainingSamples.RemoveRange(0, gamesCountToRemove);
            }

            AnsiConsole.MarkupLine("[cyan1] Total samples: {0}[/]", _trainingSamples.Count);

            var trainingSamples = new List<TrainingSample>();
            trainingSamples.AddRange(_trainingSamples.GetRange(0, _trainingSamples.Count));
            NetworkTrainer.Shuffle(ref trainingSamples);
            trainingSamples = trainingSamples.GetRange(0, Math.Min(trainingSamples.Count, TRAINING_SAMPLE_SIZE));

            AnsiConsole.MarkupLine("[cyan1] Traingin with samples: {0}[/]", trainingSamples.Count);
            NetworkTrainer.Train(startingGeneration, startingGeneration + 1, ref trainingSamples, 0.2f, BATCH_SIZE, EPOCHS);

            if (PLAY_VS_BASELINES_ENABLED)
            {
                _baseLines.Add(new GymParticipant()
                {
                    id = 1000,
                    agent = new GameAgentSettings(EstimatorType.CNN, CNNHelper.GetCNNPathByGeneration(startingGeneration), 400, 1.0f)
                });

                var newGenerationAgent = new GymParticipant()
                {
                    id = 10000,
                    agent = new GameAgentSettings(EstimatorType.CNN, CNNHelper.GetCNNPathByGeneration(startingGeneration + 1), 400, 1.0f)
                };

                AnsiConsole.MarkupLine("[yellow] Starting estimation games for new network {0}[/]", newGenerationAgent.agent.modelPath);
                int[] wonGamesCount = new int[_baseLines.Count];
                int[] losesGamesCount = new int[_baseLines.Count];
                const int GYM_GAMES_COUNT = 4;
                List<Task<int>> estimationPlayTasks = new List<Task<int>>();
                for (int agentIndex = 0; agentIndex < _baseLines.Count; agentIndex++)
                {
                    wonGamesCount[agentIndex] = 0;
                    for (int evaluationGameIndex = 0; evaluationGameIndex < GYM_GAMES_COUNT; evaluationGameIndex++)
                    {
                        bool newNetFirst = evaluationGameIndex % 2 == 0;

                        var agent1 = newNetFirst ? newGenerationAgent : _baseLines[agentIndex];
                        var agent2 = newNetFirst ? _baseLines[agentIndex] : newGenerationAgent;

                        //AnsiConsole.MarkupLine("Starting game estimation game {0} vs {1}", evaluationGameIndex, _baseLines[agentIndex].agent.modelPath);
                        estimationPlayTasks.Add(Task.Factory.StartNew(() =>
                        {
                            return EstimationPlay(agent1, agent2);
                        }));
                    }
                }

                Task.WaitAll(estimationPlayTasks.ToArray());
                for (int agentIndex = 0; agentIndex < _baseLines.Count; agentIndex++)
                {
                    for (int evaluationGameIndex = 0; evaluationGameIndex < GYM_GAMES_COUNT; evaluationGameIndex++)
                    {
                        var gameResult = estimationPlayTasks[agentIndex * GYM_GAMES_COUNT + evaluationGameIndex].Result;
                        bool newNetFirst = evaluationGameIndex % 2 == 0;
                        if (gameResult == Constants.CROSS_COLOR && newNetFirst)
                        {
                            wonGamesCount[agentIndex]++;
                        }

                        if (gameResult == Constants.CROSS_COLOR && !newNetFirst)
                        {
                            losesGamesCount[agentIndex]++;
                        }

                        if (gameResult == Constants.ZERO_COLOR && !newNetFirst)
                        {
                            wonGamesCount[agentIndex]++;
                        }

                        if (gameResult == Constants.ZERO_COLOR && newNetFirst)
                        {
                            losesGamesCount[agentIndex]++;
                        }
                    }
                }

                for (int i = 0; i < wonGamesCount.Length; i++)
                {
                    int drawsGamesCount = GYM_GAMES_COUNT - wonGamesCount[i] - losesGamesCount[i];
                    AnsiConsole.MarkupLine("Winrate against {0}: {1}-{2}-{3}", CNNHelper.GetCNNGeneration(_baseLines[i].agent.modelPath), wonGamesCount[i], losesGamesCount[i], drawsGamesCount);
                }
                _baseLines.RemoveAt(_baseLines.Count - 1);

                float draws = GYM_GAMES_COUNT - wonGamesCount[0] - losesGamesCount[0];
                if (((float)wonGamesCount[0] + draws) / GYM_GAMES_COUNT >= 0.51f)
                {
                    AnsiConsole.MarkupLine("[green]New generation created.[/]");
                    return startingGeneration + 1;
                }

                AnsiConsole.MarkupLine("[red]Reverting to current generation...[/]");
                return startingGeneration;
            }

            //_knownPositions.Clear();
            //_trainingSamples.Clear();

            AnsiConsole.MarkupLine("[yellow]New generation created without estimation.[/]");
            return startingGeneration + 1;
        }
    }
}
