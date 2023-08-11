using ImGuiNET;
using Keras;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Numpy;
using Numpy.Models;
using Python.Runtime;
using Raylib_cs;
using rlImGui_cs;
using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.PortableExecutable;
using System.Resources;
using System.Text;
using System.Threading;
using System.Xml;
using System.Xml.Linq;

namespace GomokuNN.Sources
{
    class GameApp
    {
        enum AgentType
        {
            MCTS = 0,
            CNN,
            RANDOM
        }

        private int _boardSize;
        private GameBoard _gameBoard;
        private bool _autoEstimation = false;
        private IGameEstimator _estimator;
        private Texture2D[] _textures;
        private Texture2D[] _helperTextures;

        private float _explorationRate = 1.0f;

        private float _agent1ExplorationRate = 1.0f;
        private float _agent2ExplorationRate = 1.0f;

        private bool _playForBlack = false;
        private bool _playForWhite = false;

        private bool _showAIBestMove = false;
        private bool _showAIProbabilities = false;

        private int _agent1PlayoutsBeforeMoveSelection = 100;
        private int _agent2PlayoutsBeforeMoveSelection = 100;

        private int _playoutsBeforeMoveSelection = 100;
        private int _simulationsCountPerFrame = 386;

        private int _trainingEpochCount = 10;

        private bool _isGameEnded = false;

        private bool _isTraining = false;

        private IGameEstimator _agent1, _agent2;
        private List<string> _gameNames = new List<string>();
        private int _currentGameIndex = 0;

        private GameHistory _currentGameHistory;
        private int _currentMoveIndex = 0;
        private List<TrainingSample> _trainingSamples = new List<TrainingSample>();

        private int _totalGames = 0;
        private int _agent1Won = 0;
        private int _agent2Won = 0;

        private static int network_latest_version = 5;
        private int _latestNetworkGeneration = network_latest_version;
        private int _agent1NetworkGeneration = network_latest_version;
        private int _agent2NetworkGeneration = network_latest_version;

        private int _gamesBeforeTraining = 50;

        private AgentType _agent1Type = AgentType.MCTS;
        private AgentType _agent2Type = AgentType.CNN;

        List<string> _loadedGames = new List<string>();

        Random _rndProvider = new Random();

        public GameApp()
        {
            //Keras.Initializer;
            _boardSize = Constants.DEFAULT_BOARD_SIZE;
            ZobristHash.Init(Constants.DEFAULT_BOARD_SIZE);

            _gameBoard = new GameBoard(new ArrayGameBoardState(_boardSize), Constants.CROSS_COLOR);
            
            _estimator = new MCTSEstimator();
            _estimator.InitFromState(_gameBoard.GetBoardState(), _gameBoard.GetCurrentTurnColor());

            _textures = new Texture2D[3];
            _helperTextures = new Texture2D[3];
        }

        private static void SetupPyEnv()
        {
            //string envPythonHome = @"C:\Python38\";
            //string envPythonLib = envPythonHome + "Lib\\;" + envPythonHome + @"Lib\site-packages\";
            //Environment.SetEnvironmentVariable("PYTHONHOME", envPythonHome, EnvironmentVariableTarget.Process);
            //Environment.SetEnvironmentVariable("PATH", envPythonHome + ";" + envPythonLib + ";" + Environment.GetEnvironmentVariable("PATH", EnvironmentVariableTarget.Machine), EnvironmentVariableTarget.Process);
            //Environment.SetEnvironmentVariable("PYTHONPATH", envPythonLib, EnvironmentVariableTarget.User);

            //Runtime.PythonDLL = @"C:\Python38\python38.dll";
            //PythonEngine.PythonHome = envPythonHome;
            //PythonEngine.PythonPath = Environment.GetEnvironmentVariable("PYTHONPATH");

            string pythonPath1 = @"C:\Python38\";
            string pythonPath2 = @"C:\Python38\site-packages\";

            Runtime.PythonDLL = @"C:\Python38\python38.dll";

            //Environment.SetEnvironmentVariable("PATH", pythonPath1, EnvironmentVariableTarget.Process);
            //Environment.SetEnvironmentVariable("PYTHONHOME", pythonPath1, EnvironmentVariableTarget.Process);
            //Environment.SetEnvironmentVariable("PYTHONPATH", pythonPath2, EnvironmentVariableTarget.Process);
            //Environment.SetEnvironmentVariable("PYTHONNET_PYDLL", @"/-/opt/anaconda3/lib/libpython3.9.dylib");
        }

        private static void SetupModel()
        {
            //var inputLayer = new Input(shape: (4, Constants.DEFAULT_BOARD_SIZE, Constants.DEFAULT_BOARD_SIZE), name: "input_layer");

            //BaseLayer netLayer = new Conv2D(128, (5, 5).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last", strides: (1, 1).ToTuple()).Set(inputLayer);
            //netLayer = new BatchNormalization().Set(netLayer);
            //netLayer = new Activation("relu").Set(netLayer);

            //var firstLayer = new Conv2D(128, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last", strides: (1, 1).ToTuple()).Set(netLayer);
            //firstLayer = new BatchNormalization().Set(firstLayer);
            //firstLayer = new Activation("relu").Set(firstLayer);
            //firstLayer = new Conv2D(128, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last", strides: (1, 1).ToTuple()).Set(firstLayer);
            //firstLayer = new BatchNormalization().Set(firstLayer);
            //firstLayer = new Keras.Layers.Add(new BaseLayer[] { firstLayer, netLayer });
            //firstLayer = new Activation("relu").Set(firstLayer);

            //var secondLayer = new Conv2D(128, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last", strides: (1, 1).ToTuple()).Set(firstLayer);
            //secondLayer = new BatchNormalization().Set(secondLayer);
            //secondLayer = new Activation("relu").Set(secondLayer);
            //secondLayer = new Conv2D(128, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last", strides: (1, 1).ToTuple()).Set(secondLayer);
            //secondLayer = new BatchNormalization().Set(secondLayer);
            //secondLayer = new Keras.Layers.Add(new BaseLayer[] { secondLayer, firstLayer });
            //secondLayer = new Activation("relu").Set(secondLayer);

            //var thirdLayer = new Conv2D(128, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last", strides: (1, 1).ToTuple()).Set(secondLayer);
            //thirdLayer = new BatchNormalization().Set(thirdLayer);
            //thirdLayer = new Activation("relu").Set(thirdLayer);
            //thirdLayer = new Conv2D(128, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last", strides: (1, 1).ToTuple()).Set(thirdLayer);
            //thirdLayer = new BatchNormalization().Set(thirdLayer);
            //thirdLayer = new Keras.Layers.Add(new BaseLayer[] { thirdLayer, secondLayer });
            //thirdLayer = new Activation("relu").Set(thirdLayer);

            //var forthLayer = new Conv2D(128, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last", strides: (1, 1).ToTuple()).Set(thirdLayer);
            //forthLayer = new BatchNormalization().Set(forthLayer);
            //forthLayer = new Activation("relu").Set(forthLayer);
            //forthLayer = new Conv2D(128, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last", strides: (1, 1).ToTuple()).Set(forthLayer);
            //forthLayer = new BatchNormalization().Set(forthLayer);
            //forthLayer = new Keras.Layers.Add(new BaseLayer[] { forthLayer, thirdLayer });
            //forthLayer = new Activation("relu").Set(forthLayer);

            //var fifthLayer = new Conv2D(128, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last", strides: (1, 1).ToTuple()).Set(forthLayer);
            //fifthLayer = new BatchNormalization().Set(fifthLayer);
            //fifthLayer = new Activation("relu").Set(fifthLayer);
            //fifthLayer = new Conv2D(128, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last", strides: (1, 1).ToTuple()).Set(fifthLayer);
            //fifthLayer = new BatchNormalization().Set(fifthLayer);
            //fifthLayer = new Keras.Layers.Add(new BaseLayer[] { fifthLayer, forthLayer });
            //fifthLayer = new Activation("relu").Set(fifthLayer);

            //var policyOutput = new Conv2D(2, (1, 1).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last", strides: (1, 1).ToTuple()).Set(fifthLayer);
            //policyOutput = new BatchNormalization().Set(policyOutput);
            //policyOutput = new Activation("relu").Set(policyOutput);
            //policyOutput = new Flatten().Set(policyOutput);
            //policyOutput = new Dense(Constants.DEFAULT_BOARD_SIZE * Constants.DEFAULT_BOARD_SIZE, activation: "softmax", name: "policy_net").Set(policyOutput);

            //var valueOutput = new Conv2D(1, (1, 1).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last").Set(fifthLayer);
            //valueOutput = new BatchNormalization().Set(valueOutput);
            //valueOutput = new Activation("relu").Set(valueOutput);
            //valueOutput = new Flatten().Set(valueOutput);
            //valueOutput = new Dense(256, activation: "relu").Set(valueOutput);
            //valueOutput = new Dense(1, activation: "tanh", name: "value_net").Set(valueOutput);

            //var network = new Keras.Models.Model(new BaseLayer[] { inputLayer }, new BaseLayer[] { policyOutput, valueOutput }, name: "gomoku_net");
            //var network = Keras.Models.Model.LoadModel(String.Format("{0}_{1}_{2}.keras", MODEL_NAME, Constants.DEFAULT_BOARD_SIZE, network_latest_version));

            //network.Compile(optimizer: new Keras.Optimizers.Adam(learning_rate: 0.00001f),
            //    loss: new Dictionary<string, string> { { "policy_net", "categorical_crossentropy" }, { "value_net", "mean_squared_error" } },
            //    metrics: new Dictionary<string, string> { { "policy_net", "categorical_accuracy" }, { "value_net", "accuracy" } },
            //    weighted_metrics: new Dictionary<string, string> { { "policy_net", "categorical_accuracy" }, { "value_net", "accuracy" } },
            //    loss_weights: new Dictionary<string, float> { { "policy_net", 1.0f }, { "value_net", 1.0f } });
            ////network.Compile(optimizer: new Keras.Optimizers.Adam(learning_rate: 0.01f), loss: new string[] { "categorical_crossentropy", "mean_squared_error" }, metrics: new string[] { "categorical_accuracy", "accuracy" });

            //network.Summary();

            //network.Save(String.Format("{0}_{1}_{2}.keras", MODEL_NAME, Constants.DEFAULT_BOARD_SIZE, network_latest_version + 1));
        }

        private IGameEstimator createAgent(ref IGameEstimator previousAgent, AgentType agentType, bool isTraining, int networkGeneration, IGameBoardState state)
        {
            if (agentType == AgentType.CNN)
            {
                var agent = new CNNEstimator(false);
                agent.LoadModel(String.Format("{0}_{1}_{2}.keras", Constants.MODEL_NAME, Constants.DEFAULT_BOARD_SIZE, networkGeneration));
                agent.InitFromState(_gameBoard.GetBoardState(), Constants.CROSS_COLOR);

                return agent;
            }
            else if (agentType == AgentType.MCTS)
            {
                var agent = new MCTSEstimator();
                agent.InitFromState(_gameBoard.GetBoardState(), Constants.CROSS_COLOR);

                return agent;
            }
            else
            {

                var agent = new RandomEstimator();
                agent.InitFromState(_gameBoard.GetBoardState(), Constants.CROSS_COLOR);

                return agent;
            }
        }

        public void TrainNetworkOnSelfPlayData()
        {
            if (_agent1Type == AgentType.CNN && _agent2Type == AgentType.CNN)
            {
                if (_agent1Won > _agent2Won)
                {
                    NetworkTrainer.Train(_agent1NetworkGeneration, Math.Max(_agent1NetworkGeneration + 1, _agent2NetworkGeneration + 1), ref _trainingSamples, 0.1f, 32, _trainingEpochCount);

                    _latestNetworkGeneration = Math.Max(_agent1NetworkGeneration + 1, _agent2NetworkGeneration + 1);
                    _agent1NetworkGeneration = _latestNetworkGeneration;
                    _agent2NetworkGeneration = _latestNetworkGeneration;
                }
                else
                {
                    NetworkTrainer.Train(_agent2NetworkGeneration, Math.Max(_agent1NetworkGeneration + 1, _agent2NetworkGeneration + 1), ref _trainingSamples, 0.1f, 32, _trainingEpochCount);

                    _latestNetworkGeneration = Math.Max(_agent1NetworkGeneration + 1, _agent2NetworkGeneration + 1);
                    _agent1NetworkGeneration = _latestNetworkGeneration;
                    _agent2NetworkGeneration = _latestNetworkGeneration;
                }

            }
            else if (_agent1Type == AgentType.CNN)
            {
                NetworkTrainer.Train(_agent1NetworkGeneration, Math.Max(_agent1NetworkGeneration + 1, _agent2NetworkGeneration + 1), ref _trainingSamples, 0.1f, 32, _trainingEpochCount);

                _latestNetworkGeneration = Math.Max(_agent1NetworkGeneration + 1, _agent2NetworkGeneration + 1);
                _agent1NetworkGeneration = _latestNetworkGeneration;
                _agent2NetworkGeneration = _latestNetworkGeneration;
            }
            else if (_agent1Type == AgentType.CNN)
            {
                NetworkTrainer.Train(_agent2NetworkGeneration, Math.Max(_agent1NetworkGeneration + 1, _agent2NetworkGeneration + 1), ref _trainingSamples, 0.1f, 32, _trainingEpochCount);

                _latestNetworkGeneration = Math.Max(_agent1NetworkGeneration + 1, _agent2NetworkGeneration + 1);
                _agent1NetworkGeneration = _latestNetworkGeneration;
                _agent2NetworkGeneration = _latestNetworkGeneration;
            }
            else
            {
                NetworkTrainer.Train(_latestNetworkGeneration, _latestNetworkGeneration + 1, ref _trainingSamples, 0.1f, 32, _trainingEpochCount);

                _latestNetworkGeneration = _latestNetworkGeneration + 1;
                _agent1NetworkGeneration = _latestNetworkGeneration;
                _agent2NetworkGeneration = _latestNetworkGeneration;
            }
        }

        public void Run()
        {
            SetupPyEnv();
            SetupModel();

            Raylib.InitWindow(1280, 768, "TEST");

            rlImGui.Setup(true);
            ImGui.GetIO().ConfigFlags |= ImGuiConfigFlags.DockingEnable;
            ImGui.GetIO().ConfigFlags |= ImGuiConfigFlags.ViewportsEnable;

            //var crossTextureImage = Raylib.LoadImage("Resources/black_piece.png");
            _textures[Constants.CROSS_COLOR] = Raylib.LoadTexture("Resources/black_piece.png");
            _textures[Constants.ZERO_COLOR] = Raylib.LoadTexture("Resources/white_piece.png");
            _textures[Constants.EMPTY_COLOR] = Raylib.LoadTexture("Resources/empty_piece.png");

            _helperTextures[Constants.CROSS_COLOR] = Raylib.LoadTexture("Resources/black_piece_ai.png");
            _helperTextures[Constants.ZERO_COLOR] = Raylib.LoadTexture("Resources/white_piece_ai.png");
            _helperTextures[Constants.EMPTY_COLOR] = Raylib.LoadTexture("Resources/empty_piece.png");

            const int offsetX = Constants.BOARD_DRAW_OFFSET;
            const int offsetY = Constants.BOARD_DRAW_OFFSET;
            while (!Raylib.WindowShouldClose())
            {
                Raylib.BeginDrawing();
                Raylib.ClearBackground(Color.RAYWHITE);

                for (int y = 0; y < _gameBoard.GetBoardSize(); y++)
                {
                    for (int x = 0; x < _gameBoard.GetBoardSize(); x++)
                    {
                        int cellState = _gameBoard.GetCellState(x, y);
                        Raylib.DrawTexture(_textures[cellState], x * Constants.CELL_SIZE + offsetX, y * Constants.CELL_SIZE + offsetY, Color.WHITE);

                        if (_showAIProbabilities)
                        {
                            if (!_isTraining)
                            {
                                float probability = _estimator.GetMoveProbability(x, y);
                                Raylib.DrawText(probability.ToString("F2"), x * Constants.CELL_SIZE + offsetX + (int)(Constants.CELL_SIZE * 0.25f), y * Constants.CELL_SIZE + offsetY + (int)(Constants.CELL_SIZE * 0.4f), 12, Color.BLACK);
                            }
                            else
                            {
                                if (_gameBoard.GetCurrentTurnColor() == Constants.CROSS_COLOR)
                                {
                                    float probability = _agent1.GetMoveProbability(x, y);
                                    Raylib.DrawText(probability.ToString("F2"), x * Constants.CELL_SIZE + offsetX + (int)(Constants.CELL_SIZE * 0.25f), y * Constants.CELL_SIZE + offsetY + (int)(Constants.CELL_SIZE * 0.4f), 12, Color.BLACK);
                                }
                                else
                                {
                                    float probability = _agent2.GetMoveProbability(x, y);
                                    Raylib.DrawText(probability.ToString("F2"), x * Constants.CELL_SIZE + offsetX + (int)(Constants.CELL_SIZE * 0.25f), y * Constants.CELL_SIZE + offsetY + (int)(Constants.CELL_SIZE * 0.4f), 12, Color.BLACK);
                                }
                            }
                        }
                    }
                }

                if (_showAIBestMove && !_isGameEnded && !_isTraining)
                {
                    var bestMove = _estimator.GetBestMove();
                    if (bestMove.X >= 0)
                    {
                        Raylib.DrawTexture(_helperTextures[_gameBoard.GetCurrentTurnColor()], bestMove.X * Constants.CELL_SIZE + offsetX, bestMove.Y * Constants.CELL_SIZE + offsetY, Color.WHITE);
                    }
                }

                if (_showAIBestMove && !_isGameEnded && _isTraining)
                {
                    if (_gameBoard.GetCurrentTurnColor() == Constants.CROSS_COLOR)
                    {
                        var bestMove = _agent1.GetBestMove();
                        if (bestMove.X >= 0)
                        {
                            Raylib.DrawTexture(_helperTextures[_gameBoard.GetCurrentTurnColor()], bestMove.X * Constants.CELL_SIZE + offsetX, bestMove.Y * Constants.CELL_SIZE + offsetY, Color.WHITE);
                        }
                    }
                    else
                    {
                        var bestMove = _agent2.GetBestMove();
                        if (bestMove.X >= 0)
                        {
                            Raylib.DrawTexture(_helperTextures[_gameBoard.GetCurrentTurnColor()], bestMove.X * Constants.CELL_SIZE + offsetX, bestMove.Y * Constants.CELL_SIZE + offsetY, Color.WHITE);
                        }
                    }
                }


                if (!_isGameEnded)
                {
                    int textPosX = (_gameBoard.GetBoardSize() + 1) * Constants.CELL_SIZE + 10;
                    Raylib.DrawText("Current move", textPosX, 32, 24, Color.BLACK);
                    Raylib.DrawTexture(_textures[_gameBoard.GetCurrentTurnColor()], textPosX + 164, 30, Color.WHITE);
                }

                rlImGui.Begin();
                
                bool isOpen = true;
                ImGui.Begin("About Dear ImGui", ref isOpen, ImGuiWindowFlags.AlwaysAutoResize);


                if (ImGui.Button("Open games"))
                {
                    var currentDirectory = Directory.GetCurrentDirectory();
                    var games = Directory.GetFiles(currentDirectory + "\\Resources\\DataBase\\4");
                    _gameNames.Clear();
                    foreach (var gameName in games)
                    {
                        _gameNames.Add(gameName);
                    }

                    _currentGameIndex = 0;

                    _currentGameHistory = GameHistory.CreateFromPSQFile(_gameNames[_currentGameIndex]);
                    _currentMoveIndex = 0;
                    _gameBoard = new GameBoard(new ArrayGameBoardState(_boardSize), _currentGameHistory.startingColor);
                    foreach (var move in _currentGameHistory.moves)
                    {
                        if (!_gameBoard.MakeMove(move.X, move.Y))
                        {
                            Console.WriteLine("Invalid move while opening game: " + _gameNames[_currentGameIndex]);
                            break;
                        }

                        _currentMoveIndex++;
                    }
                }

                if (_gameNames.Count > 0)
                {
                    if (ImGui.Button("<<"))
                    {
                        _currentGameIndex--;
                        if (_currentGameIndex < 0) { _currentGameIndex = 0; }

                        _currentGameHistory = GameHistory.CreateFromPSQFile(_gameNames[_currentGameIndex]);
                        _currentMoveIndex = 0;
                        _gameBoard = new GameBoard(new ArrayGameBoardState(_boardSize), _currentGameHistory.startingColor);
                        foreach (var move in _currentGameHistory.moves)
                        {
                            if (!_gameBoard.MakeMove(move.X, move.Y))
                            {
                                Console.WriteLine("Invalid move while opening game: " + _gameNames[_currentGameIndex]);
                                break;
                            }

                            _currentMoveIndex++;
                        }
                    }
                    ImGui.SameLine();
                    if (ImGui.Button(">>"))
                    {
                        _currentGameIndex++;
                        if (_currentGameIndex >= _gameNames.Count) { _currentGameIndex = _gameNames.Count - 1; }

                        _currentGameHistory = GameHistory.CreateFromPSQFile(_gameNames[_currentGameIndex]);
                        _currentMoveIndex = 0;
                        _gameBoard = new GameBoard(new ArrayGameBoardState(_boardSize), _currentGameHistory.startingColor);
                        foreach (var move in _currentGameHistory.moves)
                        {
                            if (!_gameBoard.MakeMove(move.X, move.Y))
                            {
                                Console.WriteLine("Invalid move while opening game: " + _gameNames[_currentGameIndex]);
                                break;
                            }

                            _currentMoveIndex++;
                        }
                    }

                    ImGui.Text("Current Game: " + _gameNames[_currentGameIndex]);
                }

                if (ImGui.Button("Open game"))
                {
                    var currentDirectory = Directory.GetCurrentDirectory();
                    var gamePath = currentDirectory + "\\Resources\\DataBase\\4\\0_8_9_1.psq";
                    _currentGameHistory = GameHistory.CreateFromPSQFile(currentDirectory + "\\Resources\\DataBase\\4\\0_8_9_1.psq");

                    _currentMoveIndex = 0;
                    _gameBoard = new GameBoard(new ArrayGameBoardState(_boardSize), _currentGameHistory.startingColor);
                    foreach (var move in _currentGameHistory.moves)
                    {
                        if (!_gameBoard.MakeMove(move.X, move.Y))
                        {
                            Console.WriteLine("Invalid move while opening game: " + gamePath);
                            break;
                        }

                        _currentMoveIndex++;
                    }
                }

                if (_currentGameHistory != null)
                {
                    if (ImGui.Button("<"))
                    {
                        _currentMoveIndex--;
                        if (_currentMoveIndex < 0) {  _currentMoveIndex = 0; }

                        _gameBoard = new GameBoard(new ArrayGameBoardState(_boardSize), _currentGameHistory.startingColor);
                        for (int i = 0; i <= _currentMoveIndex; i++)
                        {
                            var move = _currentGameHistory.moves[i];
                            if (!_gameBoard.MakeMove(move.X, move.Y))
                            {
                                Console.WriteLine("Invalid move while opening game: ");
                                break;
                            }
                        }
                    }
                    ImGui.SameLine();
                    if (ImGui.Button(">"))
                    {
                        _currentMoveIndex++;
                        if (_currentMoveIndex >= _currentGameHistory.moves.Count) { _currentMoveIndex = _currentGameHistory.moves.Count - 1; }

                        _gameBoard = new GameBoard(new ArrayGameBoardState(_boardSize), _currentGameHistory.startingColor);
                        for (int i = 0; i <= _currentMoveIndex; i++)
                        {
                            var move = _currentGameHistory.moves[i];
                            if (!_gameBoard.MakeMove(move.X, move.Y))
                            {
                                Console.WriteLine("Invalid move while opening game: ");
                                break;
                            }
                        }
                    }

                    ImGui.Text("Winner: " + _currentGameHistory.winningColor + " ; " + _gameBoard.GetCurrentTurnColor());
                }

                if (ImGui.Button("Train on games DB"))
                {
                    HashSet<long> knownPositions = new HashSet<long>();
                    _trainingSamples.Clear();

                    var currentDirectory = Directory.GetCurrentDirectory();

                    XmlDocument gamesLib = new XmlDocument();
                    gamesLib.Load(currentDirectory + "\\Resources\\DataBase\\renjunet_v10_20230809.rif");

                    char[] MOVE_COORD = new char[] { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'};

                    var gamesData = gamesLib.DocumentElement.SelectSingleNode("games");
                    var games = gamesData.SelectNodes("game");
                    for (int i = 0; i < games.Count; i++)
                    {
                        var game = games[i];
                        float result = float.Parse(game.Attributes["bresult"].Value);

                        var movesNode = game.SelectSingleNode("move");
                        var movesStr = movesNode.InnerText;
                        string[] moves = movesStr.Split(' ');

                        if (moves.Length < 5)
                        {
                            continue;
                        }

                        var gameHistory = new GameHistory();
                        gameHistory.startingColor = Constants.CROSS_COLOR;
                        gameHistory.winningColor = result > 0.6f ? Constants.CROSS_COLOR : result < 0.4f ? Constants.ZERO_COLOR : Constants.EMPTY_COLOR;
                        gameHistory.result = gameHistory.winningColor == Constants.EMPTY_COLOR ? Constants.GameResult.TIE : gameHistory.startingColor == gameHistory.winningColor ? Constants.GameResult.WIN : Constants.GameResult.LOSE;
                        gameHistory.moves = new List<Constants.MovePosition>();

                        foreach (var move in moves)
                        {
                            int x = Array.IndexOf(MOVE_COORD, move[0]);
                            int y = int.Parse(move.Substring(1)) - 1;

                            gameHistory.moves.Add(new Constants.MovePosition(x, y));
                        }

                        TrainingSample.FillFromGameHistory(ref _trainingSamples, gameHistory, ref knownPositions);

                        if (_trainingSamples.Count > 2000000)
                        {
                            Console.WriteLine("Training with " + i + " lib game...");
                            Console.WriteLine("Board positions count: " + _trainingSamples.Count);

                            NetworkTrainer.Train(_latestNetworkGeneration, _latestNetworkGeneration + 1, ref _trainingSamples, 0.2f, 32, _trainingEpochCount);

                            _trainingSamples.Clear();

                            _latestNetworkGeneration = _latestNetworkGeneration + 1;
                            _agent1NetworkGeneration = _latestNetworkGeneration;
                            _agent2NetworkGeneration = _latestNetworkGeneration;
                        }
                    }

                    if (_trainingSamples.Count > 0)
                    {
                        Console.WriteLine("Board positions count: " + _trainingSamples.Count);

                        NetworkTrainer.Train(_latestNetworkGeneration, _latestNetworkGeneration + 1, ref _trainingSamples, 0.2f, 32, _trainingEpochCount);

                        _trainingSamples.Clear();

                        _latestNetworkGeneration = _latestNetworkGeneration + 1;
                        _agent1NetworkGeneration = _latestNetworkGeneration;
                        _agent2NetworkGeneration = _latestNetworkGeneration;
                    }
                }


                if (ImGui.Button("Train on game files"))
                {
                    HashSet<long> knownPositions = new HashSet<long>();
                    _trainingSamples.Clear();

                    var currentDirectory = Directory.GetCurrentDirectory();
                    for (int sample = 1; sample <= 10; sample++)
                    {
                        var games = Directory.GetFiles(currentDirectory + "\\Resources\\DataBase\\" + sample.ToString());
                        foreach (var gameName in games)
                        {
                            var gameHistory = GameHistory.CreateFromPSQFile(gameName);

                            const int HISTORIES_COUNT = 8;
                            var gameHistories = new GameHistory[HISTORIES_COUNT];
                            for (int gameIndex = 0; gameIndex < HISTORIES_COUNT; gameIndex++)
                            {
                                gameHistories[gameIndex] = new GameHistory();
                                gameHistories[gameIndex].moves = new List<Constants.MovePosition>();

                                gameHistories[gameIndex].startingColor = gameHistory.startingColor;
                                gameHistories[gameIndex].winningColor = gameHistory.winningColor;
                                gameHistories[gameIndex].result = gameHistory.result;

                                foreach (var move in gameHistory.moves)
                                {
                                    if (gameIndex == 0)
                                    {
                                        gameHistories[gameIndex].moves.Add(new Constants.MovePosition(move.X, move.Y));
                                    }
                                    else if (gameIndex == 1)
                                    {
                                        gameHistories[gameIndex].moves.Add(new Constants.MovePosition(15 - 1 - move.X, move.Y));
                                    }
                                    else if (gameIndex == 2)
                                    {
                                        gameHistories[gameIndex].moves.Add(new Constants.MovePosition(move.X, 15 - 1 - move.Y));
                                    }
                                    else if (gameIndex == 3)
                                    {
                                        gameHistories[gameIndex].moves.Add(new Constants.MovePosition(15 - 1 - move.X, 15 - 1 - move.Y));
                                    }
                                    else if (gameIndex == 4)
                                    {
                                        gameHistories[gameIndex].moves.Add(new Constants.MovePosition(move.Y, move.X));
                                    }
                                    else if (gameIndex == 5)
                                    {
                                        gameHistories[gameIndex].moves.Add(new Constants.MovePosition(15 - 1 - move.Y, move.X));
                                    }
                                    else if (gameIndex == 6)
                                    {
                                        gameHistories[gameIndex].moves.Add(new Constants.MovePosition(move.Y, 15 - 1 - move.X));
                                    }
                                    else if (gameIndex == 7)
                                    {
                                        gameHistories[gameIndex].moves.Add(new Constants.MovePosition(15 - 1 - move.Y, 15 - 1 - move.X));

                                    }
                                }
                            }

                            for (int gameIndex = 0; gameIndex < HISTORIES_COUNT; gameIndex++)
                            {
                                TrainingSample.FillFromGameHistory(ref _trainingSamples, gameHistories[gameIndex], ref knownPositions);
                            }

                            if (_trainingSamples.Count > 2000000)
                            {
                                Console.WriteLine("Training with " + sample + " sample...");
                                Console.WriteLine("Board positions count: " + _trainingSamples.Count);

                                NetworkTrainer.Train(_latestNetworkGeneration, _latestNetworkGeneration + 1, ref _trainingSamples, 0.2f, 32, _trainingEpochCount);

                                _trainingSamples.Clear();

                                _latestNetworkGeneration = _latestNetworkGeneration + 1;
                                _agent1NetworkGeneration = _latestNetworkGeneration;
                                _agent2NetworkGeneration = _latestNetworkGeneration;
                            }
                        }

                        if (_trainingSamples.Count > 2000000)
                        {
                            Console.WriteLine("Training with " + sample + " sample...");
                            Console.WriteLine("Board positions count: " + _trainingSamples.Count);

                            NetworkTrainer.Train(_latestNetworkGeneration, _latestNetworkGeneration + 1, ref _trainingSamples, 0.2f, 32, _trainingEpochCount);

                            _trainingSamples.Clear();

                            _latestNetworkGeneration = _latestNetworkGeneration + 1;
                            _agent1NetworkGeneration = _latestNetworkGeneration;
                            _agent2NetworkGeneration = _latestNetworkGeneration;
                        }
                    }

                    if (_trainingSamples.Count > 0)
                    {
                        Console.WriteLine("Board positions count: " + _trainingSamples.Count);

                        NetworkTrainer.Train(_latestNetworkGeneration, _latestNetworkGeneration + 1, ref _trainingSamples, 0.2f, 32, _trainingEpochCount);

                        _trainingSamples.Clear();

                        _latestNetworkGeneration = _latestNetworkGeneration + 1;
                        _agent1NetworkGeneration = _latestNetworkGeneration;
                        _agent2NetworkGeneration = _latestNetworkGeneration;
                    }
                }

                if (!_isTraining && ImGui.Button("Train"))
                {
                    _totalGames = 0;
                    _agent1Won = 0;

                    _isTraining = true;

                    Console.WriteLine("Starting new training game");

                    _currentGameHistory = new GameHistory();
                    _currentGameHistory.startingColor = Constants.CROSS_COLOR;
                    _currentGameHistory.moves = new List<Constants.MovePosition>();

                    _gameBoard = new GameBoard(new ArrayGameBoardState(_boardSize), Constants.CROSS_COLOR);

                    _agent1 = createAgent(ref _agent1, _agent1Type, true, _agent1NetworkGeneration, _gameBoard.GetBoardState());
                    _agent2 = createAgent(ref _agent2, _agent2Type, true, _agent2NetworkGeneration, _gameBoard.GetBoardState());
                }

                string[] agentTypes = new string[] { "MCTS", "CNN", "Random" };
                int agent1TypeInt = (int)_agent1Type;
                if (ImGui.BeginCombo("Agent 1 type", agentTypes[agent1TypeInt])) // The second parameter is the label previewed before opening the combo.
                {
                    for (int n = 0; n < agentTypes.Length; n++)
                    {
                        bool is_selected = (agent1TypeInt == n); // You can store your selection however you want, outside or inside your objects
                        if (ImGui.Selectable(agentTypes[n], is_selected)) {
                            agent1TypeInt = n;
                            _agent1Type = (AgentType)agent1TypeInt;
                        }
                        if (is_selected)
                            ImGui.SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
                    }
                    ImGui.EndCombo();
                }

                int agent2TypeInt = (int)_agent2Type;
                if (ImGui.BeginCombo("Agent 2 type", agentTypes[agent2TypeInt])) // The second parameter is the label previewed before opening the combo.
                {
                    for (int n = 0; n < agentTypes.Length; n++)
                    {
                        bool is_selected = (agent2TypeInt == n); // You can store your selection however you want, outside or inside your objects
                        if (ImGui.Selectable(agentTypes[n], is_selected))
                        {
                            agent2TypeInt = n;
                            _agent2Type = (AgentType)agent2TypeInt;
                        }
                        if (is_selected)
                            ImGui.SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
                    }
                    ImGui.EndCombo();
                }

                if (ImGui.SliderFloat("Agent 1 expand rate", ref _agent1ExplorationRate, 0.1f, 5.0f))
                {
                    if (_agent1 != null) _agent1.SetExplorationConst(_agent1ExplorationRate);
                }
                if (ImGui.SliderFloat("Agent 2 expand rate", ref _agent2ExplorationRate, 0.1f, 5.0f))
                {
                    if (_agent2 != null) _agent2.SetExplorationConst(_agent2ExplorationRate);
                }

                ImGui.Text("Latests generation: " + _latestNetworkGeneration);
                ImGui.SliderInt("Games per training", ref _gamesBeforeTraining, 1, 2000);

                if (_isTraining)
                {
                    if (_gameBoard.GetGameState() != Constants.GameResult.IN_PROGRESS)
                    {
                        if (_totalGames > _gamesBeforeTraining)
                        {
                            TrainNetworkOnSelfPlayData();

                            _agent1Won = 0;
                            _agent2Won = 0;
                            _totalGames = 0;
                        }
                        Console.WriteLine("Starting new training game");

                        _currentGameHistory = new GameHistory();
                        _currentGameHistory.startingColor = Constants.CROSS_COLOR;
                        _currentGameHistory.moves = new List<Constants.MovePosition>();

                        _gameBoard = new GameBoard(new ArrayGameBoardState(_boardSize), Constants.CROSS_COLOR);

                        _agent1 = createAgent(ref _agent1, _agent1Type, true, _agent1NetworkGeneration, _gameBoard.GetBoardState());
                        _agent2 = createAgent(ref _agent2, _agent2Type, true, _agent2NetworkGeneration, _gameBoard.GetBoardState());
                    }

                    if (_agent1PlayoutsBeforeMoveSelection > _agent1.GetCurrentPlayoutsCount())
                    {
                        _agent1.Estimate(_simulationsCountPerFrame);
                    }
                    if (_agent2PlayoutsBeforeMoveSelection > _agent2.GetCurrentPlayoutsCount())
                    {
                        _agent2.Estimate(_simulationsCountPerFrame);
                    }

                    if (_gameBoard.GetCurrentTurnColor() == Constants.CROSS_COLOR && _agent1PlayoutsBeforeMoveSelection <= _agent1.GetCurrentPlayoutsCount())
                    {
                        var move = _agent1.GetBestMove();
                        if (_gameBoard.MakeMove(move.X, move.Y))
                        {

                            _agent1.SelectNextNode(move.X, move.Y, Constants.CROSS_COLOR, false);
                            _agent2.SelectNextNode(move.X, move.Y, Constants.CROSS_COLOR, false);

                            _currentGameHistory.moves.Add(new Constants.MovePosition(move.X, move.Y));
                        }
                    } else if (_gameBoard.GetCurrentTurnColor() == Constants.ZERO_COLOR && _agent2PlayoutsBeforeMoveSelection <= _agent2.GetCurrentPlayoutsCount())
                    {
                        var move = _agent2.GetBestMove();
                        if (_gameBoard.MakeMove(move.X, move.Y))
                        {
                            _agent1.SelectNextNode(move.X, move.Y, Constants.ZERO_COLOR, false);
                            _agent2.SelectNextNode(move.X, move.Y, Constants.ZERO_COLOR, false);

                            _currentGameHistory.moves.Add(new Constants.MovePosition(move.X, move.Y));
                        }
                    }

                    if (_gameBoard.GetGameState() != Constants.GameResult.IN_PROGRESS)
                    {
                        _currentGameHistory.result = _gameBoard.GetGameState();
                        if (_gameBoard.GetGameState() != Constants.GameResult.TIE)
                        {
                            _currentGameHistory.winningColor = _gameBoard.GetCurrentTurnColor();
                        }
                        else
                        {
                            _currentGameHistory.winningColor = 0;
                        }

                        
                        _totalGames++;
                        if (_currentGameHistory.result == Constants.GameResult.WIN && _currentGameHistory.winningColor == Constants.CROSS_COLOR)
                        {
                            _agent1Won++;
                            _agent1.FillTrainingSamples(ref _trainingSamples, Constants.CROSS_COLOR);
                        }
                        if (_currentGameHistory.result == Constants.GameResult.WIN && _currentGameHistory.winningColor == Constants.ZERO_COLOR)
                        {
                            _agent2Won++;
                            _agent2.FillTrainingSamples(ref _trainingSamples, Constants.ZERO_COLOR);
                        }
                    }

                    ImGui.Text("Agent 1 wins: " + _agent1Won);
                    ImGui.Text("Agent 2 wins: " + _agent2Won);
                    ImGui.Text("Total Games: " + _totalGames);
                }

                if (_isTraining)
                {
                    ImGui.Text("Training games: " + _totalGames);
                    ImGui.Text("Agent 1 playouts: " + _agent1.GetCurrentPlayoutsCount());
                    ImGui.Text("Agent 2 playouts: " + _agent2.GetCurrentPlayoutsCount());
                }

                ImGui.SliderInt("Epoch count", ref _trainingEpochCount, 1, 100);

                if (_isTraining && ImGui.SliderInt("Agent1 playouts", ref _agent1PlayoutsBeforeMoveSelection, 100, 5000)) { }
                if (_isTraining && ImGui.SliderInt("Agent2 playouts", ref _agent2PlayoutsBeforeMoveSelection, 100, 5000)) { }

                if (_isTraining && ImGui.Button("Stop training"))
                {
                    _isTraining = false;
                }

                if (_isTraining && ImGui.Button("Learn from training"))
                {
                    _isTraining = false;
                    TrainNetworkOnSelfPlayData();

                    _agent1Won = 0;
                    _agent2Won = 0;
                    _totalGames = 0;
                }

                if (ImGui.Button("New Game"))
                {
                    _gameBoard = new GameBoard(new ArrayGameBoardState(_boardSize), Constants.CROSS_COLOR);
                    var estimator = new CNNEstimator(false);
                    estimator.LoadModel(String.Format("{0}_{1}_{2}.keras", Constants.MODEL_NAME, Constants.DEFAULT_BOARD_SIZE, _latestNetworkGeneration));
                    _estimator = estimator;

                    //_estimator = new MCTSEstimator();
                    _estimator.InitFromState(_gameBoard.GetBoardState(), Constants.CROSS_COLOR);
                    _isGameEnded = false;
                }
                ImGui.SliderInt("Board size", ref _boardSize, 3, 19);

                if (_autoEstimation && !_isGameEnded)
                {
                    _estimator.Estimate(_simulationsCountPerFrame);
                }

                if (ImGui.Button("Estimate once"))
                {
                    _estimator.Estimate(1);
                }

                if (ImGui.Button("Restart estimation"))
                {
                    _estimator.InitFromState(_gameBoard.GetBoardState(), _gameBoard.GetCurrentTurnColor());
                }

                ImGui.Checkbox("Auto estimation", ref _autoEstimation);
                ImGui.SliderInt("Playouts to move", ref _playoutsBeforeMoveSelection, 1, 1000);
                //ImGui.SliderInt("Playouts to move", ref _playoutsBeforeMoveSelection, 5000, 1000000);

                ImGui.SliderInt("Simulation per frame", ref _simulationsCountPerFrame, 1, 1024);
                if (ImGui.SliderFloat("Exploration rate", ref _explorationRate, 0.1f, 2.0f))
                {
                    _estimator.SetExplorationConst(_explorationRate);
                }

                ImGui.Checkbox("Play black", ref _playForBlack);
                ImGui.Checkbox("Play white", ref _playForWhite);

                ImGui.Checkbox("Show probabilities", ref _showAIProbabilities);
                ImGui.Checkbox("Show best move", ref _showAIBestMove);


                _estimator.DebugMenuDraw(ref _gameBoard);

                ImGui.End();
                rlImGui.End();

                if ((ImGui.GetIO().ConfigFlags & ImGuiConfigFlags.ViewportsEnable) > 0)
                {
                    ImGui.UpdatePlatformWindows();
                    ImGui.RenderPlatformWindowsDefault();
                }


                _estimator.DebugFieldDraw();

                if (_playForBlack && _gameBoard.GetCurrentTurnColor() == Constants.CROSS_COLOR)
                {
                    if (_estimator.GetCurrentPlayoutsCount() > _playoutsBeforeMoveSelection)
                    {
                        var bestMode = _estimator.GetBestMove();
                        _gameBoard.MakeMove(bestMode.X, bestMode.Y);

                        if (_gameBoard.GetGameState() != Constants.GameResult.IN_PROGRESS)
                        {
                            _isGameEnded = true;
                        }

                        if (!_estimator.SelectNextNode(bestMode.X, bestMode.Y, Constants.CROSS_COLOR))
                        {
                            _estimator.InitFromState(_gameBoard.GetBoardState(), _gameBoard.GetCurrentTurnColor());
                        }
                    }
                }

                if (_playForWhite && _gameBoard.GetCurrentTurnColor() == Constants.ZERO_COLOR)
                {
                    if (_estimator.GetCurrentPlayoutsCount() > _playoutsBeforeMoveSelection)
                    {
                        var bestMode = _estimator.GetBestMove();
                        _gameBoard.MakeMove(bestMode.X, bestMode.Y);

                        if (_gameBoard.GetGameState() != Constants.GameResult.IN_PROGRESS)
                        {
                            _isGameEnded = true;
                        }

                        if (!_estimator.SelectNextNode(bestMode.X, bestMode.Y, Constants.ZERO_COLOR))
                        {
                            _estimator.InitFromState(_gameBoard.GetBoardState(), _gameBoard.GetCurrentTurnColor());
                        }
                    }
                }

                var position = Raylib.GetMousePosition();
                if (Raylib.IsMouseButtonPressed(0) && !_isGameEnded)
                {
                    int x = ((int)position.X - offsetX) / Constants.CELL_SIZE;
                    int y = ((int)position.Y - offsetY) / Constants.CELL_SIZE;

                    if (x >= 0 && y >= 0 && x < _gameBoard.GetBoardSize() && y < _gameBoard.GetBoardSize())
                    {
                        var currentColor = _gameBoard.GetCurrentTurnColor();
                        _gameBoard.MakeMove(x, y);

                        if (_gameBoard.GetGameState() != Constants.GameResult.IN_PROGRESS)
                        {
                            _isGameEnded = true;
                        }

                        if (!_estimator.SelectNextNode(x, y, currentColor))
                        {
                            _estimator.InitFromState(_gameBoard.GetBoardState(), _gameBoard.GetCurrentTurnColor());
                        }
                    }
                }

                Raylib.EndDrawing();
            }

            rlImGui.Shutdown();     // cleans up ImGui
            Raylib.CloseWindow();
        }
    }
}
