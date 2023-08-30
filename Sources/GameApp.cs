using ImGuiNET;
using Keras;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
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
using static GomokuNN.Sources.Constants;

namespace GomokuNN.Sources
{
    class GameApp
    {
        private Config _config;
        private CNNModelsLoader _modelsLoader;

        private RenderHelper _renderHelper;

        private GameController? _currentGame = null;
        private GameSettings _newGameSettings = new GameSettings();

        private Constants.MovePosition _lastMovePosition = new Constants.MovePosition();
        //private IGameEstimator _agent1, _agent2;
        //private List<string> _gameNames = new List<string>();
        //private int _currentGameIndex = 0;

        //private GameHistory _currentGameHistory;
        //private int _currentMoveIndex = 0;
        private List<TrainingSample> _trainingSamples = new List<TrainingSample>();


        //private static int network_latest_version = 6;
        //private int _latestNetworkGeneration = network_latest_version;
        //private int _agent1NetworkGeneration = network_latest_version;
        //private int _agent2NetworkGeneration = network_latest_version;


        List<string> _loadedGames = new List<string>();

        Random _rndProvider = new Random();

        public GameApp()
        {
            SetupPyEnv();

            ZobristHash.Init(Constants.DEFAULT_BOARD_SIZE);

            _config = Config.Load("config.json");
            _modelsLoader = new CNNModelsLoader(_config);

            _currentGame = null;

            _renderHelper = new RenderHelper();
        }

        private static void SetupPyEnv()
        {
            if (OperatingSystem.IsWindows())
            {
                Runtime.PythonDLL = @"C:\Python38\python38.dll";
            }
            else
            {
                Runtime.PythonDLL = @"/usr/local/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/lib/libpython3.11.dylib";
                Environment.SetEnvironmentVariable("PYTHONNET_PYDLL", @"/usr/local/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/lib/libpython3.11.dylib");
            }
        }

        private static BaseLayer AddResidualBlock(int filtersCount, BaseLayer x)
        {
            var firstLayer = new Conv2D(filtersCount, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last", strides: (1, 1).ToTuple()).Set(x);
            firstLayer = new BatchNormalization().Set(firstLayer);
            firstLayer = new Activation("relu").Set(firstLayer);
            firstLayer = new Conv2D(filtersCount, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last", strides: (1, 1).ToTuple()).Set(firstLayer);
            firstLayer = new BatchNormalization().Set(firstLayer);
            firstLayer = new Keras.Layers.Add(new BaseLayer[] { firstLayer, x });
            firstLayer = new Activation("relu").Set(firstLayer);

            return firstLayer;
        }

        private static void SetupModel()
        {
            const int FILTERS_COUNT = 128;

            var inputLayer = new Input(shape: (4, Constants.DEFAULT_BOARD_SIZE, Constants.DEFAULT_BOARD_SIZE), name: "input_layer");

            BaseLayer netLayer = new Conv2D(FILTERS_COUNT, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last", strides: (1, 1).ToTuple()).Set(inputLayer);
            netLayer = new BatchNormalization().Set(netLayer);
            netLayer = new Activation("relu").Set(netLayer);

            var resBlock = netLayer;
            for (int i = 0; i < 1; i++)
            {
                resBlock = AddResidualBlock(FILTERS_COUNT, resBlock);
            }

            var policyOutput = new Conv2D(2, (1, 1).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last", strides: (1, 1).ToTuple()).Set(resBlock);
            policyOutput = new BatchNormalization().Set(policyOutput);
            policyOutput = new Activation("relu").Set(policyOutput);
            policyOutput = new Flatten().Set(policyOutput);
            policyOutput = new Dense(Constants.DEFAULT_BOARD_SIZE * Constants.DEFAULT_BOARD_SIZE, activation: "softmax", name: "policy_net").Set(policyOutput);

            var valueOutput = new Conv2D(2, (1, 1).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last").Set(resBlock);
            valueOutput = new BatchNormalization().Set(valueOutput);
            valueOutput = new Activation("relu").Set(valueOutput);
            valueOutput = new Flatten().Set(valueOutput);
            valueOutput = new Dense(256, activation: "relu").Set(valueOutput);
            valueOutput = new Dense(1, activation: "tanh", name: "value_net").Set(valueOutput);

            var network = new Keras.Models.Model(new BaseLayer[] { inputLayer }, new BaseLayer[] { policyOutput, valueOutput }, name: "gomoku_net");
            //var network = Keras.Models.Model.LoadModel(String.Format("{0}_{1}_{2}.keras", Constants.MODEL_NAME, Constants.DEFAULT_BOARD_SIZE, network_latest_version));

            network.Compile(optimizer: new Keras.Optimizers.Adam(learning_rate: 0.001f),
                loss: new Dictionary<string, string> { { "policy_net", "categorical_crossentropy" }, { "value_net", "mean_squared_error" } },
                metrics: new Dictionary<string, string> { { "policy_net", "categorical_accuracy" }, { "value_net", "accuracy" } },
                weighted_metrics: new Dictionary<string, string> { { "policy_net", "categorical_accuracy" }, { "value_net", "accuracy" } },
                loss_weights: new Dictionary<string, float> { { "policy_net", 1.0f }, { "value_net", 1.0f } });
            //network.Compile(optimizer: new Keras.Optimizers.Adam(learning_rate: 0.01f), loss: new string[] { "categorical_crossentropy", "mean_squared_error" }, metrics: new string[] { "categorical_accuracy", "accuracy" });

            network.Summary();

            network.Save(String.Format("{0}_{1}_{2}.keras", Constants.MODEL_NAME, Constants.DEFAULT_BOARD_SIZE, 0));
            network.SaveOnnx(String.Format("{0}_{1}_{2}.keras.onnx", Constants.MODEL_NAME, Constants.DEFAULT_BOARD_SIZE, 0));
        }

        private static void SetModelLearningRate(int generation, int resultGeneration, float rate)
        {
            var network = Keras.Models.Model.LoadModel(String.Format("{0}_{1}_{2}.keras", Constants.MODEL_NAME, Constants.DEFAULT_BOARD_SIZE, generation));

            network.Compile(optimizer: new Keras.Optimizers.Adam(learning_rate: rate),
                loss: new Dictionary<string, string> { { "policy_net", "categorical_crossentropy" }, { "value_net", "mean_squared_error" } },
                metrics: new Dictionary<string, string> { { "policy_net", "categorical_accuracy" }, { "value_net", "accuracy" } },
                weighted_metrics: new Dictionary<string, string> { { "policy_net", "categorical_accuracy" }, { "value_net", "accuracy" } },
                loss_weights: new Dictionary<string, float> { { "policy_net", 1.0f }, { "value_net", 1.0f } });

            network.Summary();

            network.Save(String.Format("{0}_{1}_{2}.keras", Constants.MODEL_NAME, Constants.DEFAULT_BOARD_SIZE, resultGeneration));
            network.SaveOnnx(String.Format("{0}_{1}_{2}.keras.onnx", Constants.MODEL_NAME, Constants.DEFAULT_BOARD_SIZE, resultGeneration));
        }

        public void Run()
        {
            Raylib.InitWindow(1280, 768, "TEST");

            rlImGui.Setup(true);
            ImGui.GetIO().ConfigFlags |= ImGuiConfigFlags.DockingEnable;
            ImGui.GetIO().ConfigFlags |= ImGuiConfigFlags.ViewportsEnable;

            _renderHelper.LoadTextures();

            //TensorflowModelHelper.BuildModels();

            while (!Raylib.WindowShouldClose())
            {
                Raylib.BeginDrawing();
                Raylib.ClearBackground(Color.RAYWHITE);

                for (int y = 0; y < Constants.DEFAULT_BOARD_SIZE; y++)
                {
                    for (int x = 0; x < Constants.DEFAULT_BOARD_SIZE; x++)
                    {
                        int cellState = _currentGame != null && _currentGame.isGameStarted ? _currentGame.gameBoard.GetCellState(x, y) : 0;

                        Color color = _lastMovePosition.X == x && _lastMovePosition.Y == y ? Color.LIGHTGRAY : Color.WHITE;
                        _renderHelper.DrawBoardCell(cellState, x, y, color);

                        _currentGame?.firstAgent?.OnDrawCell(x, y, ref _renderHelper);
                        _currentGame?.secondAgent?.OnDrawCell(x, y, ref _renderHelper);
                    }
                }

                rlImGui.Begin();
                
                bool isOpen = true;
                ImGui.Begin("Game options", ref isOpen, ImGuiWindowFlags.AlwaysAutoResize);

                //if (ImGui.Button("Open games"))
                //{
                //    var currentDirectory = Directory.GetCurrentDirectory();
                //    var games = Directory.GetFiles(currentDirectory + "\\Resources\\DataBase\\4");
                //    _gameNames.Clear();
                //    foreach (var gameName in games)
                //    {
                //        _gameNames.Add(gameName);
                //    }

                //    _currentGameIndex = 0;

                //    _currentGameHistory = GameHistory.CreateFromPSQFile(_gameNames[_currentGameIndex]);
                //    _currentMoveIndex = 0;
                //    _gameBoard = new GameBoard(new ArrayGameBoardState(_boardSize), _currentGameHistory.startingColor);
                //    foreach (var move in _currentGameHistory.moves)
                //    {
                //        if (!_gameBoard.MakeMove(move.X, move.Y))
                //        {
                //            Console.WriteLine("Invalid move while opening game: " + _gameNames[_currentGameIndex]);
                //            break;
                //        }

                //        _currentMoveIndex++;
                //    }
                //}

                //if (_gameNames.Count > 0)
                //{
                //    if (ImGui.Button("<<"))
                //    {
                //        _currentGameIndex--;
                //        if (_currentGameIndex < 0) { _currentGameIndex = 0; }

                //        _currentGameHistory = GameHistory.CreateFromPSQFile(_gameNames[_currentGameIndex]);
                //        _currentMoveIndex = 0;
                //        _gameBoard = new GameBoard(new ArrayGameBoardState(_boardSize), _currentGameHistory.startingColor);
                //        foreach (var move in _currentGameHistory.moves)
                //        {
                //            if (!_gameBoard.MakeMove(move.X, move.Y))
                //            {
                //                Console.WriteLine("Invalid move while opening game: " + _gameNames[_currentGameIndex]);
                //                break;
                //            }

                //            _currentMoveIndex++;
                //        }
                //    }
                //    ImGui.SameLine();
                //    if (ImGui.Button(">>"))
                //    {
                //        _currentGameIndex++;
                //        if (_currentGameIndex >= _gameNames.Count) { _currentGameIndex = _gameNames.Count - 1; }

                //        _currentGameHistory = GameHistory.CreateFromPSQFile(_gameNames[_currentGameIndex]);
                //        _currentMoveIndex = 0;
                //        _gameBoard = new GameBoard(new ArrayGameBoardState(_boardSize), _currentGameHistory.startingColor);
                //        foreach (var move in _currentGameHistory.moves)
                //        {
                //            if (!_gameBoard.MakeMove(move.X, move.Y))
                //            {
                //                Console.WriteLine("Invalid move while opening game: " + _gameNames[_currentGameIndex]);
                //                break;
                //            }

                //            _currentMoveIndex++;
                //        }
                //    }

                //    ImGui.Text("Current Game: " + _gameNames[_currentGameIndex]);
                //}

                //if (ImGui.Button("Open game"))
                //{
                //    var currentDirectory = Directory.GetCurrentDirectory();
                //    var gamePath = currentDirectory + "\\Resources\\DataBase\\4\\0_8_9_1.psq";
                //    _currentGameHistory = GameHistory.CreateFromPSQFile(currentDirectory + "\\Resources\\DataBase\\4\\0_8_9_1.psq");

                //    _currentMoveIndex = 0;
                //    _gameBoard = new GameBoard(new ArrayGameBoardState(_boardSize), _currentGameHistory.startingColor);
                //    foreach (var move in _currentGameHistory.moves)
                //    {
                //        if (!_gameBoard.MakeMove(move.X, move.Y))
                //        {
                //            Console.WriteLine("Invalid move while opening game: " + gamePath);
                //            break;
                //        }

                //        _currentMoveIndex++;
                //    }
                //}

                //if (_currentGameHistory != null)
                //{
                //    if (ImGui.Button("<"))
                //    {
                //        _currentMoveIndex--;
                //        if (_currentMoveIndex < 0) {  _currentMoveIndex = 0; }

                //        _gameBoard = new GameBoard(new ArrayGameBoardState(_boardSize), _currentGameHistory.startingColor);
                //        for (int i = 0; i <= _currentMoveIndex; i++)
                //        {
                //            var move = _currentGameHistory.moves[i];
                //            if (!_gameBoard.MakeMove(move.X, move.Y))
                //            {
                //                Console.WriteLine("Invalid move while opening game: ");
                //                break;
                //            }
                //        }
                //    }
                //    ImGui.SameLine();
                //    if (ImGui.Button(">"))
                //    {
                //        _currentMoveIndex++;
                //        if (_currentMoveIndex >= _currentGameHistory.moves.Count) { _currentMoveIndex = _currentGameHistory.moves.Count - 1; }

                //        _gameBoard = new GameBoard(new ArrayGameBoardState(_boardSize), _currentGameHistory.startingColor);
                //        for (int i = 0; i <= _currentMoveIndex; i++)
                //        {
                //            var move = _currentGameHistory.moves[i];
                //            if (!_gameBoard.MakeMove(move.X, move.Y))
                //            {
                //                Console.WriteLine("Invalid move while opening game: ");
                //                break;
                //            }
                //        }
                //    }

                //    ImGui.Text("Winner: " + _currentGameHistory.winningColor + " ; " + _gameBoard.GetCurrentTurnColor());
                //}

                //if (ImGui.Button("Train on games DB"))
                //{
                //    int networkGeneration = 0;

                //    HashSet<long> knownPositions = new HashSet<long>();
                //    _trainingSamples.Clear();

                //    var currentDirectory = Directory.GetCurrentDirectory();

                //    XmlDocument gamesLib = new XmlDocument();
                //    gamesLib.Load(currentDirectory + "\\Resources\\DataBase\\renjunet_v10_20230809.rif");

                //    char[] MOVE_COORD = new char[] { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't' };

                //    var gamesData = gamesLib.DocumentElement.SelectSingleNode("games");
                //    var games = gamesData.SelectNodes("game");
                //    for (int i = 0; i < games.Count; i++)
                //    {
                //        var game = games[i];
                //        float result = float.Parse(game.Attributes["bresult"].Value);

                //        var movesNode = game.SelectSingleNode("move");
                //        var movesStr = movesNode.InnerText;
                //        string[] moves = movesStr.Split(' ');

                //        if (moves.Length < 5)
                //        {
                //            continue;
                //        }

                //        var gameHistory = new GameHistory();
                //        gameHistory.startingColor = Constants.CROSS_COLOR;
                //        gameHistory.winningColor = result > 0.6f ? Constants.CROSS_COLOR : result < 0.4f ? Constants.ZERO_COLOR : Constants.EMPTY_COLOR;
                //        gameHistory.result = gameHistory.winningColor == Constants.EMPTY_COLOR ? Constants.GameResult.TIE : gameHistory.startingColor == gameHistory.winningColor ? Constants.GameResult.WIN : Constants.GameResult.LOSE;
                //        gameHistory.moves = new List<Constants.MovePosition>();

                //        foreach (var move in moves)
                //        {
                //            int x = Array.IndexOf(MOVE_COORD, move[0]);
                //            int y = int.Parse(move.Substring(1)) - 1;

                //            gameHistory.moves.Add(new Constants.MovePosition(x, y));
                //        }

                //        TrainingSample.FillFromGameHistory(ref _trainingSamples, gameHistory, ref knownPositions);

                //        if (_trainingSamples.Count > 2000000)
                //        {
                //            Console.WriteLine("Training with " + i + " lib game...");
                //            Console.WriteLine("Board positions count: " + _trainingSamples.Count);

                //            TensorflowModelHelper.Train(networkGeneration, networkGeneration + 1, ref _trainingSamples, 0.2f, 32, 4);
                //            networkGeneration++;

                //            _trainingSamples.Clear();
                //        }
                //    }

                //    if (_trainingSamples.Count > 0)
                //    {
                //        Console.WriteLine("Board positions count: " + _trainingSamples.Count);

                //        TensorflowModelHelper.Train(networkGeneration, networkGeneration + 1, ref _trainingSamples, 0.2f, 32, 4);
                //        networkGeneration++;

                //        _trainingSamples.Clear();
                //    }
                //}

                if (ImGui.Button("Recreate Model"))
                {
                    SetupModel();
                }

                if (ImGui.Button("Train on game files"))
                {
                    var _latestNetworkGeneration = 0;
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

                            if (_trainingSamples.Count > 1000000)
                            {
                                Console.WriteLine("Training with " + sample + " sample...");
                                Console.WriteLine("Board positions count: " + _trainingSamples.Count);

                                NetworkTrainer.Train(_latestNetworkGeneration, _latestNetworkGeneration + 1, ref _trainingSamples, 0.2f, 1024, 6);

                                _trainingSamples.Clear();

                                _latestNetworkGeneration = _latestNetworkGeneration + 1;
                            }
                        }

                        if (_trainingSamples.Count > 1000000)
                        {
                            Console.WriteLine("Training with " + sample + " sample...");
                            Console.WriteLine("Board positions count: " + _trainingSamples.Count);

                            NetworkTrainer.Train(_latestNetworkGeneration, _latestNetworkGeneration + 1, ref _trainingSamples, 0.2f, 1024, 6);

                            _trainingSamples.Clear();

                            _latestNetworkGeneration = _latestNetworkGeneration + 1;
                        }
                    }

                    if (_trainingSamples.Count > 0)
                    {
                        Console.WriteLine("Board positions count: " + _trainingSamples.Count);

                        NetworkTrainer.Train(_latestNetworkGeneration, _latestNetworkGeneration + 1, ref _trainingSamples, 0.2f, 1024, 6);

                        _trainingSamples.Clear();
                    }
                }

                if (ImGui.Button("Train CNN"))
                {
                    var gym = new AIGym();

                    //gym.AddBaseLineParticipant(new GymParticipant()
                    //{
                    //    id = 1,
                    //    agent = new GameAgentSettings(EstimatorType.CNN, CNNHelper.GetCNNPathByGeneration(10), 2500, 2.0f)
                    //});

                    //int trainingAgent = 97;
                    //SetModelLearningRate(6, 7, 0.0001f);

                    int trainingAgent = 99;
                    for (int i = 0; i < 100; i++)
                    {
                        trainingAgent = gym.TrainAgent(trainingAgent);
                    }
                }

                if (ImGui.Button("Test"))
                {
                    var pos = new HashSet<long>();

                    var samples1 = _currentGame.firstAgent?.GetTrainingSamples(1, ref pos);
                    var samples2 = _currentGame.secondAgent?.GetTrainingSamples(2, ref pos);

                    Console.WriteLine("Test!");

                    //inferenceSession.
                    //policyOutput.F

                    //float bestValue = -1000;
                    //var samples = new List<TrainingSample>();
                    //var nextMove = new MovePosition();
                    //var stateSample = new TrainingSample(_currentGame.gameBoard.GetBoardState(), new MovePosition(), _lastMovePosition, _currentGame.gameBoard.GetCurrentTurnColor(), valueOutput[0]);
                    //for (int y = 0; y < _currentGame.gameBoard.GetBoardSize(); y++)
                    //{
                    //    for (int x = 0; x < _currentGame.gameBoard.GetBoardSize(); x++)
                    //    {
                    //        stateSample.SetPolicyOutputForMove(_currentGame.gameBoard.GetBoardSize(), x, y, policyOutput.ElementAt(y * _currentGame.gameBoard.GetBoardSize() + x));
                    //        if (policyOutput.ElementAt(y * _currentGame.gameBoard.GetBoardSize() + x) > bestValue)
                    //        {
                    //            bestValue = policyOutput.ElementAt(y * _currentGame.gameBoard.GetBoardSize() + x);
                    //            Console.WriteLine("Best value: " + x + " " + y + " " + bestValue);
                    //        }
                    //    }
                    //}
                    //samples.Add(stateSample);

                    //NetworkTrainer.Train(6, 99, ref samples, 0, 1, 1);
                }

                if (ImGui.Button("New Game"))
                {
                    _currentGame = new GameController(_newGameSettings);
                    _currentGame.CreateEstimators();
                    _currentGame.StartGame();
                }

                string[] agentTypes = new string[] { EstimatorType.NONE.ToString(), EstimatorType.CNN.ToString(), EstimatorType.MCTS.ToString(), EstimatorType.IDDFS.ToString() };

                int firstAgentIndex = 0;
                int secondAgentIndex = 0;
                for (int agentIndex = 0; agentIndex < agentTypes.Length; agentIndex++)
                {
                    if (agentTypes[agentIndex] == _newGameSettings.firstAgent.type.ToString())
                    {
                        firstAgentIndex = agentIndex;
                    }

                    if (agentTypes[agentIndex] == _newGameSettings.secondAgent.type.ToString())
                    {
                        secondAgentIndex = agentIndex;
                    }
                }

                if (ImGui.BeginCombo("First agent type", agentTypes[firstAgentIndex])) // The second parameter is the label previewed before opening the combo.
                {
                    for (int n = 0; n < agentTypes.Length; n++)
                    {
                        bool is_selected = (firstAgentIndex == n); // You can store your selection however you want, outside or inside your objects
                        if (ImGui.Selectable(agentTypes[n], is_selected))
                        {
                            firstAgentIndex = n;
                             _newGameSettings.firstAgent.type = EstimatorTypeHelper.ParseEstimatorType(agentTypes[n]);

                            if (is_selected)
                            {
                                ImGui.SetItemDefaultFocus();
                            }
                        }
                    }

                    ImGui.EndCombo();
                }

                int firstAgentModel = _modelsLoader.availableModels.IndexOf(_newGameSettings.firstAgent.modelPath);
                if (firstAgentModel == -1)
                {
                    firstAgentModel = 0;
                    _newGameSettings.firstAgent.modelPath = _modelsLoader.availableModels[0];
                }
                if (ImGui.BeginCombo("First agent model", Path.GetFileName(_modelsLoader.availableModels[firstAgentModel])))
                {
                    for (int n = 0; n < _modelsLoader.availableModels.Count; n++)
                    {
                        bool is_selected = (_newGameSettings.firstAgent.modelPath == _modelsLoader.availableModels[n]); // You can store your selection however you want, outside or inside your objects
                        var modelName = Path.GetFileName(_modelsLoader.availableModels[n]);
                        if (ImGui.Selectable(modelName, is_selected))
                        {
                            firstAgentModel = n;
                            _newGameSettings.firstAgent.modelPath = _modelsLoader.availableModels[n];

                            if (is_selected)
                            {
                                ImGui.SetItemDefaultFocus();
                            }
                        }
                    }

                    ImGui.EndCombo();
                }
                var firstAgentPlayouts = _newGameSettings.firstAgent.playoutsCount;
                if (ImGui.SliderInt("First agent playouts", ref firstAgentPlayouts, 100, 3000))
                {
                    _newGameSettings.firstAgent.playoutsCount = firstAgentPlayouts;
                }

                if (ImGui.BeginCombo("Second agent type", agentTypes[secondAgentIndex])) // The second parameter is the label previewed before opening the combo.
                {
                    for (int n = 0; n < agentTypes.Length; n++)
                    {
                        bool is_selected = (secondAgentIndex == n); // You can store your selection however you want, outside or inside your objects
                        if (ImGui.Selectable(agentTypes[n], is_selected))
                        {
                            secondAgentIndex = n;
                            _newGameSettings.secondAgent.type = EstimatorTypeHelper.ParseEstimatorType(agentTypes[n]);

                            if (is_selected)
                            {
                                ImGui.SetItemDefaultFocus();
                            }
                        }
                    }

                    ImGui.EndCombo();
                }

                int secondAgentModel = _modelsLoader.availableModels.IndexOf(_newGameSettings.secondAgent.modelPath);
                if (secondAgentModel == -1)
                {
                    secondAgentModel = 0;
                    _newGameSettings.secondAgent.modelPath = _modelsLoader.availableModels[0];
                }
                if (ImGui.BeginCombo("Second agent model", Path.GetFileName(_modelsLoader.availableModels[secondAgentModel])))
                {
                    for (int n = 0; n < _modelsLoader.availableModels.Count; n++)
                    {
                        bool is_selected = (_newGameSettings.secondAgent.modelPath == _modelsLoader.availableModels[n]); // You can store your selection however you want, outside or inside your objects
                        var modelName = Path.GetFileName(_modelsLoader.availableModels[n]);
                        if (ImGui.Selectable(modelName, is_selected))
                        {
                            secondAgentModel = n;
                            _newGameSettings.secondAgent.modelPath = _modelsLoader.availableModels[n];

                            if (is_selected)
                            {
                                ImGui.SetItemDefaultFocus();
                            }
                        }
                    }

                    ImGui.EndCombo();
                }
                var secondAgentPlayouts = _newGameSettings.secondAgent.playoutsCount;
                if (ImGui.SliderInt("Second agent playouts", ref secondAgentPlayouts, 100, 3000))
                {
                    _newGameSettings.secondAgent.playoutsCount = secondAgentPlayouts;
                }

                if (ImGui.TreeNode("First agent debug data"))
                {
                    _currentGame?.firstAgent?.DebugMenuDraw(ref _currentGame.gameBoard);
                    ImGui.TreePop();
                }
                if (ImGui.TreeNode("Second agent debug data"))
                {
                    _currentGame?.secondAgent?.DebugMenuDraw(ref _currentGame.gameBoard);
                    ImGui.TreePop();
                }

                ImGui.End();
                rlImGui.End();

                if ((ImGui.GetIO().ConfigFlags & ImGuiConfigFlags.ViewportsEnable) > 0)
                {
                    ImGui.UpdatePlatformWindows();
                    ImGui.RenderPlatformWindowsDefault();
                }

                if (_currentGame != null)
                {
                    if (_currentGame.isGameInProgress)
                    {
                        if (_currentGame.firstAgent != null && !_currentGame.firstAgent.HasContiniousEstimationSupport())
                        {
                            _currentGame.firstAgent.EstimateOnce();
                        }
                        if (_currentGame.secondAgent != null && !_currentGame.secondAgent.HasContiniousEstimationSupport())
                        {
                            _currentGame.secondAgent.EstimateOnce();
                        }

                        var position = Raylib.GetMousePosition();
                        if (Raylib.IsMouseButtonPressed(0))
                        {
                            int x = ((int)position.X - Constants.BOARD_DRAW_OFFSET) / Constants.CELL_SIZE;
                            int y = ((int)position.Y - Constants.BOARD_DRAW_OFFSET) / Constants.CELL_SIZE;

                            if (x >= 0 && y >= 0 && x < Constants.DEFAULT_BOARD_SIZE && y < Constants.DEFAULT_BOARD_SIZE)
                            {
                                if (_currentGame.MakeMove(x, y))
                                {
                                    _lastMovePosition.X = x;
                                    _lastMovePosition.Y = y;
                                }
                            }
                        }

                        if (_currentGame.firstAgent?.GetCurrentPlayoutsCount() > _currentGame.gameSettings.firstAgent.playoutsCount && _currentGame.gameBoard.GetCurrentTurnColor() == Constants.CROSS_COLOR)
                        {
                            var bestMove = _currentGame.firstAgent.GetBestMove();
                            if (_currentGame.MakeMove(bestMove.X, bestMove.Y))
                            {
                                _lastMovePosition = bestMove;
                            }
                        }

                        if (_currentGame.secondAgent?.GetCurrentPlayoutsCount() > _currentGame.gameSettings.secondAgent.playoutsCount && _currentGame.gameBoard.GetCurrentTurnColor() == Constants.ZERO_COLOR)
                        {
                            var bestMove = _currentGame.secondAgent.GetBestMove();
                            if (_currentGame.MakeMove(bestMove.X, bestMove.Y))
                            {
                                _lastMovePosition = bestMove;
                            }
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
