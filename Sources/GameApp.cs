using ImGuiNET;
using Keras;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Numpy;
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
using System.Text;
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

        private int _trainingEpochCount = 50;

        private bool _isGameEnded = false;

        private bool _isTraining = false;

        private IGameEstimator _agent1, _agent2;
        private GameHistory currentGameHistory;
        private List<GameHistory> _gameHistories = new List<GameHistory>();

        private int _totalGames = 0;
        private int _agent1Won = 0;
        private int _agent2Won = 0;

        private static int network_latest_version = 38;
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

            // Runtime.PythonDLL = @"C:\Python38\python38.dll";
            //PythonEngine.PythonHome = envPythonHome;
            //PythonEngine.PythonPath = Environment.GetEnvironmentVariable("PYTHONPATH");

            string pythonPath1 = @"C:\Python38\";
            string pythonPath2 = @"C:\Python38\site-packages\";

            Runtime.PythonDLL = @"/usr/local/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/lib/libpython3.11.dylib";

            //Environment.SetEnvironmentVariable("PATH", pythonPath1, EnvironmentVariableTarget.Process);
            //Environment.SetEnvironmentVariable("PYTHONHOME", pythonPath1, EnvironmentVariableTarget.Process);
            //Environment.SetEnvironmentVariable("PYTHONPATH", pythonPath2, EnvironmentVariableTarget.Process);
            Environment.SetEnvironmentVariable("PYTHONNET_PYDLL", @"/usr/local/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/lib/libpython3.11.dylib");
        }

        private static void SetupModel()
        {
            //var model = new Sequential();
            //model.Add(new Conv2D(64, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last"));
            //model.Add(new BatchNormalization(axis: -1));
            //model.Add(new Conv2D(64, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last"));
            //model.Add(new BatchNormalization(axis: -1));
            //model.Add(new Conv2D(64, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last"));
            //model.Add(new BatchNormalization(axis: -1));
            //model.Add(new Conv2D(64, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last"));
            //model.Add(new BatchNormalization(axis: -1));
            //model.Add(new Conv2D(64, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last"));
            //model.Add(new BatchNormalization(axis: -1));
            //model.Add(new Conv2D(64, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last"));
            //model.Add(new BatchNormalization(axis: -1));
            //model.Add(new Conv2D(64, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last"));
            //model.Add(new BatchNormalization(axis: -1));
            //model.Add(new Conv2D(64, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last"));
            //model.Add(new BatchNormalization(axis: -1));
            //model.Add(new Conv2D(64, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last"));
            //model.Add(new BatchNormalization(axis: -1));
            //model.Add(new Flatten());
            //model.Add(new Dense(512, activation: "softmax"));
            //model.Add(new Dense(10, activation: "tanh"));

            //model.Summary();

            //model.Compile(optimizer: "Adam", loss: "mean_squared_error", metrics: new string[] { "accuracy" });

            //var policyModel = new Sequential();
            //policyModel.Add(new Input((3, 3, 3)));
            //policyModel.Add(new Conv2D(32, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last"));
            ////policyModel.Add(new BatchNormalization(axis: -1));
            //policyModel.Add(new Conv2D(64, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last"));
            ////policyModel.Add(new BatchNormalization(axis: -1));
            //policyModel.Add(new Conv2D(128, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last"));
            ////policyModel.Add(new BatchNormalization(axis: -1));

            //policyModel.Add(new Conv2D(32, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last"));
            //policyModel.Add(new Flatten());
            //policyModel.Add(new Dense(9, activation: "softmax"));

            ////policyModel.Compile(optimizer: "Adam", loss: "categorical_crossentropy", metrics: new string[] { "accuracy" });

            //var valueModel = new Sequential();
            //valueModel.Add(new Input((3, 3, 3)));
            //valueModel.Add(new Conv2D(32, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last"));
            ////policyModel.Add(new BatchNormalization(axis: -1));
            //valueModel.Add(new Conv2D(64, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last"));
            ////policyModel.Add(new BatchNormalization(axis: -1));
            //valueModel.Add(new Conv2D(128, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last"));
            //valueModel.Add(new Conv2D(2, (1, 1).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last"));
            //valueModel.Add(new Flatten());
            //valueModel.Add(new Dense(64, activation: "relu"));
            //valueModel.Add(new Dense(1, activation: "tanh"));

            //valueModel.Compile(optimizer: "Adam", loss: "mean_squared_error", metrics: new string[] { "accuracy" });

            //var inputLayer = new Input(shape: (3, Constants.DEFAULT_BOARD_SIZE, Constants.DEFAULT_BOARD_SIZE), name: "input_layer");
            ////var inputLayer = new Numpy.Models.Shape(3, 3, 3);

            //BaseLayer netLayer = new Conv2D(32, (5, 5).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last").Set(inputLayer);
            //netLayer = new BatchNormalization(axis: -1).Set(netLayer);
            //netLayer = new Conv2D(32, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last").Set(netLayer);
            //netLayer = new BatchNormalization(axis: -1).Set(netLayer);
            //netLayer = new Conv2D(32, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last").Set(netLayer);
            //netLayer = new BatchNormalization(axis: -1).Set(netLayer);
            //netLayer = new Conv2D(32, (3, 3).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last").Set(netLayer);
            //netLayer = new BatchNormalization(axis: -1).Set(netLayer);

            //var policyOutput = new Conv2D(4, (1, 1).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last").Set(netLayer);
            //policyOutput = new Flatten().Set(policyOutput);
            //policyOutput = new Dense(Constants.DEFAULT_BOARD_SIZE * Constants.DEFAULT_BOARD_SIZE, activation: "softmax", name: "policy_net").Set(policyOutput);

            ////policyModel.Compile(optimizer: "Adam", loss: "categorical_crossentropy", metrics: new string[] { "accuracy" });

            //var valueOutput = new Conv2D(2, (1, 1).ToTuple(), activation: "relu", padding: "same", data_format: "channels_last").Set(netLayer);
            //valueOutput = new Flatten().Set(valueOutput);
            //valueOutput = new Dense(64, activation: "relu").Set(valueOutput);
            //valueOutput = new Dense(1, activation: "tanh", name: "value_net").Set(valueOutput);

            //var network = new Keras.Models.Model(new BaseLayer[] { inputLayer }, new BaseLayer[] { policyOutput, valueOutput }, name: "gomoku_net");
            ////var network = new Keras.Models.Model(new BaseLayer[] { inputLayer }, new BaseLayer[] { policyOutput }, name: "gomoku_net");

            ////network.Compile(optimizer: new Keras.Optimizers.Adam(learning_rate: 0.8f), loss: new Dictionary<string, string> { { "policy_net", "categorical_crossentropy" }, { "value_net", "mean_squared_error" } }, metrics: new Dictionary<string, string> { { "policy_net", "categorical_accuracy" }, { "value_net", "accuracy" } });
            //network.Compile(optimizer: new Keras.Optimizers.Adam(learning_rate: 0.01f), loss: new string[] { "categorical_crossentropy", "mean_squared_error" }, metrics: new string[] { "categorical_accuracy", "accuracy" });

            //network.Summary();

            //network.Save(String.Format("gomoku_zero_{0}_{1}.keras", Constants.DEFAULT_BOARD_SIZE, network_latest_version));

            var input = np.array(new int[] {
                1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}).reshape((1, 3, 15, 15));

            //var policyOutputData = np.array(new int[] {
            //    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            //    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            //    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            //    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            //    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            //    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            //    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            //    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            //    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            //    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            //    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            //    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            //    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            //    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            //    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}).reshape(1, 225);

            //var valueOutputData = np.array(new float[] {
            //    0});

            //network.Fit(input, policyOutputData, batch_size: 1, epochs: 10);
            //network.Fit(input, new NDarray[] { policyOutputData, valueOutputData }, batch_size: 1, epochs: 1);

            var testModel = Keras.Models.Model.LoadModel(String.Format("gomoku_zero_{0}_{1}.keras", Constants.DEFAULT_BOARD_SIZE, network_latest_version));

            //network.Fit(input, new NDarray[] { policyOutputData, valueOutputData }, batch_size: 1, epochs: 1);

            //var input = np.array(new int[] {
            //    0, 0, 0,
            //    0, 1, 0,
            //    0, 0, 0,
            //    0, 0, 0, // second
            //    0, 0, 0,
            //    0, 0, 0,
            //    0, 0, 0, // third
            //    0, 1, 0,
            //    0, 0, 0,
            //}).reshape((1, 3, 3, 3));

            //var prediction = testModel.PredictMultipleOutputs(input);
            //Console.WriteLine(prediction[0]);
            //Console.WriteLine(prediction[1]);

            //testModel.Compile(optimizer: new Keras.Optimizers.Adam(learning_rate: 1.0f), loss: new Dictionary<string, string> { { "policy_net", "categorical_crossentropy" }, { "value_net", "mean_squared_error" } }, metrics: new Dictionary<string, string> { { "policy_net", "categorical_accuracy" }, { "value_net", "accuracy" } });
            testModel.Summary();

            //testModel.Save(String.Format("gomoku_zero_{0}_{1}.keras", Constants.DEFAULT_BOARD_SIZE, network_latest_version));

            //var output = np.array(new float[]
            //{
            //    0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5f
            //}).reshape(1, 10);

            //Console.WriteLine(input);

            //network.Save("gomoku_net_generation_0.keras");

            //policyModel.Save("gomoku_policy_net_generation_0.keras");
            //valueModel.Save("gomoku_value_net_generation_0.keras");

            //var policyResult = policyModel.Predict(input);
            //Console.WriteLine(policyResult);

            //var valueModelResult = valueModel.Predict(input);
            //Console.WriteLine(valueModelResult);
        }

        private IGameEstimator createAgent(ref IGameEstimator previousAgent, AgentType agentType, bool isTraining, int networkGeneration, IGameBoardState state)
        {
            if (agentType == AgentType.CNN)
            {
                var agent = new CNNEstimator(false);
                agent.LoadModel(String.Format("gomoku_zero_{0}_{1}.keras", Constants.DEFAULT_BOARD_SIZE, networkGeneration));
                //_agent1 = new RandomEstimator();
                //_agent1 = new MCTSEstimator();
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

        public void TrainNetwork(int networkGeneration)
        {
            int trainingDataCount = 1;
            for (int i = 0; i < _gameHistories.Count; i++)
            {
                trainingDataCount += _gameHistories[i].moves.Count;
            }

            int inputIndex = 0;
            int inputDataOffset = _gameBoard.GetBoardSize() * _gameBoard.GetBoardSize() * 3;
            int secondPlayerOffset = _gameBoard.GetBoardSize() * _gameBoard.GetBoardSize();
            int[] inputData = new int[_gameBoard.GetBoardSize() * _gameBoard.GetBoardSize() * 3 * trainingDataCount];
            float[] outputData1 = new float[_gameBoard.GetBoardSize() * _gameBoard.GetBoardSize() * trainingDataCount];
            float[] outputData2 = new float[trainingDataCount];
            for (int i = 0; i < _gameHistories.Count; i++, inputIndex++)
            {
                var boardState = new ArrayGameBoardState(_gameBoard.GetBoardSize());
                var color = _gameHistories[i].startingColor;
                var winningColor = _gameHistories[i].winningColor;
                var result = _gameHistories[i].result;
                for (int moveIndex = 0; moveIndex < _gameHistories[i].moves.Count; moveIndex++)
                {
                    var move = _gameHistories[i].moves[moveIndex];
                    boardState.SetCellState(move.X, move.Y, color);
                    color = Constants.RotateColor(color);

                    for (int index = 0; index < boardState.GetBoardSize() * boardState.GetBoardSize(); index++)
                    {
                        var value = boardState.GetRawCellState(index);
                        inputData[inputIndex * inputDataOffset + index] = value == Constants.CROSS_COLOR ? 1 : 0;
                        inputData[inputIndex * inputDataOffset + index + secondPlayerOffset] = value == Constants.ZERO_COLOR ? 1 : 0;
                    }

                    var lastMoveHash = boardState.GetPositionHash(move.X, move.Y);
                    inputData[inputIndex * inputDataOffset + secondPlayerOffset * 2 + lastMoveHash] = 1;

                    //string outputData1String = "[";
                    var nextMoveHash = _gameHistories[i].moves.Count <= (moveIndex + 1) ? -1 : boardState.GetPositionHash(_gameHistories[i].moves[moveIndex + 1].X, _gameHistories[i].moves[moveIndex + 1].Y);
                    for (int index = 0; index < boardState.GetBoardSize() * boardState.GetBoardSize(); index++)
                    {
                        if (index == nextMoveHash)
                        {
                            outputData1[inputIndex * inputDataOffset + index] = result != Constants.GameResult.TIE ? (color == winningColor ? 1 : -1) : 0;
                            //outputData1String += outputData1[inputIndex * inputDataOffset + index].ToString() + " ";
                        }
                        else
                        {
                            outputData1[inputIndex * inputDataOffset + index] = 0;
                            //outputData1String += outputData1[inputIndex * inputDataOffset + index].ToString() + " ";
                        }
                    }
                    //outputData1String += "]";

                    outputData2[inputIndex] = result != Constants.GameResult.TIE ? (color == winningColor ? 1 : -1) : 0;

                    //Console.WriteLine("Current board state:");
                    //for (int ty = 0; ty < boardState.GetBoardSize(); ty++)
                    //{
                    //    string stateString = "";
                    //    for (int tx = 0; tx < boardState.GetBoardSize(); tx++)
                    //    {
                    //        stateString += boardState.GetCellState(tx, ty).ToString() + " ";
                    //    }

                    //    Console.WriteLine(stateString);
                    //}

                    //Console.WriteLine("Choice: " + move.X + " " + move.Y + " " + Constants.RotateColor(color) + " " + winningColor);
                    //Console.WriteLine(outputData1String);
                    //Console.WriteLine(outputData2[inputIndex]);
                }
            }

            var input = np.array(inputData).reshape((trainingDataCount, 3, _gameBoard.GetBoardSize(), _gameBoard.GetBoardSize()));
            var output1 = np.array(outputData1).reshape((trainingDataCount, _gameBoard.GetBoardSize() * _gameBoard.GetBoardSize()));
            var output2 = np.array(outputData2).astype(np.float32);
            var output = new NDarray[] { output1, output2 };


            var model = Keras.Models.Model.LoadModel(String.Format("gomoku_zero_{0}_{1}.keras", Constants.DEFAULT_BOARD_SIZE, networkGeneration));
            var history = model.Fit(input, output, epochs: _trainingEpochCount, batch_size: 16, verbose: 1, validation_split: 0.1f);

            //var optimizer = model.keras.optimizer;
            //optimizer.learning_rate.assign(0.1);
            //Console.WriteLine(history.HistoryLogs.ToString());

            model.Save(String.Format("gomoku_zero_{0}_{1}.keras", Constants.DEFAULT_BOARD_SIZE, networkGeneration + 1));

            //_gameHistories.Clear();
        }

        public void Run()
        {
            SetupPyEnv();
            SetupModel();

            Raylib.InitWindow(1280, 768, "TEST");

            //var model = Keras.Models.Model.LoadModel("test_model.keras");
            //var testModel = new Sequential();
            //testModel.Add(new Input(shape: new Numpy.Models.Shape(1)));
            //testModel.Add(new Dense(32, activation: "relu"));
            //testModel.Add(new Dense(64, activation: "relu"));
            //testModel.Add(new Dense(128, activation: "relu"));
            //testModel.Add(new Dense(128, activation: "relu"));
            //testModel.Add(new Dense(32, activation: "relu"));
            //testModel.Add(new Dense(1, activation: "sigmoid"));

            //testModel.Summary();


            ////var optimizer = new Keras.Optimizers.Adam(lr: 0.00002f);
            ///
            //var inputLayer = new Input(new Numpy.Models.Shape(3, 3, 3));
            //var netLayer = new Dense(32, activation: "relu").Set(inputLayer);
            //netLayer = new Dense(64, activation: "relu").Set(netLayer);
            //netLayer = new Dense(1, activation: "relu").Set(netLayer);

            //var testModel = new Keras.Models.Model(new BaseLayer[] { inputLayer }, new BaseLayer[] { netLayer });
            //testModel.Compile(optimizer: "Adadelta", loss: "binary_crossentropy", metrics: new string[] { "accuracy" });

            //const int TEST_DATA_COUNT = 100000;

            //var rnd = new Random();

            //int[] inputArray = new int[TEST_DATA_COUNT];
            //float[] outputArray = new float[TEST_DATA_COUNT];

            //int[] testIputArray = new int[TEST_DATA_COUNT];
            //float[] testOutputArray = new float[TEST_DATA_COUNT];
            //for (int i = 0; i < TEST_DATA_COUNT; i++)
            //{
            //    inputArray[i] = rnd.Next(100);
            //    outputArray[i] = inputArray[i] > 50 ? 1.0f : 0.0f;

            //    testIputArray[i] = rnd.Next(100);
            //    testOutputArray[i] = testIputArray[i] > 50 ? 1.0f : 0.0f;
            //}

            //NDarray input = np.array(inputArray);
            //NDarray output = np.array(outputArray);

            //var history = model.Fit(input, output, epochs: 50, batch_size: 10, verbose: 1);
            //Console.WriteLine(history.HistoryLogs.ToString());

            //var results = model.Evaluate(testIputArray, testOutputArray);
            //Console.WriteLine("test loss, test acc: " + results);

            //var result = model.Predict(new int[] { 20, 60, 33, 70, 10, 80, 99, 1 });
            //Console.WriteLine(result);

            //testModel.Save("test_model");

            // before your game loop
            //RlImgui.Setup(() => new System.Numerics.Vector2(650, 20), true);	// sets up ImGui with ether a dark or light default theme
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

                if (ImGui.Button("Train on DB"))
                {
                    _gameHistories.Clear();

                    var currentDirectory = Directory.GetCurrentDirectory();
                    for (int sample = 1; sample <= 4; sample++)
                    {
                        var games = Directory.GetFiles(currentDirectory + "\\Resources\\DataBase\\" + sample.ToString());
                        foreach (var gameName in games)
                        {
                            //if (_loadedGames.Contains(gameName))
                            //{
                            //    continue;
                            //}

                            //if (!gameName.EndsWith("0_11_7_0.psq"))
                            //{
                            //    continue;
                            //}

                            const int HISTORIES_COUNT = 4;
                            var gameHistories = new GameHistory[HISTORIES_COUNT];
                            for (int gameIndex = 0; gameIndex < HISTORIES_COUNT; gameIndex++)
                            {
                                gameHistories[gameIndex] = new GameHistory();
                                gameHistories[gameIndex].moves = new List<Constants.MovePosition>();
                            }

                            //Console.WriteLine(gameName);
                            int firstMove = 0;
                            int currentMove = 0;
                            var gameFile = File.ReadAllLines(gameName);
                            for (int i = 0; i < gameFile.Length; i++)
                            {
                                var move = gameFile[i].Split(",");
                                if (move == null)
                                {
                                    continue;
                                }

                                if (i == 0)
                                {
                                    for (int gameIndex = 0; gameIndex < HISTORIES_COUNT; gameIndex++)
                                    {
                                        gameHistories[gameIndex].startingColor = int.Parse(move[move.Length - 1]);
                                    }
                                    //currentMove = gameHistory.startingColor;

                                    //_gameBoard = new GameBoard(new ArrayGameBoardState(_boardSize), gameHistory.startingColor + 1);
                                }

                                if (i > 0 && i < gameFile.Length - 5)
                                {
                                    int x = int.Parse(move[0]) - 1;
                                    int y = int.Parse(move[1]) - 1;
                                    for (int gameIndex = 0; gameIndex < HISTORIES_COUNT; gameIndex++)
                                    {
                                        if (gameIndex == 0)
                                        {
                                            gameHistories[gameIndex].moves.Add(new Constants.MovePosition(x, y));
                                        }
                                        else if (gameIndex == 1)
                                        {
                                            gameHistories[gameIndex].moves.Add(new Constants.MovePosition(15 - 1 - x, y));
                                        }
                                        else if (gameIndex == 2)
                                        {
                                            gameHistories[gameIndex].moves.Add(new Constants.MovePosition(x, 15 - 1 - y));
                                        }
                                        else if (gameIndex == 3)
                                        {
                                            gameHistories[gameIndex].moves.Add(new Constants.MovePosition(15 - 1 - x, 15 - 1 - y));
                                        }
                                    }

                                    //_gameBoard.MakeMove(x, y);
                                }

                                if (i == gameFile.Length - 1)
                                {
                                    //gameHistory
                                    //Console.WriteLine(_gameBoard.GetCurrentTurnColor());
                                    //Console.WriteLine(_gameBoard.GetGameState());
                                    //Console.WriteLine(gameHistory.moves[gameHistory.moves.Count - 1]);
                                    //Console.WriteLine(move[0]);
                                    int winningColor = int.Parse(move[0]);
                                    if (winningColor == 1)
                                    {
                                        for (int gameIndex = 0; gameIndex < HISTORIES_COUNT; gameIndex++)
                                        {
                                            gameHistories[gameIndex].winningColor = 2;
                                            gameHistories[gameIndex].result = Constants.GameResult.WIN;
                                        }

                                    }
                                    else if (winningColor == 2)
                                    {
                                        for (int gameIndex = 0; gameIndex < HISTORIES_COUNT; gameIndex++)
                                        {
                                            gameHistories[gameIndex].winningColor = 1;
                                            gameHistories[gameIndex].result = Constants.GameResult.LOSE;
                                        }
                                    }
                                    else
                                    {
                                        for (int gameIndex = 0; gameIndex < HISTORIES_COUNT; gameIndex++)
                                        {
                                            gameHistories[gameIndex].winningColor = 0;
                                            gameHistories[gameIndex].result = Constants.GameResult.TIE;
                                        }
                                    }
                                }
                            }

                            for (int gameIndex = 0; gameIndex < HISTORIES_COUNT; gameIndex++)
                            {
                                if (gameHistories[gameIndex].moves.Count < 4)
                                {
                                    int k = 0;
                                }
                                _gameHistories.Add(gameHistories[gameIndex]);
                            }

                            //_loadedGames.Add(gameName);
                            //break;
                        }
                    }

                    TrainNetwork(_latestNetworkGeneration);
                    _latestNetworkGeneration = _latestNetworkGeneration + 1;
                    _agent1NetworkGeneration = _latestNetworkGeneration;
                    _agent2NetworkGeneration = _latestNetworkGeneration;
                }

                if (!_isTraining && ImGui.Button("Train"))
                {
                    _totalGames = 0;
                    _agent1Won = 0;

                    _isTraining = true;

                    Console.WriteLine("Starting new training game");

                    currentGameHistory = new GameHistory();
                    currentGameHistory.startingColor = Constants.CROSS_COLOR;
                    currentGameHistory.moves = new List<Constants.MovePosition>();

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
                            if (_agent1Type == AgentType.CNN && _agent2Type == AgentType.CNN)
                            {
                                if (_agent1Won > _agent2Won)
                                {
                                    TrainNetwork(_agent1NetworkGeneration);
                                    _latestNetworkGeneration = _agent1NetworkGeneration + 1;
                                    _agent1NetworkGeneration = _latestNetworkGeneration;
                                    _agent2NetworkGeneration = _latestNetworkGeneration;
                                } else
                                {
                                    TrainNetwork(_agent2NetworkGeneration);
                                    _latestNetworkGeneration = _agent2NetworkGeneration + 1;
                                    _agent1NetworkGeneration = _latestNetworkGeneration;
                                    _agent2NetworkGeneration = _latestNetworkGeneration;
                                }
                                
                            }
                            else if (_agent1Type == AgentType.CNN)
                            {
                                TrainNetwork(_agent1NetworkGeneration);
                                _latestNetworkGeneration = _agent1NetworkGeneration + 1;
                                _agent1NetworkGeneration = _latestNetworkGeneration;
                                _agent2NetworkGeneration = _latestNetworkGeneration;
                            }
                            else if (_agent1Type == AgentType.CNN)
                            {
                                TrainNetwork(_agent2NetworkGeneration);
                                _latestNetworkGeneration = _agent2NetworkGeneration + 1;
                                _agent1NetworkGeneration = _latestNetworkGeneration;
                                _agent2NetworkGeneration = _latestNetworkGeneration;
                            } else
                            {
                                TrainNetwork(_latestNetworkGeneration);
                                _latestNetworkGeneration = _latestNetworkGeneration + 1;
                                _agent1NetworkGeneration = _latestNetworkGeneration;
                                _agent2NetworkGeneration = _latestNetworkGeneration;
                            }
                                
                            _agent1Won = 0;
                            _agent2Won = 0;
                            _totalGames = 0;
                        }
                        Console.WriteLine("Starting new training game");

                        currentGameHistory = new GameHistory();
                        currentGameHistory.startingColor = Constants.CROSS_COLOR;
                        currentGameHistory.moves = new List<Constants.MovePosition>();

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

                            currentGameHistory.moves.Add(new Constants.MovePosition(move.X, move.Y));
                        }
                    } else if (_gameBoard.GetCurrentTurnColor() == Constants.ZERO_COLOR && _agent2PlayoutsBeforeMoveSelection <= _agent2.GetCurrentPlayoutsCount())
                    {
                        var move = _agent2.GetBestMove();
                        if (_gameBoard.MakeMove(move.X, move.Y))
                        {
                            _agent1.SelectNextNode(move.X, move.Y, Constants.ZERO_COLOR, false);
                            _agent2.SelectNextNode(move.X, move.Y, Constants.ZERO_COLOR, false);

                            currentGameHistory.moves.Add(new Constants.MovePosition(move.X, move.Y));
                        }
                    }

                    if (_gameBoard.GetGameState() != Constants.GameResult.IN_PROGRESS)
                    {
                        currentGameHistory.result = _gameBoard.GetGameState();
                        if (_gameBoard.GetGameState() != Constants.GameResult.TIE)
                        {
                            currentGameHistory.winningColor = _gameBoard.GetCurrentTurnColor();
                        }
                        else
                        {
                            currentGameHistory.winningColor = 0;
                        }
                        
                        _gameHistories.Add(currentGameHistory);

                        var reversedGameHistory = new GameHistory();
                        reversedGameHistory.result = currentGameHistory.result;
                        reversedGameHistory.startingColor = currentGameHistory.startingColor;
                        reversedGameHistory.winningColor = currentGameHistory.winningColor;
                        reversedGameHistory.moves = new List<Constants.MovePosition>();
                        foreach (var move in currentGameHistory.moves)
                        {
                            reversedGameHistory.moves.Add(new Constants.MovePosition(move.Y, move.X));
                        }
                        _gameHistories.Add(reversedGameHistory);

                        var flippedX = new GameHistory();
                        flippedX.result = currentGameHistory.result;
                        flippedX.startingColor = currentGameHistory.startingColor;
                        flippedX.winningColor = currentGameHistory.winningColor;
                        flippedX.moves = new List<Constants.MovePosition>();
                        foreach (var move in currentGameHistory.moves)
                        {
                            flippedX.moves.Add(new Constants.MovePosition(_gameBoard.GetBoardSize() - 1 - move.X, move.Y));
                        }
                        _gameHistories.Add(flippedX);

                        var flippedY = new GameHistory();
                        flippedY.result = currentGameHistory.result;
                        flippedY.startingColor = currentGameHistory.startingColor;
                        flippedY.winningColor = currentGameHistory.winningColor;
                        flippedY.moves = new List<Constants.MovePosition>();
                        foreach (var move in currentGameHistory.moves)
                        {
                            flippedY.moves.Add(new Constants.MovePosition(move.X, _gameBoard.GetBoardSize() - 1 - move.Y));
                        }
                        _gameHistories.Add(flippedY);

                        var flippedReversedX = new GameHistory();
                        flippedReversedX.result = currentGameHistory.result;
                        flippedReversedX.startingColor = currentGameHistory.startingColor;
                        flippedReversedX.winningColor = currentGameHistory.winningColor;
                        flippedReversedX.moves = new List<Constants.MovePosition>();
                        foreach (var move in currentGameHistory.moves)
                        {
                            flippedReversedX.moves.Add(new Constants.MovePosition(move.Y, _gameBoard.GetBoardSize() - 1 - move.X));
                        }
                        _gameHistories.Add(flippedReversedX);

                        var flippedRevertedY = new GameHistory();
                        flippedRevertedY.result = currentGameHistory.result;
                        flippedRevertedY.startingColor = currentGameHistory.startingColor;
                        flippedRevertedY.winningColor = currentGameHistory.winningColor;
                        flippedRevertedY.moves = new List<Constants.MovePosition>();
                        foreach (var move in currentGameHistory.moves)
                        {
                            flippedRevertedY.moves.Add(new Constants.MovePosition(_gameBoard.GetBoardSize() - 1 - move.Y, move.X));
                        }
                        _gameHistories.Add(flippedRevertedY);

                        _totalGames++;
                        if (currentGameHistory.result == Constants.GameResult.WIN && currentGameHistory.winningColor == Constants.CROSS_COLOR)
                        {
                            _agent1Won++;
                        }
                        if (currentGameHistory.result == Constants.GameResult.WIN && currentGameHistory.winningColor == Constants.ZERO_COLOR)
                        {
                            _agent2Won++;
                        }
                    }

                    ImGui.Text("Agent 1 wins: " + _agent1Won);
                    ImGui.Text("Agent 2 wins: " + _agent2Won);
                    ImGui.Text("Total Games: " + _totalGames);
                }

                if (_isTraining)
                {
                    ImGui.Text("Training games: " + _gameHistories.Count);
                    ImGui.Text("Agent 1 playouts: " + _agent1.GetCurrentPlayoutsCount());
                    ImGui.Text("Agent 2 playouts: " + _agent2.GetCurrentPlayoutsCount());
                }

                ImGui.SliderInt("Epoch count", ref _trainingEpochCount, 1, 100);

                if (_isTraining && ImGui.SliderInt("Agent1 playouts", ref _agent1PlayoutsBeforeMoveSelection, 1, 10000)) { }
                if (_isTraining && ImGui.SliderInt("Agent2 playouts", ref _agent2PlayoutsBeforeMoveSelection, 1, 10000)) { }

                if (_isTraining && ImGui.Button("Stop training"))
                {
                    _isTraining = false;
                }

                if (_isTraining && ImGui.Button("Learn from training"))
                {
                    _isTraining = false;
                    if (_agent1Type == AgentType.CNN && _agent2Type == AgentType.CNN)
                    {
                        if (_agent1Won > _agent2Won)
                        {
                            TrainNetwork(_agent1NetworkGeneration);
                            _latestNetworkGeneration = _agent1NetworkGeneration + 1;
                            _agent1NetworkGeneration = _latestNetworkGeneration;
                            _agent2NetworkGeneration = _latestNetworkGeneration;
                        }
                        else
                        {
                            TrainNetwork(_agent2NetworkGeneration);
                            _latestNetworkGeneration = _agent2NetworkGeneration + 1;
                            _agent1NetworkGeneration = _latestNetworkGeneration;
                            _agent2NetworkGeneration = _latestNetworkGeneration;
                        }

                    }
                    else if (_agent1Type == AgentType.CNN)
                    {
                        TrainNetwork(_agent1NetworkGeneration);
                        _latestNetworkGeneration = _agent1NetworkGeneration + 1;
                        _agent1NetworkGeneration = _latestNetworkGeneration;
                        _agent2NetworkGeneration = _latestNetworkGeneration;
                    }
                    else if (_agent1Type == AgentType.CNN)
                    {
                        TrainNetwork(_agent2NetworkGeneration);
                        _latestNetworkGeneration = _agent2NetworkGeneration + 1;
                        _agent1NetworkGeneration = _latestNetworkGeneration;
                        _agent2NetworkGeneration = _latestNetworkGeneration;
                    }
                    else
                    {
                        TrainNetwork(_latestNetworkGeneration);
                        _latestNetworkGeneration = _latestNetworkGeneration + 1;
                        _agent1NetworkGeneration = _latestNetworkGeneration;
                        _agent2NetworkGeneration = _latestNetworkGeneration;
                    }

                    _agent1Won = 0;
                    _agent2Won = 0;
                    _totalGames = 0;
                }

                if (ImGui.Button("New Game"))
                {
                    _gameBoard = new GameBoard(new ArrayGameBoardState(_boardSize), Constants.CROSS_COLOR);
                    var estimator = new CNNEstimator(false);
                    estimator.LoadModel(String.Format("gomoku_zero_{0}_{1}.keras", Constants.DEFAULT_BOARD_SIZE, _latestNetworkGeneration));
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
