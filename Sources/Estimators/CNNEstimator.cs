using ImGuiNET;
using MathNet.Numerics.Distributions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Xml.Linq;
using static GomokuNN.Sources.Constants;

namespace GomokuNN.Sources.Estimators
{
    internal class CNNEstimator : IGameEstimator
    {

        ArrayGameBoardState _state = new ArrayGameBoardState(3);
        MCTSTreeNode _root, _node;
        IncrementalMovesPolicy _policy;
        float _mctsExplorationConst = 5.0f;

        bool _isTraining = false;
        bool _isEstimationInProgress = false;
        Task? _estimationTask = null;
        Random _rndGenerator = new Random();

        bool _debugDrawBestMove = false;
        bool _debugDrawProbabilities = false;

        InferenceSession? _inferenceSession = null;

        public CNNEstimator(bool isTraining)
        {
            _isTraining = isTraining;
        }

        public void InitFromState(IGameBoardState gameState, int turnColor, int estimatorColor)
        {
            var boardSize = gameState.GetBoardSize();
            _state.Init(boardSize);
            for (int x = 0; x < boardSize; x++)
            {
                for (int y = 0; y < boardSize; y++)
                {
                    int state = gameState.GetCellState(x, y);
                    _state.SetCellState(x, y, state);
                }
            }

            _policy = new IncrementalMovesPolicy(CNN_MOVE_POLICY_EXPANSION);
            _policy.Init(_state);

            var availableMoves = _policy.GetHashedPositions();
            _root = new MCTSTreeNode(null)
            {
                MovePosition = new MovePosition(),
                MoveColor = NULL_COLOR
            };

            float[] inputData = new float[_state.GetBoardSize() * _state.GetBoardSize() * 4];
            int boardOffset = _state.GetBoardSize() * _state.GetBoardSize();
            int dataOffset = 0;

            var currentNode = _root;
            for (int i = 0; i < _state.GetBoardSize() * _state.GetBoardSize(); i++)
            {
                var value = _state.GetRawCellState(i);
                inputData[i + dataOffset] = value == CROSS_COLOR ? 1 : 0;
                inputData[i + boardOffset + dataOffset] = value == ZERO_COLOR ? 1 : 0;

                inputData[i + boardOffset * 3 + dataOffset] = turnColor == CROSS_COLOR ? 1 : 0;
            }

            var inputTensor = new DenseTensor<float>(inputData, new int[] { 1, 4, _state.GetBoardSize(), _state.GetBoardSize() });
            var modelInput = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("input_layer", inputTensor) };

            var output = _inferenceSession.Run(modelInput).ToArray();
            var policyOutput = output[0].AsTensor<float>();

            int index = 0;
            foreach (var availableMove in availableMoves)
            {
                int posX = _policy.GetUnhashedPositionX(availableMove);
                int posY = _policy.GetUnhashedPositionY(availableMove);

                _root.Leafs.Add(new MCTSTreeNode(_root)
                {
                    PolicyProbability = policyOutput.ElementAt<float>(posY * _state.GetBoardSize() + posX),
                    MovePosition = new MovePosition(posX, posY),
                    MoveColor = turnColor
                });

                index++;
            }

            _node = _root;
        }

        public void LoadModel(string modelPath)
        {
            _inferenceSession = CNNModelCache.Instance.LoadModel(modelPath + ".onnx");
        }

        public void SaveModel(string modelPath)
        {

            //_model.Save(modelPath, true, "keras");
        }

        public bool SelectNextNode(int x, int y, int color, bool cleanUpTree)
        {
            bool isEstimationInProgress = IsEstimationInProgress();
            StopEstimation();
            if (isEstimationInProgress)
            {
                StartEstimation();
            }

            if (_node == null)
            {
                return false;
            }

            if (_state.GetCellState(x, y) != EMPTY_COLOR)
            {
                return false;
            }

            _state.SetCellState(x, y, color);
            IGameBoardState state = _state;

            _policy.Update(x, y, ref state);

            MCTSTreeNode? nextNode = null;
            foreach (var node in _node.Leafs)
            {
                if (node.MovePosition.X == x && node.MovePosition.Y == y && node.MoveColor == color)
                {
                    nextNode = node;
                    break;
                }
            }

            if (nextNode == null)
            {
                var newNode = new MCTSTreeNode(_node)
                {
                    MovePosition = new MovePosition(x, y),
                    MoveColor = color
                };
                _node.Leafs.Add(newNode);

                nextNode = newNode;
            }

            _node = nextNode;
            return true;
        }

        public void StartEstimation()
        {
            if (!HasContiniousEstimationSupport())
            {
                return;
            }

            StopEstimation();

            _isEstimationInProgress = true;
            _estimationTask = Task.Factory.StartNew(() =>
            {
                while (_isEstimationInProgress)
                {
                    EstimateOnce();
                }
            });
        }

        public bool IsEstimationInProgress()
        {
            return _isEstimationInProgress;
        }

        public bool HasContiniousEstimationSupport()
        {
            return false;
        }

        public void EstimateOnce()
        {
            RunMTCSSimulation();
        }

        public void StopEstimation()
        {
            _isEstimationInProgress = false;
            _estimationTask?.Wait();
            _estimationTask = null;
        }


        public void RunMTCSSimulation()
        {
            // Selection
            // Select tree node with best value until leaf node is reached
            List<int> bestNodes = new List<int>();

            var selectedNode = _node;
            ArrayGameBoardState state = new ArrayGameBoardState(_state.GetBoardSize());
            state.Copy(_state);
            while (!selectedNode.IsEndPoint)
            {
                double[] dirichlet = null;

                if (_isTraining)
                {
                    var alpha = new double[selectedNode.Leafs.Count];
                    Array.Fill(alpha, 0.15);
                    dirichlet = Dirichlet.Sample(_rndGenerator, alpha);
                }

                MCTSTreeNode? nextNode = null;
                var bestSelectionValue = double.MinValue;
                bestNodes.Clear();
                for (int index = 0; index < selectedNode.Leafs.Count(); index++)
                {
                    var node = selectedNode.Leafs[index];
                    var selectionValue = _isTraining ? GetNodeSelectionValue(ref selectedNode, ref node, index, dirichlet) : GetNodeSelectionValue(ref selectedNode, ref node);
                    if (node.IsTerminal)
                    {
                        nextNode = node;
                        break;
                    }

                    if (selectionValue > bestSelectionValue)
                    {
                        nextNode = node;
                        bestSelectionValue = selectionValue;

                        bestNodes.Clear();
                        bestNodes.Add(index);
                    }
                    else if (selectionValue == bestSelectionValue)
                    {
                        bestNodes.Add(index);
                    }
                }

                if (bestNodes.Count <= 1)
                {
                    selectedNode = nextNode;
                } 
                else
                {
                    var rndNodeIndex = _rndGenerator.Next(bestNodes.Count);
                    selectedNode = selectedNode.Leafs[bestNodes[rndNodeIndex]];
                }
                state.SetCellState(selectedNode.MovePosition.X, selectedNode.MovePosition.Y, selectedNode.MoveColor);
            }

            var movesPolicy = new IncrementalMovesPolicy(CNN_MOVE_POLICY_EXPANSION);
            movesPolicy.Init(state);

            var availableMovesAtSelectedNode = movesPolicy.GetHashedPositions();

            float score;
            var gameResultAfterSelection = selectedNode.IsTerminal ? GameResult.WIN : GameResultEstimator.EstimateResult(state, selectedNode.MovePosition.X, selectedNode.MovePosition.Y);
            if (gameResultAfterSelection != GameResult.IN_PROGRESS)
            {
                // terminal node
                score = ConvertGameResultToScore(gameResultAfterSelection);
                availableMovesAtSelectedNode.Clear();

                // Backpropagate
                selectedNode.IsTerminal = gameResultAfterSelection == GameResult.WIN;

                var propagationNode = selectedNode;
                while (propagationNode != null)
                {
                    propagationNode.PlayoutsCount++;
                    propagationNode.PlayoutScore += score;
                    propagationNode.WinProbability = (float)propagationNode.PlayoutScore / propagationNode.PlayoutsCount;

                    propagationNode = propagationNode.Parent;
                    score = score * -1.0f;
                }
            }
            else
            {
                float[] inputData = new float[state.GetBoardSize() * state.GetBoardSize() * 4];
                int boardOffset = state.GetBoardSize() * state.GetBoardSize();
                int dataOffset = 0;

                var currentNode = selectedNode;
                for (int i = 0; i < state.GetBoardSize() * state.GetBoardSize(); i++)
                {
                    var value = state.GetRawCellState(i);
                    inputData[i + dataOffset] = value == CROSS_COLOR ? 1 : 0;
                    inputData[i + boardOffset + dataOffset] = value == ZERO_COLOR ? 1 : 0;

                    inputData[i + boardOffset * 3 + dataOffset] = RotateColor(selectedNode.MoveColor) == CROSS_COLOR ? 1 : 0;
                }

                inputData[boardOffset * 2 + state.GetPositionHash(selectedNode.MovePosition.X, selectedNode.MovePosition.Y)] = 1;

                //var input = np.array(inputData).reshape((1, 4, _state.GetBoardSize(), _state.GetBoardSize()));

                //var result = _model.PredictMultipleOutputs(input, verbose: 0);

                var inputTensor = new DenseTensor<float>(inputData, new int[] { 1, 4, _state.GetBoardSize(), _state.GetBoardSize() });
                var modelInput = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("input_layer", inputTensor) };

                var output = _inferenceSession.Run(modelInput).ToArray();
                var policyOutput = output[0].AsTensor<float>();
                var valueOutput = output[1].AsTensor<float>();

                //var policyArray = result[0].GetData<float>();
                //var resultArray = result[1].GetData<float>();
                score = valueOutput[0];

                // Backpropagate
                var propagationNode = selectedNode;
                while (propagationNode != null)
                {
                    propagationNode.PlayoutsCount++;
                    propagationNode.PlayoutScore += score;
                    propagationNode.WinProbability = (float)propagationNode.PlayoutScore / propagationNode.PlayoutsCount;

                    propagationNode = propagationNode.Parent;
                    score = score * -1.0f;
                }

                if (selectedNode.IsEndPoint && availableMovesAtSelectedNode.Count > 0)
                {
                    // expand node
                    int turnColor = RotateColor(selectedNode.MoveColor);

                    foreach (var availableMove in availableMovesAtSelectedNode)
                    {
                        int posX = _policy.GetUnhashedPositionX(availableMove);
                        int posY = _policy.GetUnhashedPositionY(availableMove);

                        var newLeaf = new MCTSTreeNode(selectedNode)
                        {
                            PolicyProbability = policyOutput.ElementAt<float>(posY * state.GetBoardSize() + posX),
                            MovePosition = new MovePosition(posX, posY),
                            MoveColor = turnColor
                        };

                        selectedNode.Leafs.Add(newLeaf);

                        ArrayGameBoardState newNodeState = new ArrayGameBoardState(_state.GetBoardSize());
                        newNodeState.Copy(state);
                        newNodeState.SetCellState(posX, posY, turnColor);

                        var gameResult = GameResultEstimator.EstimateResult(newNodeState, posX, posY);
                        if (gameResult == GameResult.WIN)
                        {
                            newLeaf.IsTerminal = true;
                        }
                    }
                }
            }
        }

        public float ConvertGameResultToScore(GameResult result)
        {
            return result == GameResult.WIN ? 1.0f : result == GameResult.LOSE ? -1.0f : 0.0f;
        }

        public double GetNodeSelectionValue(ref MCTSTreeNode parent, ref MCTSTreeNode child)
        {
            if (child.PlayoutsCount == 0 && child.PolicyProbability < 0.0001 && parent.PlayoutsCount > 1)
            {
                // DO SOMETING???
                //return _mctsExplorationConst * child.PolicyProbability * Math.Sqrt(parent.PlayoutsCount) / (1.0f + child.PlayoutsCount);
            }

            return child.WinProbability + _mctsExplorationConst * child.PolicyProbability * Math.Sqrt(parent.PlayoutsCount) / (1.0f + child.PlayoutsCount);
        }

        public double GetNodeSelectionValue(ref MCTSTreeNode parent, ref MCTSTreeNode child, int nodeIndex, double[] dirichletNoise)
        {
            const double epsilon = 0.25;
            var policyProbability = (1 - epsilon) * child.PolicyProbability + epsilon * dirichletNoise[nodeIndex];

            if (child.PlayoutsCount == 0)
            {
                return _mctsExplorationConst * policyProbability * Math.Sqrt(parent.PlayoutsCount) / (1.0f + child.PlayoutsCount);
            }

            return child.WinProbability + _mctsExplorationConst * policyProbability * Math.Sqrt(parent.PlayoutsCount) / (1.0f + child.PlayoutsCount);
        }

        public void SetExplorationConst(float value)
        {
            _mctsExplorationConst = value;
        }

        public void OnDrawCell(int x, int y, ref RenderHelper renderHelper)
        {
            if (_debugDrawBestMove)
            {
                var bestMove = GetBestMove();
                if (x == bestMove.X && y == bestMove.Y)
                {
                    renderHelper.DrawAINextMoveBoardCell(RotateColor(_node.MoveColor), x, y);
                }
            }

            if (_debugDrawProbabilities)
            {
                foreach (var leaf in _node.Leafs)
                {
                    if (x == leaf.MovePosition.X && y == leaf.MovePosition.Y)
                    {
                        renderHelper.DrawPolicyText(leaf.PolicyProbability.ToString("F2"), x, y);
                        renderHelper.DrawWinrateText(leaf.WinProbability.ToString("F2"), x, y);
                    }
                }
            }
        }

        public void DebugMenuDraw(ref GameBoard gameBoard)
        {
            if (_node == null)
            {
                return;
            }

            ImGui.Text("Playouts count: " + _node.PlayoutsCount.ToString());
            if (_node.Leafs != null && _node.Leafs.Count > 0)
            {
                ImGui.Text("Node win probability: " + _node.Leafs.Average(leaf => leaf.WinProbability).ToString());
            }

            //ImGui.Checkbox("Show probabilities", ref _showAIProbabilities);
            ImGui.Checkbox("Show best move", ref _debugDrawBestMove);
            ImGui.Checkbox("Show probabilities", ref _debugDrawProbabilities);

            if (ImGui.Button("Move back"))
            {
                if (_node != null && _node.Parent != null)
                {
                    _state.UndoSetCellState(_node.MovePosition.X, _node.MovePosition.Y);
                    gameBoard.GetBoardState().SetCellState(_node.MovePosition.X, _node.MovePosition.Y, EMPTY_COLOR);
                    gameBoard.SetCurrentTurnColor(_node.MoveColor);

                    _policy.Init(_state);

                    _node = _node.Parent;
                }
            }
        }

        public int GetCurrentPlayoutsCount()
        {
            if (_node == null)
            {
                return 0;
            }

            return _node.PlayoutsCount;
        }

        public List<TrainingSample> GetTrainingSamples(int winnerColor, ref HashSet<long> knownPositions)
        {
            List<TrainingSample> samples = new List<TrainingSample>();

            FillTrainingSamples(ref samples, winnerColor, ref knownPositions);

            return samples;
        }

        public void FillTrainingSamples(ref List<TrainingSample> samples, int winnerColor, ref HashSet<long> knownPositions)
        {
            ArrayGameBoardState propagationState = new ArrayGameBoardState(_state.GetBoardSize());
            propagationState.Copy(_state);

            var currentNode = _node;
            MCTSTreeNode? nextNode = null;
            while (currentNode != null && currentNode.Parent != null)
            {
                if (currentNode.MovePosition.X < 0 || currentNode.MovePosition.Y < 0)
                {
                    break;
                }

                var currentColor = RotateColor(currentNode.MoveColor);
                var reward = currentColor == winnerColor ? 1.0f : -1.0f;

                long zobristHash = propagationState.GetBoardStateHash();
                if (!knownPositions.Contains(zobristHash))
                {
                    var winRate = Math.Clamp((nextNode == null ? -currentNode.WinProbability : nextNode.WinProbability) + reward, -1.0f, 1.0f);
                    var stateSample = new TrainingSample(propagationState, nextNode == null ? new MovePosition() : nextNode.MovePosition, currentNode.MovePosition, currentColor, nextNode == null ? -currentNode.WinProbability : nextNode.WinProbability);
                    foreach (var leaf in currentNode.Leafs)
                    {
                        stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), leaf.MovePosition.X, leaf.MovePosition.Y, (float)leaf.PlayoutsCount / (float)currentNode.PlayoutsCount);
                    }

                    // softmax
                    if (currentNode.Leafs.Count > 0)
                    {
                        float max = stateSample.networkOutput.Sum();
                        for (int i = 0; i < stateSample.networkOutput.Length; i++)
                        {
                            stateSample.networkOutput[i] = stateSample.networkOutput[i] / max;
                        }
                    }

                    samples.Add(stateSample);
                    knownPositions.Add(propagationState.GetBoardStateHash());
                }

                var rotatedPos = new Constants.MovePosition();
                ArrayGameBoardState rotatedState = new ArrayGameBoardState(_state.GetBoardSize());
                for (int y = 0; y < _state.GetBoardSize(); y++)
                {
                    for (int x = 0; x < _state.GetBoardSize(); x++)
                    {
                        rotatedState.SetCellState(y, propagationState.GetBoardSize() - 1 - x, propagationState.GetCellState(x, y));
                    }
                }

                long rotatedZobristHash = propagationState.GetBoardStateHash();
                if (!knownPositions.Contains(rotatedZobristHash))
                {
                    rotatedPos = new Constants.MovePosition(currentNode.MovePosition.Y, propagationState.GetBoardSize() - 1 - currentNode.MovePosition.X);
                    var stateSample = new TrainingSample(rotatedState, nextNode == null ? new MovePosition() : nextNode.MovePosition, rotatedPos, currentColor, nextNode == null ? -currentNode.WinProbability : nextNode.WinProbability);
                    foreach (var leaf in currentNode.Leafs)
                    {
                        //if (leaf != nextNode)
                        //{
                        stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), leaf.MovePosition.Y, propagationState.GetBoardSize() - 1 - leaf.MovePosition.X, (float)leaf.PlayoutsCount / (float)currentNode.PlayoutsCount);
                        //}
                        //else
                        //{
                        //    var rewardValue = Math.Min(Math.Max(0.01f, reward + leaf.PolicyProbability), 1.0f);
                        //    stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), leaf.MovePosition.Y, propagationState.GetBoardSize() - 1 - leaf.MovePosition.X, rewardValue);
                        //}
                    }

                    // softmax
                    if (currentNode.Leafs.Count > 0)
                    {
                        float max = stateSample.networkOutput.Sum();
                        for (int i = 0; i < stateSample.networkOutput.Length; i++)
                        {
                            stateSample.networkOutput[i] = stateSample.networkOutput[i] / max;
                        }
                    }
                    samples.Add(stateSample);
                }


                rotatedState = new ArrayGameBoardState(_state.GetBoardSize());
                for (int y = 0; y < _state.GetBoardSize(); y++)
                {
                    for (int x = 0; x < _state.GetBoardSize(); x++)
                    {
                        rotatedState.SetCellState(propagationState.GetBoardSize() - 1 - x, propagationState.GetBoardSize() - 1 - y, propagationState.GetCellState(x, y));
                    }
                }

                rotatedZobristHash = propagationState.GetBoardStateHash();
                if (!knownPositions.Contains(rotatedZobristHash))
                {
                    rotatedPos = new Constants.MovePosition(propagationState.GetBoardSize() - 1 - currentNode.MovePosition.X, propagationState.GetBoardSize() - 1 - currentNode.MovePosition.Y);
                    var stateSample = new TrainingSample(rotatedState, nextNode == null ? new MovePosition() : nextNode.MovePosition, rotatedPos, currentColor, nextNode == null ? -currentNode.WinProbability : nextNode.WinProbability);
                    foreach (var leaf in currentNode.Leafs)
                    {
                        //if (leaf != nextNode)
                        //{
                        stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), propagationState.GetBoardSize() - 1 - leaf.MovePosition.X, propagationState.GetBoardSize() - 1 - leaf.MovePosition.Y, (float)leaf.PlayoutsCount / (float)currentNode.PlayoutsCount);
                        //}
                        //else
                        //{
                        //    var rewardValue = Math.Min(Math.Max(0.01f, reward + leaf.PolicyProbability), 1.0f);
                        //    stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), propagationState.GetBoardSize() - 1 - leaf.MovePosition.X, propagationState.GetBoardSize() - 1 - leaf.MovePosition.Y, rewardValue);
                        //}
                    }

                    // softmax
                    if (currentNode.Leafs.Count > 0)
                    {
                        float max = stateSample.networkOutput.Sum();
                        for (int i = 0; i < stateSample.networkOutput.Length; i++)
                        {
                            stateSample.networkOutput[i] = stateSample.networkOutput[i] / max;
                        }
                    }
                    samples.Add(stateSample);
                }

                //
                rotatedState = new ArrayGameBoardState(_state.GetBoardSize());
                for (int y = 0; y < _state.GetBoardSize(); y++)
                {
                    for (int x = 0; x < _state.GetBoardSize(); x++)
                    {
                        rotatedState.SetCellState(propagationState.GetBoardSize() - 1 - y, x, propagationState.GetCellState(x, y));
                    }
                }

                rotatedZobristHash = propagationState.GetBoardStateHash();
                if (!knownPositions.Contains(rotatedZobristHash))
                {
                    rotatedPos = new Constants.MovePosition(propagationState.GetBoardSize() - 1 - currentNode.MovePosition.Y, currentNode.MovePosition.X);
                    var stateSample = new TrainingSample(rotatedState, nextNode == null ? new MovePosition() : nextNode.MovePosition, rotatedPos, currentColor, nextNode == null ? -currentNode.WinProbability : nextNode.WinProbability);
                    foreach (var leaf in currentNode.Leafs)
                    {
                        //if (leaf != nextNode)
                        //{
                        stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), propagationState.GetBoardSize() - 1 - leaf.MovePosition.Y, leaf.MovePosition.X, (float)leaf.PlayoutsCount / (float)currentNode.PlayoutsCount);
                        //}
                        //else
                        //{
                        //    var rewardValue = Math.Min(Math.Max(0.01f, reward + leaf.PolicyProbability), 1.0f);
                        //    stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), propagationState.GetBoardSize() - 1 - leaf.MovePosition.Y, leaf.MovePosition.X, rewardValue);
                        //}
                    }

                    // softmax
                    if (currentNode.Leafs.Count > 0)
                    {
                        float max = stateSample.networkOutput.Sum();
                        for (int i = 0; i < stateSample.networkOutput.Length; i++)
                        {
                            stateSample.networkOutput[i] = stateSample.networkOutput[i] / max;
                        }
                    }
                    samples.Add(stateSample);
                }

                //
                rotatedState = new ArrayGameBoardState(_state.GetBoardSize());
                for (int y = 0; y < _state.GetBoardSize(); y++)
                {
                    for (int x = 0; x < _state.GetBoardSize(); x++)
                    {
                        rotatedState.SetCellState(propagationState.GetBoardSize() - 1 - x, y, propagationState.GetCellState(x, y));
                    }
                }

                rotatedZobristHash = propagationState.GetBoardStateHash();
                if (!knownPositions.Contains(rotatedZobristHash))
                {
                    rotatedPos = new Constants.MovePosition(propagationState.GetBoardSize() - 1 - currentNode.MovePosition.X, currentNode.MovePosition.Y);
                    var stateSample = new TrainingSample(rotatedState, nextNode == null ? new MovePosition() : nextNode.MovePosition, rotatedPos, currentColor, nextNode == null ? -currentNode.WinProbability : nextNode.WinProbability);
                    foreach (var leaf in currentNode.Leafs)
                    {
                        //if (leaf != nextNode)
                        //{
                        stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), propagationState.GetBoardSize() - 1 - leaf.MovePosition.X, leaf.MovePosition.Y, (float)leaf.PlayoutsCount / (float)currentNode.PlayoutsCount);
                        //    }
                        //else
                        //{
                        //    var rewardValue = Math.Min(Math.Max(0.01f, reward + leaf.PolicyProbability), 1.0f);
                        //    stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), propagationState.GetBoardSize() - 1 - leaf.MovePosition.X, leaf.MovePosition.Y, rewardValue);
                        //}
                    }

                    // softmax
                    if (currentNode.Leafs.Count > 0)
                    {
                        float max = stateSample.networkOutput.Sum();
                        for (int i = 0; i < stateSample.networkOutput.Length; i++)
                        {
                            stateSample.networkOutput[i] = stateSample.networkOutput[i] / max;
                        }
                    }
                    samples.Add(stateSample);
                }

                rotatedState = new ArrayGameBoardState(_state.GetBoardSize());
                for (int y = 0; y < _state.GetBoardSize(); y++)
                {
                    for (int x = 0; x < _state.GetBoardSize(); x++)
                    {
                        rotatedState.SetCellState(y, x, propagationState.GetCellState(x, y));
                    }
                }

                rotatedZobristHash = propagationState.GetBoardStateHash();
                if (!knownPositions.Contains(rotatedZobristHash))
                {
                    rotatedPos = new Constants.MovePosition(currentNode.MovePosition.Y, currentNode.MovePosition.X);
                    var stateSample = new TrainingSample(rotatedState, nextNode == null ? new MovePosition() : nextNode.MovePosition, rotatedPos, currentColor, nextNode == null ? -currentNode.WinProbability : nextNode.WinProbability);
                    foreach (var leaf in currentNode.Leafs)
                    {
                        //if (leaf != nextNode)
                        //{
                        stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), leaf.MovePosition.Y, leaf.MovePosition.X, (float)leaf.PlayoutsCount / (float)currentNode.PlayoutsCount);
                        //    }
                        //else
                        //{
                        //    var rewardValue = Math.Min(Math.Max(0.01f, reward + leaf.PolicyProbability), 1.0f);
                        //    stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), leaf.MovePosition.Y, leaf.MovePosition.X, rewardValue);
                        //}
                    }

                    // softmax
                    if (currentNode.Leafs.Count > 0)
                    {
                        float max = stateSample.networkOutput.Sum();
                        for (int i = 0; i < stateSample.networkOutput.Length; i++)
                        {
                            stateSample.networkOutput[i] = stateSample.networkOutput[i] / max;
                        }
                    }
                    samples.Add(stateSample);
                }

                rotatedState = new ArrayGameBoardState(_state.GetBoardSize());
                for (int y = 0; y < _state.GetBoardSize(); y++)
                {
                    for (int x = 0; x < _state.GetBoardSize(); x++)
                    {
                        rotatedState.SetCellState(propagationState.GetBoardSize() - 1 - y, propagationState.GetBoardSize() - 1 - x, propagationState.GetCellState(x, y));
                    }
                }

                rotatedZobristHash = propagationState.GetBoardStateHash();
                if (!knownPositions.Contains(rotatedZobristHash))
                {
                    rotatedPos = new Constants.MovePosition(propagationState.GetBoardSize() - 1 - currentNode.MovePosition.Y, propagationState.GetBoardSize() - 1 - currentNode.MovePosition.X);
                    var stateSample = new TrainingSample(rotatedState, nextNode == null ? new MovePosition() : nextNode.MovePosition, rotatedPos, currentColor, nextNode == null ? -currentNode.WinProbability : nextNode.WinProbability);
                    foreach (var leaf in currentNode.Leafs)
                    {
                        //if (leaf != nextNode)
                        //{
                        stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), propagationState.GetBoardSize() - 1 - leaf.MovePosition.Y, propagationState.GetBoardSize() - 1 - leaf.MovePosition.X, (float)leaf.PlayoutsCount / (float)currentNode.PlayoutsCount);
                        //    }
                        //else
                        //{
                        //    var rewardValue = Math.Min(Math.Max(0.01f, reward + leaf.PolicyProbability), 1.0f);
                        //    stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), leaf.MovePosition.Y, leaf.MovePosition.X, rewardValue);
                        //}
                    }

                    // softmax
                    if (currentNode.Leafs.Count > 0)
                    {
                        float max = stateSample.networkOutput.Sum();
                        for (int i = 0; i < stateSample.networkOutput.Length; i++)
                        {
                            stateSample.networkOutput[i] = stateSample.networkOutput[i] / max;
                        }
                    }
                    samples.Add(stateSample);
                }

                rotatedState = new ArrayGameBoardState(_state.GetBoardSize());
                for (int y = 0; y < _state.GetBoardSize(); y++)
                {
                    for (int x = 0; x < _state.GetBoardSize(); x++)
                    {
                        rotatedState.SetCellState(x, propagationState.GetBoardSize() - 1 - y, propagationState.GetCellState(x, y));
                    }
                }

                rotatedZobristHash = propagationState.GetBoardStateHash();
                if (!knownPositions.Contains(rotatedZobristHash))
                {
                    rotatedPos = new Constants.MovePosition(currentNode.MovePosition.X, propagationState.GetBoardSize() - 1 - currentNode.MovePosition.Y);
                    var stateSample = new TrainingSample(rotatedState, nextNode == null ? new MovePosition() : nextNode.MovePosition, rotatedPos, currentColor, nextNode == null ? -currentNode.WinProbability : nextNode.WinProbability);
                    foreach (var leaf in currentNode.Leafs)
                    {
                        //if (leaf != nextNode)
                        //{
                        stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), leaf.MovePosition.X, propagationState.GetBoardSize() - 1 - leaf.MovePosition.Y, (float)leaf.PlayoutsCount / (float)currentNode.PlayoutsCount);
                        //    }
                        //else
                        //{
                        //    var rewardValue = Math.Min(Math.Max(0.01f, reward + leaf.PolicyProbability), 1.0f);
                        //    stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), leaf.MovePosition.Y, leaf.MovePosition.X, rewardValue);
                        //}
                    }

                    // softmax
                    if (currentNode.Leafs.Count > 0)
                    {
                        float max = stateSample.networkOutput.Sum();
                        for (int i = 0; i < stateSample.networkOutput.Length; i++)
                        {
                            stateSample.networkOutput[i] = stateSample.networkOutput[i] / max;
                        }
                    }
                    samples.Add(stateSample);
                }

                propagationState.UndoSetCellState(currentNode.MovePosition.X, currentNode.MovePosition.Y);

                nextNode = currentNode;
                currentNode = currentNode.Parent;

                var nodesToInitialCounter = 0;
                var counterNode = currentNode;
                while (counterNode != null)
                {
                    counterNode = counterNode.Parent;
                    nodesToInitialCounter++;
                }

                if (nodesToInitialCounter < 2)
                {
                    break;
                }
            }
        }

        public MovePosition GetBestMove()
        {
            if (_node.Leafs.Count == 0)
            {
                return new MovePosition();
            }

            float probability = float.MinValue;
            MCTSTreeNode? bestNode = null;
            foreach (var node in _node.Leafs)
            {
                if (node.IsTerminal)
                {
                    bestNode = node;
                    probability = node.PlayoutsCount;
                    break;
                }

                if (node.PlayoutsCount == 0)
                {
                    continue;
                }

                if (node.PlayoutsCount > probability)
                {
                    bestNode = node;
                    probability = node.PlayoutsCount;
                }
            }

            if (bestNode != null)
            {
                return bestNode.MovePosition;
            }

            int rndChildIndex = _rndGenerator.Next(0, _node.Leafs.Count);
            return _node.Leafs[rndChildIndex].MovePosition;
        }

        public float GetMoveProbability(int x, int y)
        {
            if (_node.Leafs.Count == 0)
            {
                return 0.0f;
            }

            foreach (var leaf in _node.Leafs)
            {
                if (leaf.MovePosition.X == x && leaf.MovePosition.Y == y)
                {
                    return leaf.WinProbability;
                }
            }

            return 0.0f;
        }
    }
}
