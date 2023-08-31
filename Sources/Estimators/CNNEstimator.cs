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
        bool _debugAutoEstimation = false;

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
                NodeDepth = 0,
                MovePosition = new MovePosition(),
                MoveColor = NULL_COLOR
            };

            float[] inputData = new float[_state.GetBoardSize() * _state.GetBoardSize() * 4];
            int boardOffset = _state.GetBoardSize() * _state.GetBoardSize();
            int dataOffset = 0;

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
                    NodeDepth = 1,
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
                    NodeDepth = _node.NodeDepth + 1,
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
                    Array.Fill(alpha, 0.08);
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

                for (int i = 0; i < state.GetBoardSize() * state.GetBoardSize(); i++)
                {
                    var value = state.GetRawCellState(i);
                    inputData[i] = value == CROSS_COLOR ? 1 : 0;
                    inputData[i + boardOffset] = value == ZERO_COLOR ? 1 : 0;

                    inputData[i + boardOffset * 3] = Constants.RotateColor(selectedNode.MoveColor) == CROSS_COLOR ? 1 : 0;
                }

                inputData[boardOffset * 2 + state.GetPositionHash(selectedNode.MovePosition.X, selectedNode.MovePosition.Y)] = 1;

                var inputTensor = new DenseTensor<float>(inputData, new int[] { 1, 4, _state.GetBoardSize(), _state.GetBoardSize() });
                var modelInput = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("input_layer", inputTensor) };

                var output = _inferenceSession.Run(modelInput).ToArray();
                var policyOutput = output[0].AsTensor<float>();
                var valueOutput = output[1].AsTensor<float>();

                score = -valueOutput[0];
                selectedNode.InitialWinProbability = score;

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
                            NodeDepth = selectedNode.NodeDepth + 1,
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

                            //inputData = new float[newNodeState.GetBoardSize() * newNodeState.GetBoardSize() * 4];
                            //for (int i = 0; i < newNodeState.GetBoardSize() * newNodeState.GetBoardSize(); i++)
                            //{
                            //    var value = newNodeState.GetRawCellState(i);
                            //    inputData[i] = value == CROSS_COLOR ? 1 : 0;
                            //    inputData[i + boardOffset] = value == ZERO_COLOR ? 1 : 0;

                            //    inputData[i + boardOffset * 3] = RotateColor(turnColor) == CROSS_COLOR ? 1 : 0;
                            //}

                            //inputData[boardOffset * 2 + state.GetPositionHash(newLeaf.MovePosition.X, newLeaf.MovePosition.Y)] = 1;

                            //inputTensor = new DenseTensor<float>(inputData, new int[] { 1, 4, _state.GetBoardSize(), _state.GetBoardSize() });
                            //modelInput = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("input_layer", inputTensor) };

                            //output = _inferenceSession.Run(modelInput).ToArray();
                            //policyOutput = output[0].AsTensor<float>();
                            //valueOutput = output[1].AsTensor<float>();

                            //score = valueOutput[0];
                            //newLeaf.InitialWinProbability = -score;
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
            return child.WinProbability + _mctsExplorationConst * child.PolicyProbability * Math.Sqrt(parent.PlayoutsCount) / (1.0f + child.PlayoutsCount);
        }

        public double GetNodeSelectionValue(ref MCTSTreeNode parent, ref MCTSTreeNode child, int nodeIndex, double[] dirichletNoise)
        {
            double epsilon = (_node.NodeDepth - parent.NodeDepth <= 1) ? 0.25 : 0.0;
            var policyProbability = (1 - epsilon) * child.PolicyProbability + epsilon * dirichletNoise[nodeIndex];

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
                        renderHelper.DrawScoreText(leaf.InitialWinProbability.ToString("F2"), x, y);
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
                ImGui.Text("Current state win probability: " + _node.InitialWinProbability.ToString());
            }

            ImGui.Checkbox("Show best move", ref _debugDrawBestMove);
            ImGui.Checkbox("Show probabilities", ref _debugDrawProbabilities);
            ImGui.Checkbox("Auto estimate", ref _debugAutoEstimation);

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

            if (ImGui.Button("Estimate once") || _debugAutoEstimation)
            {
                EstimateOnce();
            }

            if (ImGui.Button("Estimate current state"))
            {
                float[] inputData = new float[_state.GetBoardSize() * _state.GetBoardSize() * 4];
                int boardOffset = _state.GetBoardSize() * _state.GetBoardSize();
                int dataOffset = 0;

                for (int i = 0; i < _state.GetBoardSize() * _state.GetBoardSize(); i++)
                {
                    var value = _state.GetRawCellState(i);
                    inputData[i + dataOffset] = value == CROSS_COLOR ? 1 : 0;
                    inputData[i + boardOffset + dataOffset] = value == ZERO_COLOR ? 1 : 0;

                    inputData[i + boardOffset * 3 + dataOffset] = Constants.RotateColor(_node.MoveColor) == CROSS_COLOR ? 1 : 0;
                }
                inputData[boardOffset * 2 + _state.GetPositionHash(_node.MovePosition.X, _node.MovePosition.Y)] = 1;


                var inputTensor = new DenseTensor<float>(inputData, new int[] { 1, 4, _state.GetBoardSize(), _state.GetBoardSize() });
                var modelInput = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("input_layer", inputTensor) };

                var output = _inferenceSession.Run(modelInput).ToArray();
                var policyOutput = output[0].AsTensor<float>();
                var valueOutput = output[1].AsTensor<float>();

                Console.WriteLine("Test: " + valueOutput[0]);
            }

            if (ImGui.Button("Sample state"))
            {
                var samples1 = GetTrainingSamples(_node.MoveColor);
                var samples2 = GetTrainingSamples(Constants.RotateColor(_node.MoveColor));
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

        public List<TrainingSample> GetTrainingSamples(int winnerColor)
        {
            List<TrainingSample> samples = new List<TrainingSample>();
            HashSet<long> knownPositions = new HashSet<long>();

            FillTrainingSamples(ref samples, winnerColor, ref knownPositions);

            return samples;
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

            int hashHitCount = 0;

            var currentNode = _node;
            MCTSTreeNode? nextNode = null;
            while (currentNode != null && currentNode.Parent != null)
            {
                if (currentNode.MovePosition.X < 0 || currentNode.MovePosition.Y < 0)
                {
                    break;
                }

                if (nextNode == null)
                {
                    propagationState.UndoSetCellState(currentNode.MovePosition.X, currentNode.MovePosition.Y);

                    nextNode = currentNode;
                    currentNode = currentNode.Parent;

                    if (currentNode.NodeDepth < 2)
                    {
                        break;
                    }

                    continue;
                }

                var currentColor = RotateColor(currentNode.MoveColor);
                var reward = winnerColor == 0 ? 0.0f : currentColor == winnerColor ? 1.0f : -1.0f;
                var winRate = reward;
                //var winRate = (nextNode == null ? -currentNode.WinProbability : nextNode.WinProbability) + reward;
                //winRate = Math.Clamp(winRate, -1.0f, 1.0f);
                if (winnerColor == 0)
                {
                    winRate = 0.0f;
                }

                long zobristHash = propagationState.GetBoardStateHash();
                if (!knownPositions.Contains(zobristHash))
                {
                    var stateSample = new TrainingSample(propagationState, nextNode == null ? new MovePosition() : nextNode.MovePosition, currentNode.MovePosition, currentColor, winRate);
                    for (int i = 0; i < stateSample.networkOutput.Length; i++)
                    {
                        stateSample.networkOutput[i] = 0;
                    }

                    foreach (var leaf in currentNode.Leafs)
                    {
                        if (leaf.IsTerminal)
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), leaf.MovePosition.X, leaf.MovePosition.Y, 1.0f);
                        }
                        else if (leaf.PlayoutsCount > 0)
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), leaf.MovePosition.X, leaf.MovePosition.Y, (float)leaf.PlayoutsCount / (float)currentNode.PlayoutsCount);
                        }
                        else
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), leaf.MovePosition.X, leaf.MovePosition.Y, (float)leaf.PolicyProbability);
                        }
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

                    stateSample.positionHash = zobristHash;
                    samples.Add(stateSample);
                    knownPositions.Add(zobristHash);
                } else { hashHitCount++; }

                var rotatedPos = new Constants.MovePosition();
                ArrayGameBoardState rotatedState = new ArrayGameBoardState(_state.GetBoardSize());
                for (int y = 0; y < _state.GetBoardSize(); y++)
                {
                    for (int x = 0; x < _state.GetBoardSize(); x++)
                    {
                        rotatedState.SetCellState(y, propagationState.GetBoardSize() - 1 - x, propagationState.GetCellState(x, y));
                    }
                }

                long rotatedZobristHash = rotatedState.GetBoardStateHash();
                if (!knownPositions.Contains(rotatedZobristHash))
                {
                    rotatedPos = new Constants.MovePosition(currentNode.MovePosition.Y, rotatedState.GetBoardSize() - 1 - currentNode.MovePosition.X);
                    var stateSample = new TrainingSample(rotatedState, nextNode == null ? new MovePosition() : new MovePosition(nextNode.MovePosition.Y, rotatedState.GetBoardSize() - 1 - nextNode.MovePosition.X), rotatedPos, currentColor, winRate);
                    for (int i = 0; i < stateSample.networkOutput.Length; i++)
                    {
                        stateSample.networkOutput[i] = 0;
                    }

                    foreach (var leaf in currentNode.Leafs)
                    {
                        if (leaf.IsTerminal)
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), leaf.MovePosition.Y, rotatedState.GetBoardSize() - 1 - leaf.MovePosition.X, 1.0f);
                        }
                        else if (leaf.PlayoutsCount > 0)
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), leaf.MovePosition.Y, rotatedState.GetBoardSize() - 1 - leaf.MovePosition.X, (float)leaf.PlayoutsCount / (float)currentNode.PlayoutsCount);
                        }
                        else
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), leaf.MovePosition.Y, rotatedState.GetBoardSize() - 1 - leaf.MovePosition.X, (float)leaf.PolicyProbability);
                        }
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

                    stateSample.positionHash = rotatedZobristHash;
                    samples.Add(stateSample);
                    knownPositions.Add(rotatedZobristHash);
                } else { hashHitCount++; }


                rotatedState = new ArrayGameBoardState(_state.GetBoardSize());
                for (int y = 0; y < _state.GetBoardSize(); y++)
                {
                    for (int x = 0; x < _state.GetBoardSize(); x++)
                    {
                        rotatedState.SetCellState(propagationState.GetBoardSize() - 1 - x, propagationState.GetBoardSize() - 1 - y, propagationState.GetCellState(x, y));
                    }
                }

                rotatedZobristHash = rotatedState.GetBoardStateHash();
                if (!knownPositions.Contains(rotatedZobristHash))
                {
                    rotatedPos = new Constants.MovePosition(rotatedState.GetBoardSize() - 1 - currentNode.MovePosition.X, rotatedState.GetBoardSize() - 1 - currentNode.MovePosition.Y);
                    var stateSample = new TrainingSample(rotatedState, nextNode == null ? new MovePosition() : new MovePosition(rotatedState.GetBoardSize() - 1 - nextNode.MovePosition.X, rotatedState.GetBoardSize() - 1 - nextNode.MovePosition.Y), rotatedPos, currentColor, winRate);
                    for (int i = 0; i < stateSample.networkOutput.Length; i++)
                    {
                        stateSample.networkOutput[i] = 0;
                    }

                    foreach (var leaf in currentNode.Leafs)
                    {
                        if (leaf.IsTerminal)
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), rotatedState.GetBoardSize() - 1 - leaf.MovePosition.X, rotatedState.GetBoardSize() - 1 - leaf.MovePosition.Y, 1.0f);
                        }
                        else if (leaf.PlayoutsCount > 0)
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), rotatedState.GetBoardSize() - 1 - leaf.MovePosition.X, rotatedState.GetBoardSize() - 1 - leaf.MovePosition.Y, (float)leaf.PlayoutsCount / (float)currentNode.PlayoutsCount);
                        }
                        else
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), rotatedState.GetBoardSize() - 1 - leaf.MovePosition.X, rotatedState.GetBoardSize() - 1 - leaf.MovePosition.Y, (float)leaf.PolicyProbability);
                        }
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

                    stateSample.positionHash = rotatedZobristHash;
                    samples.Add(stateSample);
                    knownPositions.Add(rotatedZobristHash);
                }
                else { hashHitCount++; }

                //
                rotatedState = new ArrayGameBoardState(_state.GetBoardSize());
                for (int y = 0; y < _state.GetBoardSize(); y++)
                {
                    for (int x = 0; x < _state.GetBoardSize(); x++)
                    {
                        rotatedState.SetCellState(propagationState.GetBoardSize() - 1 - y, x, propagationState.GetCellState(x, y));
                    }
                }

                rotatedZobristHash = rotatedState.GetBoardStateHash();
                if (!knownPositions.Contains(rotatedZobristHash))
                {
                    rotatedPos = new Constants.MovePosition(rotatedState.GetBoardSize() - 1 - currentNode.MovePosition.Y, currentNode.MovePosition.X);
                    var stateSample = new TrainingSample(rotatedState, nextNode == null ? new MovePosition() : new MovePosition(rotatedState.GetBoardSize() - 1 - nextNode.MovePosition.Y, nextNode.MovePosition.X), rotatedPos, currentColor, winRate);
                    for (int i = 0; i < stateSample.networkOutput.Length; i++)
                    {
                        stateSample.networkOutput[i] = 0;
                    }

                    foreach (var leaf in currentNode.Leafs)
                    {
                        if (leaf.IsTerminal)
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), rotatedState.GetBoardSize() - 1 - leaf.MovePosition.Y, leaf.MovePosition.X, 1.0f);
                        }
                        else if (leaf.PlayoutsCount > 0)
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), rotatedState.GetBoardSize() - 1 - leaf.MovePosition.Y, leaf.MovePosition.X, (float)leaf.PlayoutsCount / (float)currentNode.PlayoutsCount);
                        }
                        else
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), rotatedState.GetBoardSize() - 1 - leaf.MovePosition.Y, leaf.MovePosition.X, (float)leaf.PolicyProbability);
                        }
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

                    stateSample.positionHash = rotatedZobristHash;
                    samples.Add(stateSample);
                    knownPositions.Add(rotatedZobristHash);
                }
                else { hashHitCount++; }

                //
                rotatedState = new ArrayGameBoardState(_state.GetBoardSize());
                for (int y = 0; y < _state.GetBoardSize(); y++)
                {
                    for (int x = 0; x < _state.GetBoardSize(); x++)
                    {
                        rotatedState.SetCellState(propagationState.GetBoardSize() - 1 - x, y, propagationState.GetCellState(x, y));
                    }
                }

                rotatedZobristHash = rotatedState.GetBoardStateHash();
                if (!knownPositions.Contains(rotatedZobristHash))
                {
                    rotatedPos = new Constants.MovePosition(rotatedState.GetBoardSize() - 1 - currentNode.MovePosition.X, currentNode.MovePosition.Y);
                    var stateSample = new TrainingSample(rotatedState, nextNode == null ? new MovePosition() : new MovePosition(rotatedState.GetBoardSize() - 1 - nextNode.MovePosition.X, nextNode.MovePosition.Y), rotatedPos, currentColor, winRate);
                    for (int i = 0; i < stateSample.networkOutput.Length; i++)
                    {
                        stateSample.networkOutput[i] = 0;
                    }

                    foreach (var leaf in currentNode.Leafs)
                    {
                        if (leaf.IsTerminal)
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), rotatedState.GetBoardSize() - 1 - leaf.MovePosition.X, leaf.MovePosition.Y, 1.0f);
                        }
                        else if (leaf.PlayoutsCount > 0)
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), rotatedState.GetBoardSize() - 1 - leaf.MovePosition.X, leaf.MovePosition.Y, (float)leaf.PlayoutsCount / (float)currentNode.PlayoutsCount);
                        }
                        else
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), rotatedState.GetBoardSize() - 1 - leaf.MovePosition.X, leaf.MovePosition.Y, (float)leaf.PolicyProbability);
                        }
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

                    stateSample.positionHash = rotatedZobristHash;
                    samples.Add(stateSample);
                    knownPositions.Add(rotatedZobristHash);
                }
                else { hashHitCount++; }

                rotatedState = new ArrayGameBoardState(_state.GetBoardSize());
                for (int y = 0; y < _state.GetBoardSize(); y++)
                {
                    for (int x = 0; x < _state.GetBoardSize(); x++)
                    {
                        rotatedState.SetCellState(y, x, propagationState.GetCellState(x, y));
                    }
                }

                rotatedZobristHash = rotatedState.GetBoardStateHash();
                if (!knownPositions.Contains(rotatedZobristHash))
                {
                    rotatedPos = new Constants.MovePosition(currentNode.MovePosition.Y, currentNode.MovePosition.X);
                    var stateSample = new TrainingSample(rotatedState, nextNode == null ? new MovePosition() : new MovePosition(nextNode.MovePosition.Y, nextNode.MovePosition.X), rotatedPos, currentColor, winRate);
                    for (int i = 0; i < stateSample.networkOutput.Length; i++)
                    {
                        stateSample.networkOutput[i] = 0;
                    }

                    foreach (var leaf in currentNode.Leafs)
                    {
                        if (leaf.IsTerminal)
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), leaf.MovePosition.Y, leaf.MovePosition.X, 1.0f);
                        }
                        else if (leaf.PlayoutsCount > 0)
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), leaf.MovePosition.Y, leaf.MovePosition.X, (float)leaf.PlayoutsCount / (float)currentNode.PlayoutsCount);
                        }
                        else
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), leaf.MovePosition.Y, leaf.MovePosition.X, (float)leaf.PolicyProbability);
                        }
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

                    stateSample.positionHash = rotatedZobristHash;
                    samples.Add(stateSample);
                    knownPositions.Add(rotatedZobristHash);
                }
                else { hashHitCount++; }

                rotatedState = new ArrayGameBoardState(_state.GetBoardSize());
                for (int y = 0; y < _state.GetBoardSize(); y++)
                {
                    for (int x = 0; x < _state.GetBoardSize(); x++)
                    {
                        rotatedState.SetCellState(propagationState.GetBoardSize() - 1 - y, propagationState.GetBoardSize() - 1 - x, propagationState.GetCellState(x, y));
                    }
                }

                rotatedZobristHash = rotatedState.GetBoardStateHash();
                if (!knownPositions.Contains(rotatedZobristHash))
                {
                    rotatedPos = new Constants.MovePosition(rotatedState.GetBoardSize() - 1 - currentNode.MovePosition.Y, rotatedState.GetBoardSize() - 1 - currentNode.MovePosition.X);
                    var stateSample = new TrainingSample(rotatedState, nextNode == null ? new MovePosition() : new Constants.MovePosition(rotatedState.GetBoardSize() - 1 - nextNode.MovePosition.Y, propagationState.GetBoardSize() - 1 - nextNode.MovePosition.X), rotatedPos, currentColor, winRate);
                    for (int i = 0; i < stateSample.networkOutput.Length; i++)
                    {
                        stateSample.networkOutput[i] = 0;
                    }

                    foreach (var leaf in currentNode.Leafs)
                    {
                        if (leaf.IsTerminal)
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), rotatedState.GetBoardSize() - 1 - leaf.MovePosition.Y, rotatedState.GetBoardSize() - 1 - leaf.MovePosition.X, 1.0f);
                        }
                        else if (leaf.PlayoutsCount > 0)
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), rotatedState.GetBoardSize() - 1 - leaf.MovePosition.Y, rotatedState.GetBoardSize() - 1 - leaf.MovePosition.X, (float)leaf.PlayoutsCount / (float)currentNode.PlayoutsCount);
                        }
                        else
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), rotatedState.GetBoardSize() - 1 - leaf.MovePosition.Y, rotatedState.GetBoardSize() - 1 - leaf.MovePosition.X, (float)leaf.PolicyProbability);
                        }
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

                    stateSample.positionHash = rotatedZobristHash;
                    samples.Add(stateSample);
                    knownPositions.Add(rotatedZobristHash);
                }
                else { hashHitCount++; }

                rotatedState = new ArrayGameBoardState(_state.GetBoardSize());
                for (int y = 0; y < _state.GetBoardSize(); y++)
                {
                    for (int x = 0; x < _state.GetBoardSize(); x++)
                    {
                        rotatedState.SetCellState(x, propagationState.GetBoardSize() - 1 - y, propagationState.GetCellState(x, y));
                    }
                }

                rotatedZobristHash = rotatedState.GetBoardStateHash();
                if (!knownPositions.Contains(rotatedZobristHash))
                {
                    rotatedPos = new Constants.MovePosition(currentNode.MovePosition.X, rotatedState.GetBoardSize() - 1 - currentNode.MovePosition.Y);
                    var stateSample = new TrainingSample(rotatedState, nextNode == null ? new MovePosition() : new Constants.MovePosition(nextNode.MovePosition.X, rotatedState.GetBoardSize() - 1 - nextNode.MovePosition.Y), rotatedPos, currentColor, winRate);
                    for (int i = 0; i < stateSample.networkOutput.Length; i++)
                    {
                        stateSample.networkOutput[i] = 0;
                    }

                    foreach (var leaf in currentNode.Leafs)
                    {
                        if (leaf.IsTerminal)
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), leaf.MovePosition.X, rotatedState.GetBoardSize() - 1 - leaf.MovePosition.Y, 1.0f);
                        }
                        else if (leaf.PlayoutsCount > 0)
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), leaf.MovePosition.X, rotatedState.GetBoardSize() - 1 - leaf.MovePosition.Y, (float)leaf.PlayoutsCount / (float)currentNode.PlayoutsCount);
                        }
                        else
                        {
                            stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), leaf.MovePosition.X, rotatedState.GetBoardSize() - 1 - leaf.MovePosition.Y, (float)leaf.PolicyProbability);
                        }
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

                    stateSample.positionHash = rotatedZobristHash;
                    samples.Add(stateSample);
                    knownPositions.Add(rotatedZobristHash);
                }
                else { hashHitCount++; }

                propagationState.UndoSetCellState(currentNode.MovePosition.X, currentNode.MovePosition.Y);

                nextNode = currentNode;
                currentNode = currentNode.Parent;

                if (currentNode.NodeDepth < 2)
                {
                    break;
                }
            }

            //Console.WriteLine("Hash hits: " + hashHitCount + " " + _node.NodeDepth * 8);
        }

        public MovePosition GetBestMove()
        {
            if (_node.Leafs.Count == 0)
            {
                return new MovePosition();
            }

            int mostPlayouts = -1;
            float bestProbability = float.MinValue;
            MCTSTreeNode? bestNode = null;
            foreach (var node in _node.Leafs)
            {
                if (node.IsTerminal)
                {
                    bestNode = node;
                    mostPlayouts = node.PlayoutsCount;
                    bestProbability = node.WinProbability;
                    break;
                }

                if (node.PlayoutsCount == 0)
                {
                    continue;
                }

                if (node.PlayoutsCount > mostPlayouts)
                {
                    bestNode = node;
                    mostPlayouts = node.PlayoutsCount;
                    bestProbability = node.WinProbability;
                }

                if (node.PlayoutsCount == mostPlayouts && node.WinProbability > bestProbability)
                {
                    bestNode = node;
                    mostPlayouts = node.PlayoutsCount;
                    bestProbability = node.WinProbability;
                }
            }

            if (bestNode != null)
            {
                return bestNode.MovePosition;
            }

            Console.WriteLine("Can't find best move. Playing random");
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
