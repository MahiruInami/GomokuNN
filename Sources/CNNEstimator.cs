using ImGuiNET;
using Keras.Layers;
using Numpy;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static GomokuNN.Sources.Constants;

namespace GomokuNN.Sources
{
    internal class CNNEstimator : IGameEstimator
    {

        ArrayGameBoardState _state = new ArrayGameBoardState(3);
        MCTSTreeNode _root, _node;
        IncrementalMovesPolicy _policy;
        float _mctsExplorationConst = 5.0f;

        bool _isTraining = false;
        int _estimatorColor = 0;
        Random _rndGenerator = new Random();

        Keras.Models.BaseModel _model;

        public CNNEstimator(bool isTraining)
        {
            _isTraining = isTraining;
        }

        public void InitFromState(IGameBoardState gameState, int turnColor)
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

            _policy = new IncrementalMovesPolicy(Constants.CNN_MOVE_POLICY_EXPANSION);
            _policy.Init(_state);

            _estimatorColor = turnColor;

            var availableMoves = _policy.GetHashedPositions();
            _root = new MCTSTreeNode(null)
            {
                MovePosition = new Constants.MovePosition(),
                MoveColor = Constants.NULL_COLOR
            };

            const int movesHistory = 1;
            int[] inputData = new int[_state.GetBoardSize() * _state.GetBoardSize() * movesHistory * 4];
            int boardOffset = _state.GetBoardSize() * _state.GetBoardSize();
            int dataOffset = 0;

            var currentNode = _root;
            for (int i = 0; i < _state.GetBoardSize() * _state.GetBoardSize(); i++)
            {
                var value = _state.GetRawCellState(i);
                inputData[i + dataOffset] = value == Constants.CROSS_COLOR ? 1 : 0;
                inputData[i + boardOffset + dataOffset] = value == Constants.ZERO_COLOR ? 1 : 0;

                inputData[i + boardOffset * 3 + dataOffset] = turnColor == Constants.CROSS_COLOR ? 1 : 0;
            }

            var input = np.array(inputData).reshape((1, 4, _state.GetBoardSize(), _state.GetBoardSize()));

            var result = _model.PredictMultipleOutputs(input, verbose: 0);
            var policyArray = result[0].GetData<float>();

            int index = 0;
            foreach (var availableMove in availableMoves)
            {
                int posX = _policy.GetUnhashedPositionX(availableMove);
                int posY = _policy.GetUnhashedPositionY(availableMove);

                _root.Leafs.Add(new MCTSTreeNode(_root)
                {
                    PolicyProbability = policyArray[posY * _state.GetBoardSize() + posX],
                    MovePosition = new Constants.MovePosition(posX, posY),
                    MoveColor = turnColor
                });

                index++;
            }

            _node = _root;
        }

        public void LoadModel(string policyModelPath)
        {
            _model = Keras.Models.Model.LoadModel(policyModelPath);
            //_valueModel = Keras.Models.Model.LoadModel(valueModelPath);
        }

        public void SaveModel(string policyModelPath)
        {
            //_policyModel.Save(policyModelPath);
            //_valueModel.Save(valueModelPath);
        }

        public bool SelectNextNode(int x, int y, int color, bool cleanUpTree)
        {
            if (_node == null)
            {
                return false;
            }

            if (_state.GetCellState(x, y) != Constants.EMPTY_COLOR)
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
                    MovePosition = new Constants.MovePosition(x, y),
                    MoveColor = color
                };
                _node.Leafs.Add(newNode);

                nextNode = newNode;
            }

            _node = nextNode;
            return true;
        }

        public void Estimate(int simulationsCount)
        {
            RunMTCSSimulation();
        }

        public void RunMTCSSimulation()
        {
            // Selection
            // Select tree node with best value until leaf node is reached
            var selectedNode = _node;
            ArrayGameBoardState state = new ArrayGameBoardState(_state.GetBoardSize());
            state.Copy(_state);
            while (!selectedNode.IsEndPoint)
            {
                MCTSTreeNode nextNode = selectedNode.Leafs[0];
                var bestSelectionValue = GetNodeSelectionValue(ref selectedNode, ref nextNode);
                for (int index = 1; index < selectedNode.Leafs.Count(); index++)
                {
                    var node = selectedNode.Leafs[index];
                    var selectionValue = GetNodeSelectionValue(ref selectedNode, ref node);
                    if (selectionValue > bestSelectionValue)
                    {
                        nextNode = node;
                        bestSelectionValue = selectionValue;
                    }
                }

                selectedNode = nextNode;
                state.SetCellState(selectedNode.MovePosition.X, selectedNode.MovePosition.Y, selectedNode.MoveColor);
            }

            var movesPolicy = new IncrementalMovesPolicy(Constants.CNN_MOVE_POLICY_EXPANSION);
            movesPolicy.Init(state);

            var availableMovesAtSelectedNode = movesPolicy.GetHashedPositions();

            float score;
            IGameBoardState istate = state;
            var gameResultAfterSelection = GameResultEstimator.EstimateResult(ref istate, selectedNode.MovePosition.X, selectedNode.MovePosition.Y);
            if (gameResultAfterSelection != Constants.GameResult.IN_PROGRESS)
            {
                // terminal node
                score = ConvertGameResultToScore(gameResultAfterSelection);
                if (selectedNode.MoveColor != _estimatorColor)
                {
                    score = -score;
                }

                availableMovesAtSelectedNode.Clear();

                // Backpropagate
                selectedNode.IsGameEndPosition = (gameResultAfterSelection == Constants.GameResult.WIN);
                var propagationNode = selectedNode;
                while (propagationNode != null)
                {
                    propagationNode.PlayoutsCount++;
                    propagationNode.PlayoutScore += (propagationNode.MoveColor != _estimatorColor ? -score : score);
                    propagationNode.WinProbability = (float)propagationNode.PlayoutScore / (float)propagationNode.PlayoutsCount;

                    propagationNode = propagationNode.Parent;
                }
            }
            else
            {
                int[] inputData = new int[state.GetBoardSize() * state.GetBoardSize() * 4];
                int boardOffset = state.GetBoardSize() * state.GetBoardSize();
                int dataOffset = 0;

                var currentNode = selectedNode;
                for (int i = 0; i < state.GetBoardSize() * state.GetBoardSize(); i++)
                {
                    var value = state.GetRawCellState(i);
                    inputData[i + dataOffset] = value == Constants.CROSS_COLOR ? 1 : 0;
                    inputData[i + boardOffset + dataOffset] = value == Constants.ZERO_COLOR ? 1 : 0;

                    inputData[i + boardOffset * 3 + dataOffset] = Constants.RotateColor(selectedNode.MoveColor) == Constants.CROSS_COLOR ? 1 : 0;
                }

                inputData[boardOffset * 2 + state.GetPositionHash(selectedNode.MovePosition.X, selectedNode.MovePosition.Y)] = 1;

                var input = np.array(inputData).reshape((1, 4, _state.GetBoardSize(), _state.GetBoardSize()));

                var result = _model.PredictMultipleOutputs(input, verbose: 0);
                var policyArray = result[0].GetData<float>();
                var resultArray = result[1].GetData<float>();
                score = resultArray[0];

                // Backpropagate
                var propagationNode = selectedNode;
                while (propagationNode != null)
                {
                    propagationNode.PlayoutsCount++;
                    propagationNode.PlayoutScore += (propagationNode.MoveColor != _estimatorColor ? -score : score);
                    propagationNode.WinProbability = (float)propagationNode.PlayoutScore / (float)propagationNode.PlayoutsCount;

                    propagationNode = propagationNode.Parent;
                }

                if (selectedNode.PlayoutsCount > 1 && selectedNode.IsEndPoint && availableMovesAtSelectedNode.Count > 0)
                {
                    // expand node
                    int index = 0;
                    int turnColor = Constants.RotateColor(selectedNode.MoveColor);
                    //for (int y = 0; y < _state.GetBoardSize(); y++)
                    //{
                    //    for (int x = 0; x < _state.GetBoardSize(); x++)
                    //    {

                    //    }
                    //}

                    foreach (var availableMove in availableMovesAtSelectedNode)
                    {
                        int posX = _policy.GetUnhashedPositionX(availableMove);
                        int posY = _policy.GetUnhashedPositionY(availableMove);

                        selectedNode.Leafs.Add(new MCTSTreeNode(selectedNode)
                        {
                            PolicyProbability = policyArray[posY * state.GetBoardSize() + posX],
                            MovePosition = new Constants.MovePosition(posX, posY),
                            MoveColor = turnColor
                        });

                        ArrayGameBoardState newNodeState = new ArrayGameBoardState(_state.GetBoardSize());
                        state.Copy(state);
                        state.SetCellState(posX, posY, turnColor);

                        IGameBoardState newNodeIState = newNodeState;
                        var gameResult = GameResultEstimator.EstimateResult(ref newNodeIState, posX, posY);
                        if (gameResult == Constants.GameResult.WIN)
                        {
                            selectedNode.IsGameEndPosition = true;
                        }

                        index++;
                    }
                }
            }
        }

        public float ConvertGameResultToScore(Constants.GameResult result)
        {
            return result == Constants.GameResult.WIN ? 1.0f : result == Constants.GameResult.LOSE ? -1.0f : 0.0f;
        }

        public double GetNodeSelectionValue(ref MCTSTreeNode parent, ref MCTSTreeNode child)
        {
            if (child.PlayoutsCount == 0)
            {
                return _mctsExplorationConst * child.PolicyProbability * Math.Sqrt((float)parent.PlayoutsCount) / (1.0f + child.PlayoutsCount);
            }

            return child.WinProbability + _mctsExplorationConst * child.PolicyProbability * Math.Sqrt((float)parent.PlayoutsCount) / (1.0f + child.PlayoutsCount);
        }

        public void SetExplorationConst(float value)
        {
            _mctsExplorationConst = value;
        }

        public void DebugMenuDraw(ref GameBoard gameBoard)
        {
            if (_node == null)
            {
                return;
            }

            ImGui.Text("Playouts count: " + _node.PlayoutsCount.ToString());
            ImGui.Text("Node win probability: " + _node.WinProbability.ToString());

            if (ImGui.Button("Move back"))
            {
                if (_node != null && _node.Parent != null)
                {
                    _state.SetCellState(_node.MovePosition.X, _node.MovePosition.Y, Constants.EMPTY_COLOR);
                    gameBoard.GetBoardState().SetCellState(_node.MovePosition.X, _node.MovePosition.Y, Constants.EMPTY_COLOR);
                    gameBoard.SetCurrentTurnColor(_node.MoveColor);

                    _policy.Init(_state);

                    _node = _node.Parent;
                }
            }
        }

        public void DebugFieldDraw()
        {

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

            FillTrainingSamples(ref samples, winnerColor);

            return samples;
        }

        public void FillTrainingSamples(ref List<TrainingSample> samples, int winnerColor)
        {
            ArrayGameBoardState propagationState = new ArrayGameBoardState(_state.GetBoardSize());
            propagationState.Copy(_state);

            IGameBoardState state = _state;

            var currentNode = _node;
            MCTSTreeNode? nextNode = null;
            while (currentNode != null && currentNode.Parent != null)
            {
                var currentColor = Constants.RotateColor(currentNode.MoveColor);
                var stateSample = new TrainingSample(state, nextNode == null ? new MovePosition() : nextNode.MovePosition, currentNode.MovePosition, Constants.RotateColor(currentNode.MoveColor), currentColor == winnerColor ? 1.0f : -1.0f);
                foreach (var leaf in currentNode.Leafs)
                {
                    if (leaf != nextNode)
                    {
                        stateSample.SetPolicyOutputForMove(_state.GetBoardSize(), leaf.MovePosition.X, leaf.MovePosition.Y, leaf.PolicyProbability);
                    }
                }

                if (currentNode.MovePosition.X < 0 || currentNode.MovePosition.Y < 0)
                {
                    break;
                }

                propagationState.SetCellState(currentNode.MovePosition.X, currentNode.MovePosition.Y, Constants.EMPTY_COLOR);

                nextNode = currentNode;
                currentNode = currentNode.Parent;

                samples.Add(stateSample);
            }
        }

        public Constants.MovePosition GetBestMove()
        {
            if (_node.Leafs.Count == 0)
            {
                return new Constants.MovePosition();
            }

            if (_isTraining)
            {
                float rndValue = _rndGenerator.NextSingle();
                if (rndValue < 0.15f)
                {
                    int rndChildIndex1 = _rndGenerator.Next(0, _node.Leafs.Count);
                    return _node.Leafs[rndChildIndex1].MovePosition;
                }
            }

            float probability = float.MinValue;
            MCTSTreeNode? bestNode = null;
            foreach (var node in _node.Leafs)
            {
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
