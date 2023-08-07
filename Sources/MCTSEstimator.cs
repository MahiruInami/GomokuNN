using ImGuiNET;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace GomokuNN.Sources
{
    internal class MCTSEstimator : IGameEstimator
    {

        ArrayGameBoardState _state = new ArrayGameBoardState(3);
        MCTSTreeNode _root, _node;
        IncrementalMovesPolicy _policy;
        float _mctsExplorationConst = 1.0f;

        int _estimatorColor = 0;
        bool _isMultithreading = true;

        Random _rndGenerator = new Random();

        HashSet<int> _zobristHash = new HashSet<int>();

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

            _policy = new IncrementalMovesPolicy();
            _policy.Init(_state);

            _estimatorColor = turnColor;

            _zobristHash.Clear();

            var availableMoves = _policy.GetHashedPositions();
            _root = new MCTSTreeNode(null, availableMoves.Count)
            {
                MovePosition = new Constants.MovePosition(),
                MoveColor = Constants.NULL_COLOR
            };

            int index = 0;
            foreach (var availableMove in availableMoves)
            {
                int posX = _policy.GetUnhashedPositionX(availableMove);
                int posY = _policy.GetUnhashedPositionY(availableMove);

                _root.Leafs[index] = new MCTSTreeNode(_root)
                {
                    MovePosition = new Constants.MovePosition(posX, posY),
                    MoveColor = turnColor
                };

                index++;
            }

            _node = _root;
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

            if (_node.Leafs == null)
            {
                // create nodes
                _node.Leafs = new MCTSTreeNode[1];
                _node.Leafs[0] = new MCTSTreeNode(_node)
                {
                    MovePosition = new Constants.MovePosition(x, y),
                    MoveColor = color
                };

                _node = _node.Leafs[0];
                return true;
            }

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
                _node.Leafs = new MCTSTreeNode[1];
                _node.Leafs[0] = new MCTSTreeNode(_node)
                {
                    MovePosition = new Constants.MovePosition(x, y),
                    MoveColor = color
                };

                nextNode = _node.Leafs[0];
            }

            _node = nextNode;
            return true;
        }

        public void Estimate(int simulationsCount)
        {
            for (int i = 0; i < simulationsCount; i++)
            {
                RunMTCSSimulation();
            }
        }

        public void RunMTCSSimulation()
        {
            // Selection
            // Select tree node with best value until leaf node is reached
            var selectedNode = _node;
            ArrayGameBoardState copyState = new ArrayGameBoardState(_state.GetBoardSize());
            copyState.Copy(_state);
            IGameBoardState state = copyState;
            while (!selectedNode.IsEndPoint)
            {
                if (selectedNode.Leafs == null)
                {
                    break;
                }

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

            var movesPolicy = new IncrementalMovesPolicy();
            movesPolicy.Init(state);

            var availableMovesAtSelectedNode = movesPolicy.GetHashedPositions();

            float score;
            var gameResultAfterSelection = GameResultEstimator.EstimateResult(ref state, selectedNode.MovePosition.X, selectedNode.MovePosition.Y);
            if (gameResultAfterSelection != Constants.GameResult.IN_PROGRESS)
            {
                // terminal node
                score = ConvertGameResultToScore(gameResultAfterSelection);
                if (selectedNode.MoveColor != _estimatorColor)
                {
                    score = -score;
                }

                availableMovesAtSelectedNode.Clear();
            }
            else
            {
                // Explore node with random play simulation
                if (_isMultithreading)
                {
                    var tasks = new Task<float>[12];
                    for (int i = 0; i < tasks.Length; i++)
                    {
                        var taskState = new ArrayGameBoardState(state.GetBoardSize());
                        taskState.Copy(copyState);

                        IGameBoardState iTaskState = taskState;

                        var taskMovePolicy = new IncrementalMovesPolicy();
                        taskMovePolicy.Init(taskState);

                        tasks[i] = Task.Factory.StartNew(() =>
                        {
                            var result = PlayRandomSimulation(ref iTaskState, ref taskMovePolicy, Constants.RotateColor(selectedNode.MoveColor));
                            var taskScore = ConvertGameResultToScore(result);
                            return taskScore;
                        });
                    }
                    Task.WaitAll(tasks);

                    score = 0;
                    for (int i = 0; i < tasks.Length; i++)
                    {
                        if (Constants.RotateColor(selectedNode.MoveColor) != _estimatorColor)
                        {
                            score -= tasks[i].Result;
                        }
                        else
                        {
                            score += tasks[i].Result;
                        }
                    }
                    score /= 12.0f;
                }
                else
                {
                    var taskState = new ArrayGameBoardState(state.GetBoardSize());
                    taskState.Copy(copyState);

                    IGameBoardState iTaskState = taskState;

                    var taskMovePolicy = new IncrementalMovesPolicy();
                    taskMovePolicy.Init(taskState);

                    var result = PlayRandomSimulation(ref iTaskState, ref taskMovePolicy, Constants.RotateColor(selectedNode.MoveColor));
                    score = ConvertGameResultToScore(result);
                    if (Constants.RotateColor(selectedNode.MoveColor) != _estimatorColor)
                    {
                        score = -score;
                    }
                }
            }

            // Backpropagate
            var propagationNode = selectedNode;
            while (propagationNode != null)
            {
                propagationNode.PlayoutsCount++;
                propagationNode.PlayoutScore += (propagationNode.MoveColor != _estimatorColor ? -score : score);
                propagationNode.WinProbability = (float)propagationNode.PlayoutScore / (float)propagationNode.PlayoutsCount;

                propagationNode = propagationNode.Parent;
            }

            if (selectedNode.PlayoutsCount > Constants.PLAYOUTS_TO_EXPANSION && selectedNode.IsEndPoint && availableMovesAtSelectedNode.Count > 0)
            {
                // expand node
                int index = 0;
                int turnColor = Constants.RotateColor(selectedNode.MoveColor);
                
                //List<int> validMoves = new List<int>();
                //foreach (var availableMove in availableMovesAtSelectedNode)
                //{
                //    int posX = _policy.GetUnhashedPositionX(availableMove);
                //    int posY = _policy.GetUnhashedPositionY(availableMove);

                //    state.SetCellState(posX, posY, turnColor);
                //    var hashValue = state.GetBoardStateHash();
                //    state.SetCellState(posX, posY, Constants.EMPTY_COLOR);

                //    if (_zobristHash.Contains(hashValue))
                //    {
                //        continue;
                //    }

                //    _zobristHash.Add(hashValue);
                //    validMoves.Add(availableMove);
                //}

                //if (validMoves.Count > 0)
                //{
                    selectedNode.Leafs = new MCTSTreeNode[availableMovesAtSelectedNode.Count];
                    foreach (var availableMove in availableMovesAtSelectedNode)
                    {
                        int posX = _policy.GetUnhashedPositionX(availableMove);
                        int posY = _policy.GetUnhashedPositionY(availableMove);

                        selectedNode.Leafs[index] = new MCTSTreeNode(selectedNode)
                        {
                            MovePosition = new Constants.MovePosition(posX, posY),
                            MoveColor = turnColor
                        };

                        index++;
                    }
                //}
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
                return 2.0f;
            }

            return child.WinProbability + 2.0f * _mctsExplorationConst * Math.Sqrt(2.0f * Math.Log(parent.PlayoutsCount) / child.PlayoutsCount);
        }

        public void SetExplorationConst(float value)
        {
            _mctsExplorationConst = value;
        }

        public Constants.GameResult PlayRandomSimulation(ref IGameBoardState state, ref IncrementalMovesPolicy movesPolicy, int startingTurnColor)
        {
            int currentMoveColor = startingTurnColor;
            Constants.GameResult result = Constants.GameResult.IN_PROGRESS;
            while (true)
            {
                var moves = movesPolicy.GetHashedPositions();
                if (moves.Count == 0)
                {
                    result = Constants.GameResult.TIE;
                    break;
                }

                var moveIndex = _rndGenerator.Next(moves.Count);
                var move = moves.ElementAt(moveIndex);
                var moveX = movesPolicy.GetUnhashedPositionX(move);
                var moveY = movesPolicy.GetUnhashedPositionY(move);

                state.SetCellState(moveX, moveY, currentMoveColor);

                var gameResult = GameResultEstimator.EstimateResult(ref state, moveX, moveY);
                if (gameResult != Constants.GameResult.IN_PROGRESS)
                {
                    if (currentMoveColor == startingTurnColor)
                    {
                        result = gameResult;
                    } else
                    {
                        result = Constants.RevertGameResult(gameResult);
                    }
                    break;
                }

                movesPolicy.Update(moveX, moveY, ref state);
                currentMoveColor = Constants.RotateColor(currentMoveColor);
            }

            return result;
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

        public Constants.MovePosition GetBestMove()
        {
            if (_node.Leafs == null)
            {
                return new Constants.MovePosition();
            }

            var bestNode = _node.Leafs.MaxBy(node => node.PlayoutsCount);
            return bestNode.MovePosition;
        }

        public float GetMoveProbability(int x, int y)
        {
            if (_node.Leafs == null)
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
