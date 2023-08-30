using ImGuiNET;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Tensorflow;

namespace GomokuNN.Sources.Estimators
{
    internal class IDDFSEstimator : IGameEstimator
    {
        struct TranspositionTableNode
        {
            public enum FlagTag
            {
                EXACT,
                LOWERBOUND,
                UPPERBOUND
            }

            public FlagTag Flag;
            public bool isValid;
            public float value;
            public int depth;
        }


        ArrayGameBoardState _state = new ArrayGameBoardState(3);
        List<Constants.MovePosition> _movesHistory = new List<Constants.MovePosition>();

        bool _isEvaluationDone = false;
        Constants.MovePosition _bestMove = new Constants.MovePosition();
        bool _isEstimationInProgress = false;

        CancellationTokenSource? _cancellationTokenSource = null;
        CancellationToken? _cancellationToken = null;
        Task? _estimationTask = null;
        Random _rndGenerator = new Random();

        DenseTensor<float> _inputTensor;
        InferenceSession? _inferenceSession = null;

        bool _debugDrawBestMove = false;

        int _currentMoveColor = 0;

        public void InitFromState(IGameBoardState state, int initialColor, int estimatorColor)
        {
            _inferenceSession = CNNModelCache.Instance.LoadModel(CNNHelper.GetCNNPathByGeneration(6) + ".onnx");
            float[] inputData = new float[state.GetBoardSize() * state.GetBoardSize() * 4];
            Array.Fill(inputData, 0);
            _inputTensor = new DenseTensor<float>(inputData, new int[] { 1, 4, state.GetBoardSize(), state.GetBoardSize() });

            var boardSize = state.GetBoardSize();
            _state.Init(boardSize);
            for (int x = 0; x < boardSize; x++)
            {
                for (int y = 0; y < boardSize; y++)
                {
                    int cellState = state.GetCellState(x, y);
                    _state.SetCellState(x, y, cellState);
                }
            }

            _movesHistory.Clear();
            _currentMoveColor = initialColor;
        }


        public Constants.MovePosition GetBestMove()
        {
            if (_bestMove.X == -1)
            {

                IncrementalMovesPolicy movesPolicy = new IncrementalMovesPolicy(1);
                movesPolicy.Init(_state);

                var availableMoves = movesPolicy.GetHashedPositions();
                if (availableMoves.Count == 0)
                {
                    return new Constants.MovePosition();
                }
                int rndIndex = _rndGenerator.Next(availableMoves.Count);
                return movesPolicy.GetMovePositionFromHashed(availableMoves[rndIndex]);
            }

            return _bestMove;
        }

        public int GetCurrentPlayoutsCount()
        {
            return 0;
        }

        public float GetMoveProbability(int x, int y)
        {
            return 0.0f;
        }

        public bool SelectNextNode(int x, int y, int color, bool cleanUpTree = false)
        {
            if (_state.GetCellState(x, y) != Constants.EMPTY_COLOR)
            {
                return false;
            }

            bool isEstimationInProgress = IsEstimationInProgress();
            StopEstimation();

            _state.SetCellState(x, y, color);
            _movesHistory.Add(new Constants.MovePosition(x, y));
            _currentMoveColor = Constants.RotateColor(color);

            if (isEstimationInProgress)
            {
                StartEstimation();
            }

            return true;
        }

        public void SetExplorationConst(float value)
        {
            
        }

        public void StartEstimation()
        {
            if (!HasContiniousEstimationSupport())
            {
                return;
            }
            StopEstimation();

            _cancellationTokenSource = new CancellationTokenSource();
            _cancellationToken = _cancellationTokenSource.Token;

            _isEstimationInProgress = true;
            _estimationTask = Task.Factory.StartNew(() =>
            {
                while (_isEstimationInProgress && !_isEvaluationDone)
                {
                    var watch = System.Diagnostics.Stopwatch.StartNew();
                    Negamax(_state, 3);

                    watch.Stop();
                    Console.WriteLine("Estimation time: " + watch.ElapsedMilliseconds);
                    _isEvaluationDone = true;
                }
            });
        }

        public bool HasContiniousEstimationSupport()
        {
            return true;
        }

        public void EstimateOnce()
        {

        }

        public float Negamax(ArrayGameBoardState boardState, int currentDepth)
        {
            /***
             * 
             function IDDFS(root) is
                for depth from 0 to ∞ do
                    found, remaining ← DLS(root, depth)
                    if found ≠ null then
                        return found
                    else if not remaining then
                        return null

            function DLS(node, depth) is
                if depth = 0 then
                    if node is a goal then
                        return (node, true)
                    else
                        return (null, true)    (Not found, but may have children)

                else if depth > 0 then
                    any_remaining ← false
                    foreach child of node do
                        found, remaining ← DLS(child, depth−1)
                        if found ≠ null then
                            return (found, true)   
                        if remaining then
                            any_remaining ← true    (At least one node found at depth, let IDDFS deepen)
                    return (null, any_remaining)
            */

            ArrayGameBoardState searchState = new ArrayGameBoardState(boardState.GetBoardSize());
            searchState.Copy(boardState);

            IncrementalMovesPolicy movesPolicy = new IncrementalMovesPolicy(2);
            movesPolicy.Init(searchState);

            var zobristHash = new Dictionary<long, TranspositionTableNode>();

            float moveValue = float.MinValue;
            int bestMove = -1;
            var availablePosition = movesPolicy.GetHashedPositions();

            float alpha = float.MinValue;
            float beta = float.MaxValue;

            //Console.WriteLine(new String('=', 10));
            //Console.WriteLine(new String('=', 10));
            //Console.WriteLine(new String('=', 10));

            foreach (var position in availablePosition)
            {
                var boardHash = searchState.GetBoardStateHash();

                //Console.WriteLine(new String('=', 1) + "Start: " + movesPolicy.GetUnhashedPositionX(position) + " " + movesPolicy.GetUnhashedPositionY(position) + " " + _currentMoveColor + " " + currentDepth);

                var value = -SearchIteration(searchState, movesPolicy, zobristHash, movesPolicy.GetUnhashedPositionX(position), movesPolicy.GetUnhashedPositionY(position), -1, -1, alpha, beta, _currentMoveColor, currentDepth);


                var after = searchState.GetBoardStateHash();
                //Console.WriteLine(new String('=', 1) + "End: " + value + " " + (boardHash == after) + " " + alpha + " " + beta);
                if (value >= moveValue)
                {
                    moveValue = value;
                    bestMove = position;
                }

                alpha = Math.Max(alpha, value);
            }

            _bestMove = new Constants.MovePosition(movesPolicy.GetUnhashedPositionX(bestMove), movesPolicy.GetUnhashedPositionY(bestMove));
            return moveValue;
        }

        private float SearchIteration(ArrayGameBoardState boardState, IncrementalMovesPolicy movesPolicy, Dictionary<long, TranspositionTableNode> zobristHash, int x, int y, int prevX, int prevY, float alpha, float beta, int color, int currentDepth)
        {
            if (_cancellationToken != null && _cancellationToken.Value.IsCancellationRequested)
            {
                return 0.0f;
            }

            //Console.WriteLine(new String('+', 6 - currentDepth) + "Start: " + x + " " + y + " " + color + " " + currentDepth);

            if (currentDepth <= 0)
            {

                //Console.WriteLine(new String('-', 6 - currentDepth) + "Exit: " + x + " " + y + " " + color + " " + currentDepth);
                //Console.WriteLine(new String('-', 6 - currentDepth) + "Score: " + 0);

                float[] inputData = new float[boardState.GetBoardSize() * boardState.GetBoardSize() * 4];
                int boardOffset = _state.GetBoardSize() * _state.GetBoardSize();

                for (int i = 0; i < _state.GetBoardSize() * _state.GetBoardSize(); i++)
                {
                    var value = _state.GetRawCellState(i);
                    _inputTensor.SetValue(i, value == Constants.CROSS_COLOR ? 1 : 0);
                    _inputTensor.SetValue(i + boardOffset, value == Constants.ZERO_COLOR ? 1 : 0);
                    _inputTensor.SetValue(i + boardOffset * 2, 0);
                    _inputTensor.SetValue(i + boardOffset * 3, color == Constants.CROSS_COLOR ? 1 : 0);
                }

                _inputTensor.SetValue(boardOffset * 2 + boardState.GetPositionHash(prevX, prevY), 1);

                var modelInput = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("input_layer", _inputTensor) };
                var output = _inferenceSession.Run(modelInput).ToArray();
                var valueOutput = output[1].AsTensor<float>();

                return valueOutput[0];
            }

            boardState.SetCellState(x, y, color);
            movesPolicy.Update(x, y, ref boardState);

            var boardHash = boardState.GetBoardStateHash();
            //if (zobristHash.ContainsKey(boardHash))
            //{
            //    boardState.UndoSetCellState(x, y);
            //    movesPolicy.Remove(x, y, boardState);

            //    return zobristHash[boardHash];
            //}

            var result = GameResultEstimator.EstimateResult(boardState, x, y, false);
            if (result == Constants.GameResult.WIN)
            {
                boardState.UndoSetCellState(x, y);
                movesPolicy.Remove(x, y, boardState);

                var score = -10.0f * currentDepth * (color == _currentMoveColor ? 1.0f : 1.0f);

                //Console.WriteLine(new String('-', 6 - currentDepth) + "Exit: " + x + " " + y + " " + color + " " + currentDepth);
                //Console.WriteLine(new String('-', 6 - currentDepth) + "Score: " + score);
                return score;
            }

            float moveValue = float.MinValue;
            var availablePosition = movesPolicy.GetHashedPositions();
            foreach (var position in availablePosition)
            {
                //if (position != 5006 && position != 6008 && position != 6007 && position != 10010 && position != 10011 && position != 9010 && position != 11011 && position != 4005)
                //{
                //    continue;
                //}

                var value = -SearchIteration(boardState, movesPolicy, zobristHash, movesPolicy.GetUnhashedPositionX(position), movesPolicy.GetUnhashedPositionY(position), x, y, -beta, -alpha, Constants.RotateColor(color), currentDepth - 1);
                moveValue = Math.Max(moveValue, value);

                alpha = Math.Max(alpha, moveValue);
                if (alpha >= beta)
                {
                    break;
                }
            }

            //Debug.Assert(boardHash == boardState.GetBoardStateHash());
            //if (!zobristHash.ContainsKey(boardHash) && zobristHash.Count < int.MaxValue - 1)
            //{
            //    zobristHash.Add(boardHash, moveValue);
            //}

            boardState.UndoSetCellState(x, y);
            movesPolicy.Remove(x, y, boardState);

            //Console.WriteLine(new String('-', 6 - currentDepth) + "Exit: " + x + " " + y + " " + color + " " + currentDepth);
            //Console.WriteLine(new String('-', 6 - currentDepth) + "Score: " + moveValue);

            return moveValue;
        }

        public bool IsEstimationInProgress()
        {
            return _isEstimationInProgress;
        }

        public void StopEstimation()
        {
            _cancellationTokenSource?.Cancel();

            _isEvaluationDone = false;
            _isEstimationInProgress = false;
            _estimationTask?.Wait();
            _estimationTask = null;

            _cancellationTokenSource?.Dispose();
            _cancellationTokenSource = null;
            _cancellationToken = null;
        }

        public void FillTrainingSamples(ref List<TrainingSample> samples, int winnerColor, ref HashSet<long> knownPositions)
        {

        }

        public List<TrainingSample> GetTrainingSamples(int winnerColor, ref HashSet<long> knownPositions)
        {
            throw new NotImplementedException();
        }

        public void DebugMenuDraw(ref GameBoard gameBoard)
        {
            //ImGui.Text("Playouts count: " + _node.PlayoutsCount.ToString());
            //ImGui.Text("Node win probability: " + _node.Leafs.Average(leaf => leaf.WinProbability).ToString());

            //ImGui.Checkbox("Show probabilities", ref _showAIProbabilities);
            ImGui.Checkbox("Show best move", ref _debugDrawBestMove);

            if (!_isEvaluationDone)
            {
                ImGui.Text("Estimation in progress...");
            }
            else
            {
                ImGui.Text("Estimation is done.");
            }
        }

        public void OnDrawCell(int x, int y, ref RenderHelper renderHelper)
        {
            if (_debugDrawBestMove)
            {
                var bestMove = GetBestMove();
                if (x == bestMove.X && y == bestMove.Y)
                {
                    renderHelper.DrawAINextMoveBoardCell(_currentMoveColor, x, y);
                }
            }
        }
    }
}
