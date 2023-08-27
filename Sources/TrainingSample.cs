using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal class TrainingSample
    {
        public int[] input;
        public float[] networkOutput;
        public float valueOutput;

        public TrainingSample()
        {

        }

        public TrainingSample(IGameBoardState state, Constants.MovePosition nextMove, Constants.MovePosition lastMove, int color, float score)
        {
            var rndProvider = new Random();
            int dataOffset = state.GetBoardSize() * state.GetBoardSize();
            input = new int[state.GetBoardSize() * state.GetBoardSize() * 4];
            networkOutput = new float[state.GetBoardSize() * state.GetBoardSize()];
            valueOutput = score;

            for (int y = 0; y < state.GetBoardSize(); y++)
            {
                for (int x = 0; x < state.GetBoardSize(); x++)
                {
                    int cellState = state.GetCellState(x, y);

                    int moveIndex = y * state.GetBoardSize() + x;
                    input[moveIndex] = cellState == Constants.CROSS_COLOR ? 1 : 0;
                    input[dataOffset + moveIndex] = cellState == Constants.ZERO_COLOR ? 1 : 0;

                    input[dataOffset * 3 + moveIndex] = color == Constants.CROSS_COLOR ? 1 : 0;

                    networkOutput[moveIndex] = (nextMove.X >= 0 && nextMove.Y >= 0) ? (float)rndProvider.NextDouble() / 1000.0f : 0.0f;
                }
            }

            if (lastMove.X >= 0 && lastMove.Y >= 0)
            {
                input[dataOffset * 2 + lastMove.Y * state.GetBoardSize() + lastMove.X] = 1;
            }

            if (nextMove.X >= 0 && nextMove.Y >= 0)
            {
                int nextMoveIndex = nextMove.Y * state.GetBoardSize() + nextMove.X;
                networkOutput[nextMoveIndex] = 1.0f;
            }
        }

        public void SetPolicyOutputForMove(int boardSize, int x, int y, float value)
        {
            if (x < 0 || y < 0 || x >= boardSize || y >= boardSize)
            {
                return;
            }

            int moveIndex = y * boardSize + x;
            networkOutput[moveIndex] = value;
        }

        public void SetPolicyOutputByIndex(int moveIndex, float value)
        {
            networkOutput[moveIndex] = value;
        }


        public static List<TrainingSample> CreateFromGameHistory(GameHistory gameHistory, ref HashSet<long> knownPositions)
        {
            List<TrainingSample> samples = new List<TrainingSample>();

            FillFromGameHistory(ref samples, gameHistory, ref knownPositions);

            return samples;
        }

        public static void FillFromGameHistory(ref List<TrainingSample> samples, GameHistory gameHistory, ref HashSet<long> knownPositions)
        {
            var board = new ArrayGameBoardState(Constants.DEFAULT_BOARD_SIZE);

            int color = gameHistory.startingColor;

            Constants.MovePosition prevMovePosition = new Constants.MovePosition();
            for (int i = 0; i < gameHistory.moves.Count; i++)
            {
                var move = gameHistory.moves[i];

                long zobristHash = board.GetBoardStateHash();
                if (i > 6 && knownPositions.Contains(zobristHash))
                {
                    samples.Add(new TrainingSample(board, move, prevMovePosition, color, color == gameHistory.winningColor ? 1.0f : -1.0f));
                }

                board.SetCellState(move.X, move.Y, color);
                knownPositions.Add(board.GetBoardStateHash());

                if (i == gameHistory.moves.Count - 1)
                {
                    samples.Add(new TrainingSample(board, move, move, color, color == gameHistory.winningColor ? 1.0f : -1.0f));
                }

                color = Constants.RotateColor(color);
                prevMovePosition = move;
            }
        }
    }
}
