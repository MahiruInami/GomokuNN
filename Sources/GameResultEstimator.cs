using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal class GameResultEstimator
    {
        public static Constants.GameResult EstimateResult(ref IGameBoardState boardState, int lastX, int lastY, bool estimateForTie = false)
        {
            int targetColor = boardState.GetCellState(lastX, lastY);
            int piecesInRow = 1;
            for (int left = 1 ; ;left++)
            {
                if (boardState.GetCellState(lastX - left, lastY) != targetColor)
                {
                    break;
                }

                piecesInRow++;
            }
            for (int right = 1; ; right++)
            {
                if (boardState.GetCellState(lastX + right, lastY) != targetColor)
                {
                    break;
                }

                piecesInRow++;
            }

            if (piecesInRow >= Constants.CELLS_IN_ROW_TO_WIN)
            {
                return Constants.GameResult.WIN;
            }

            piecesInRow = 1;
            for (int up = 1; ; up++)
            {
                if (boardState.GetCellState(lastX, lastY + up) != targetColor)
                {
                    break;
                }

                piecesInRow++;
            }
            for (int down = 1; ; down++)
            {
                if (boardState.GetCellState(lastX, lastY - down) != targetColor)
                {
                    break;
                }

                piecesInRow++;
            }

            if (piecesInRow >= Constants.CELLS_IN_ROW_TO_WIN)
            {
                return Constants.GameResult.WIN;
            }

            piecesInRow = 1;
            for (int leftUp = 1; ; leftUp++)
            {
                if (boardState.GetCellState(lastX - leftUp, lastY + leftUp) != targetColor)
                {
                    break;
                }

                piecesInRow++;
            }
            for (int rightDown = 1; ; rightDown++)
            {
                if (boardState.GetCellState(lastX + rightDown, lastY - rightDown) != targetColor)
                {
                    break;
                }

                piecesInRow++;
            }

            if (piecesInRow >= Constants.CELLS_IN_ROW_TO_WIN)
            {
                return Constants.GameResult.WIN;
            }

            piecesInRow = 1;
            for (int leftDown = 1; ; leftDown++)
            {
                if (boardState.GetCellState(lastX - leftDown, lastY - leftDown) != targetColor)
                {
                    break;
                }

                piecesInRow++;
            }
            for (int rightUp = 1; ; rightUp++)
            {
                if (boardState.GetCellState(lastX + rightUp, lastY + rightUp) != targetColor)
                {
                    break;
                }

                piecesInRow++;
            }

            if (piecesInRow >= Constants.CELLS_IN_ROW_TO_WIN)
            {
                return Constants.GameResult.WIN;
            }

            if (estimateForTie)
            {
                var movesPolicy = new IncrementalMovesPolicy();
                movesPolicy.Init(boardState);

                if (movesPolicy.GetHashedPositionsCount() == 0)
                {
                    return Constants.GameResult.TIE;
                }
            }

            return Constants.GameResult.IN_PROGRESS;
        }
    }
}
