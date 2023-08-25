using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources.Estimators
{
    internal class PatternFinder
    {
        private struct Direction
        {
            public int x; public int y;
        }

        private const int HORIZONTAL = 0;
        private const int VERTICAL = 1;
        private const int DIAGONAL_1 = 2;
        private const int DIAGONAL_2 = 3;

        private readonly Dictionary<int, Direction> directions = new Dictionary<int, Direction>()
        {
            {0, new Direction { x = 1, y = 0 } },
            {1, new Direction { x = 0, y = 1 } },
            {2, new Direction { x = 1, y = 1 } },
            {3, new Direction { x = -1, y = 1 } }
        };

        public void Update(IGameBoardState boardState, int x, int y, int color)
        {
            foreach (var dir in directions.Values)
            {

                int occupiedCells = 1;
                int value = 1;
                int rightOffset = 1;
                while (boardState.GetCellState(x + dir.x * rightOffset, y + dir.y * rightOffset) == color && occupiedCells < Constants.CELLS_IN_ROW_TO_WIN)
                {
                    value = (value << 1) | 1;
                    rightOffset++;
                    occupiedCells++;
                }

                int leftOffset = -1;
                while (boardState.GetCellState(x + dir.x * leftOffset, y + dir.y * leftOffset) == color && occupiedCells < Constants.CELLS_IN_ROW_TO_WIN)
                {
                    value = (value << 1) | 1;
                    leftOffset--;
                    occupiedCells++;
                }

                if (occupiedCells >= Constants.CELLS_IN_ROW_TO_WIN)
                {
                    // update pattern
                    //
                    continue;
                }

                int maxOffset = Constants.CELLS_IN_ROW_TO_WIN - occupiedCells;
                for (int offset = 0; offset < maxOffset; offset++)
                {
                    var cellValue = boardState.GetCellState(x + dir.x * (rightOffset + offset), y + dir.y * (rightOffset + offset));
                    if (cellValue == color) 
                    {
                        
                    } 
                    else if (cellValue == Constants.EMPTY_COLOR)
                    {

                    }
                    else
                    {
                        break;
                    }
                }

            }
        }

        public void Undo(IGameBoardState boardState, int x, int y)
        {

        }
    }
}
