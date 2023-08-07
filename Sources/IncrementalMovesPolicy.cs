using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal class IncrementalMovesPolicy
    {
        private readonly int HASH_SEED = 1000;

        HashSet<int> _moves = new HashSet<int>();

        public IncrementalMovesPolicy() { }

        public void Init(IGameBoardState initialState)
        {
            _moves.Clear();

            List<Constants.MovePosition> occupiedCells = new List<Constants.MovePosition>();
            for (int x = 0; x < initialState.GetBoardSize(); x++)
            {
                for (int y = 0; y < initialState.GetBoardSize(); y++)
                {
                    var cellState = initialState.GetCellState(x, y);
                    if (cellState == Constants.CROSS_COLOR || cellState == Constants.ZERO_COLOR)
                    {
                        occupiedCells.Add(new Constants.MovePosition(x, y));
                    }
                }
            }

            if (occupiedCells.Count == 0)
            {
                _moves.Add(GetHashedPosition(initialState.GetBoardSize() / 2, initialState.GetBoardSize() / 2));
                return;
            }


            foreach (var cell in occupiedCells)
            {
                for (int x = -2; x <= 2; x++)
                {
                    for (int y = -2; y <= 2; y++)
                    {
                        int posX = cell.X + x;
                        int posY = cell.Y + y;

                        if (posX < 0 || posY < 0 || posX >= initialState.GetBoardSize() || posY >= initialState.GetBoardSize())
                        {
                            continue;
                        }

                        if (initialState.GetCellState(posX, posY) != Constants.EMPTY_COLOR)
                        {
                            continue;
                        }

                        _moves.Add(GetHashedPosition(posX, posY));
                    }
                }
            }
        }

        public void Update(int x, int y, ref IGameBoardState state)
        {
            int hashedPos = GetHashedPosition(x, y);
            _moves.Remove(hashedPos);

            for (int offsetX = -2; offsetX <= 2; offsetX++)
            {
                for (int offsetY = -2; offsetY <= 2; offsetY++)
                {
                    int posX = x + offsetX;
                    int posY = y + offsetY;

                    if (posX < 0 || posY < 0 || posX >= state.GetBoardSize() || posY >= state.GetBoardSize())
                    {
                        continue;
                    }

                    if (state.GetCellState(posX, posY) != Constants.EMPTY_COLOR)
                    {
                        continue;
                    }

                    _moves.Add(GetHashedPosition(posX, posY));
                }
            }
        }

        public List<int> GetHashedPositions()
        {
            return _moves.ToList();
        }

        public int GetHashedPositionsCount()
        {
            return _moves.Count;
        }

        public int GetHashedPosition(int x, int y)
        {
            return y * HASH_SEED + x;
        }

        public int GetUnhashedPositionX(int hashedPos)
        {
            return hashedPos % HASH_SEED;
        }

        public int GetUnhashedPositionY(int hashedPos)
        {
            return hashedPos / HASH_SEED;
        }
    }
}
