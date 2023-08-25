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
        HashSet<int> _movesHistory = new HashSet<int>();
        int _expansionValue;

        public IncrementalMovesPolicy(int expansionValue) {
            _expansionValue = expansionValue;
        }

        public void Init(IGameBoardState initialState)
        {
            _moves.Clear();
            _movesHistory.Clear();

            for (int x = 0; x < initialState.GetBoardSize(); x++)
            {
                for (int y = 0; y < initialState.GetBoardSize(); y++)
                {
                    var cellState = initialState.GetCellState(x, y);
                    if (cellState == Constants.CROSS_COLOR || cellState == Constants.ZERO_COLOR)
                    {
                        _movesHistory.Add(GetHashedPosition(x, y));
                    }
                }
            }

            if (_movesHistory.Count == 0)
            {
                _moves.Add(GetHashedPosition(initialState.GetBoardSize() / 2, initialState.GetBoardSize() / 2));
                return;
            }


            foreach (var cellHashedPos in _movesHistory)
            {
                var cell = GetMovePositionFromHashed(cellHashedPos);
                for (int x = -_expansionValue; x <= _expansionValue; x++)
                {
                    for (int y = -_expansionValue; y <= _expansionValue; y++)
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
            _movesHistory.Add(hashedPos);

            for (int offsetX = -_expansionValue; offsetX <= _expansionValue; offsetX++)
            {
                for (int offsetY = -_expansionValue; offsetY <= _expansionValue; offsetY++)
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

        public void Update(int x, int y, ref ArrayGameBoardState state)
        {
            int hashedPos = GetHashedPosition(x, y);

            _moves.Remove(hashedPos);
            _movesHistory.Add(hashedPos);

            for (int offsetX = -_expansionValue; offsetX <= _expansionValue; offsetX++)
            {
                for (int offsetY = -_expansionValue; offsetY <= _expansionValue; offsetY++)
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

        public void Remove(int moveX, int moveY, IGameBoardState state)
        {
            int hashedPos = GetHashedPosition(moveX, moveY);
            _movesHistory.Remove(hashedPos);
            _moves.Clear();

            if (_movesHistory.Count == 0)
            {
                _moves.Add(GetHashedPosition(state.GetBoardSize() / 2, state.GetBoardSize() / 2));
                return;
            }

            foreach (var cellHashedPos in _movesHistory)
            {
                var cell = GetMovePositionFromHashed(cellHashedPos);
                for (int x = -_expansionValue; x <= _expansionValue; x++)
                {
                    for (int y = -_expansionValue; y <= _expansionValue; y++)
                    {
                        int posX = cell.X + x;
                        int posY = cell.Y + y;

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

        public Constants.MovePosition GetMovePositionFromHashed(int hashedPosition)
        {
            return new Constants.MovePosition(GetUnhashedPositionX(hashedPosition), GetUnhashedPositionY(hashedPosition));
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
