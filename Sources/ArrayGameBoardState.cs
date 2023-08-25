using ImGuiNET;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal class ArrayGameBoardState : IGameBoardState
    {
        private int[] _state = new int[1];
        private int _size;

        private long _zobristHash = 0;

        public ArrayGameBoardState(int size)
        {
            Init(size);
        }

        public void Init(int size)
        {
            _state = new int[size * size];
            for (int i = 0; i < size; i++)
            {
                _state[i] = 0;
            }
            _size = size;

            _zobristHash = 0;
        }

        public int GetPositionHash(int x, int y)
        {
            return y * _size + x;
        }

        public void SetCellState(int x, int y, int color)
        {
            int hashedPos = GetPositionHash(x, y);
            _state[hashedPos] = color;

            _zobristHash ^= ZobristHash.GetCellZobristHash(_size, x, y, color);
        }

        public void UndoSetCellState(int x, int y)
        {
            int hashedPos = GetPositionHash(x, y);
            int stateColor = _state[hashedPos];

            _state[hashedPos] = 0;
            _zobristHash ^= ZobristHash.GetCellZobristHash(_size, x, y, stateColor);
        }

        public long GetBoardStateHash()
        {
            return _zobristHash;
        }

        public int GetRawCellState(int index)
        {
            return _state[index];
        }
        public int GetCellState(int x, int y)
        {
            if (y < 0 || y >= _size
                || x < 0 || x >= _size)
            {
                return Constants.NULL_COLOR;
            }

            int hashedPos = GetPositionHash(x, y);
            if (hashedPos < 0 || hashedPos > _state.Length)
            {
                return Constants.NULL_COLOR;
            }


            var state = _state[hashedPos];
            return state;
        }

        public int GetBoardSize()
        {
            return _size;
        }

        public void Copy(ArrayGameBoardState other)
        {
            if (other.GetBoardSize() != _size)
            {
                throw new Exception();
            }

            other._state.CopyTo(_state, 0);
            _zobristHash = other._zobristHash;
        }
    }
}
