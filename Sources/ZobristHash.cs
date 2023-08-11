using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal class ZobristHash
    {
        private static long[,] _hashTable;

        public static void Init(int size)
        {
            var rnd = new Random();
            _hashTable = new long[size * size, 3];
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    _hashTable[i * size + j, 0] = 0;
                    _hashTable[i * size + j, 1] = rnd.NextInt64();
                    _hashTable[i * size + j, 2] = rnd.NextInt64();
                }
            }
        }

        public static long GetCellZobristHash(int size, int x, int y, int color)
        {
            return _hashTable[y * size + x, color];
        }
    }
}
